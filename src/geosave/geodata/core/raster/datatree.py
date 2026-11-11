"""Read raster trees through the `gs` DataTree accessor.

Raster Datasets may be nested in organizational groups, including data-bearing
parents and the root. Each raster owns its coordinates and spatial reference.
Grid names are labels, not identities. Publication splits each node into Items and assets;
publication assigns Collection membership through explicit subtree paths.

Examples:
    Co-registered, so every grid node shares one grid and `gs.geobox` reads it::

        >>> scene = xr.DataTree.from_dict({"sentinel-2-l2a/r10m": optical, "dem/r10m": dem})
        /
        ├── /sentinel-2-l2a
        │   └── /r10m                red, nir            (time, y, x)
        └── /dem
            └── /r10m                elevation           (y, x)
        >>> scene.gs.geobox == optical.gs.geobox
        True

    Native resolutions, so each grid node keeps its own grid and `gs.geobox` is
    None::

        >>> sentinel = xr.DataTree.from_dict({"s2/r10m": bands_10m, "s2/r20m": bands_20m})
        >>> sentinel.gs.geobox is None
        True
        >>> sentinel.gs.to_dataset(grid="s2/r10m").gs.resolution.x
        10.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING, Unpack, cast

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import numpy as np
import pandas as pd
import xarray as xr
import geosave.geodata.attrs as attrs
from odc.geo.geobox import GeoBox
from upath import UPath


from .base import GeoRaster
from geosave.geodata.conventions import TIME_COORDINATE
from geosave.geodata.errors import RasterValidationError

if TYPE_CHECKING:
    import geopandas as gpd
    import pystac
    from collections.abc import Callable
    from odc.geo.geom import Geometry
    import holoviews as hv
    import torch
    from numpy.typing import DTypeLike
    from odc.geo.gcp import GCPGeoBox

    from geosave.geodata import Dataset, DataTree, GeoDataFrame
    from geosave.geodata.core.region import GeoRegion
    from dask.delayed import Delayed
    from geosave.geodata.io.formats.geotiff import COGWriteOptions
    from geosave.geodata.io.formats.netcdf import NetCDFEngine
    from geosave.geodata.io.formats.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.formats.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from .transform.warp import Resampling
    from xarray.core.types import CombineAttrsOptions
    from geosave.geodata.utils.datetime import DateRange


class GeoDataTree(GeoRaster["DataTree"]):
    """Read and persist one raster-tree DataTree.

    The root carries the tree's own attrs, a collection node its STAC
    metadata, and each grid node's raster its own, read through
    `rasters["<collection>/<res>"].gs`.

    Args:
        data: DataTree of collections over grid nodes, `/{collection}/{grid}`,
            or a scene tree whose children are grid nodes.

    Examples:
        >>> scene.gs.groups
        ('sentinel-2-l2a', 'dem')
        >>> scene.gs.geobox.shape
        (512, 512)
    """

    @property
    def time_coordinate(self) -> xr.DataArray | None:
        """Read this node's native temporal coordinate, including inheritance."""
        return self._data.to_dataset(inherit="all_coords").gs.time_coordinate

    @property
    def spatial_dims(self) -> tuple[str, str] | None:
        """Return the common spatial axes, or None for missing or differing values."""
        shared = None
        for dataset in self.datasets.values():
            current = dataset.gs.spatial_dims
            if current is None or (shared is not None and current != shared):
                return None
            shared = current
        return shared

    @property
    def geobox(self) -> GeoBox | GCPGeoBox | None:
        """Return the common grid, or None for missing or differing values."""
        shared = None
        for dataset in self.datasets.values():
            current = dataset.gs.geobox
            if current is None or (shared is not None and current != shared):
                return None
            shared = current
        return shared

    def validate(
        self,
        *,
        georeferenced: bool = False,
        affine: bool = False,
        same_grid: bool = False,
        same_dims: bool = False,
        same_footprint: bool = False,
        shared_time: bool = False,
    ) -> None:
        """Validate each raster and optionally compare spatial structure.

        Unreferenced pixels and different grids are valid by default.
        Spatial coordinates and existing placement are always checked without
        reading pixels. Empty inputs fail.

        Args:
            georeferenced: Require placement and a CRS for every raster.
            affine: Require one affine `GeoBox` on spatial axes shared by all
                nodes, so `gs.geobox` is a GeoBox. Implies `same_grid`
                and `same_dims`.
            same_grid: Require known, equal ODC grids across all nodes.
            same_dims: Require matching spatial axis names across all nodes.
            same_footprint: Require every node to cover the same ground, on
                any grids: each footprint, its declared extent or else its
                grid extent, lies within the others grown by one pixel of the
                coarsest grid.
            shared_time: Require nodes below this one to define no `time` of
                their own, so they inherit one timeline that xarray keeps
                aligned. A lone data node may carry its own.

        Raises:
            RasterValidationError: A raster is invalid or a requested
                guarantee fails. The error identifies the node and variable.

        Examples:
            >>> tree.gs.validate(georeferenced=True, same_grid=True)
        """
        same_grid = same_grid or affine
        same_dims = same_dims or affine
        datasets = self.datasets
        self._validate_nonempty(datasets)
        reference = None
        for path, dataset in datasets.items():
            try:
                dataset.gs.validate(
                    georeferenced=georeferenced,
                    affine=affine,
                    same_grid=same_grid,
                    same_dims=same_dims,
                )
                if reference is None:
                    reference = dataset
                    continue
                if same_dims:
                    self._validate_same_dims(dataset, reference)
                if same_grid:
                    self._validate_same_grid(dataset, reference)
            except RasterValidationError as error:
                error.path = path
                raise
        if same_footprint:
            self._validate_same_footprint(datasets)
        if shared_time and len(datasets) > 1:
            for path, node in self._data.subtree_with_keys:
                # This node is where shared time belongs; only those below it are checked.
                if path == "." or path not in datasets:
                    continue
                if TIME_COORDINATE in node.to_dataset(inherit=False).coords:
                    raise RasterValidationError(
                        "This node defines its own time. Move the time its sibling "
                        "nodes share onto their common parent, which they then inherit.",
                        path=path,
                    )

    def _validate_same_footprint(self, datasets: Mapping[str, Dataset]) -> None:
        """Require every node's footprint to match the first within one coarse pixel."""
        footprints = {}
        tolerance = 0.0
        for path, dataset in datasets.items():
            footprints[path] = dataset.gs.region.extent
            grid = dataset.gs.geobox
            if grid is not None:
                resolution = grid.resolution
                tolerance = max(tolerance, abs(resolution.x), abs(resolution.y))
        reference_path, reference = next(iter(footprints.items()))
        # The tolerance is in the first node's units, where its CRS measures.
        for path, footprint in footprints.items():
            other = footprint.to_crs(reference.crs)
            covers = reference.buffer(tolerance).contains(other)
            covered = other.buffer(tolerance).contains(reference)
            if not (covers and covered):
                raise RasterValidationError(
                    f"This node covers different ground than {reference_path!r}. "
                    "Select nodes of one place, or crop them to a shared region.",
                    path=path,
                )

    def _validate_nonempty(self, datasets: Mapping[str, Dataset]) -> None:
        """Require at least one data-bearing node."""
        if not datasets:
            raise RasterValidationError(
                "There are no raster variables to validate. Select or load raster data first."
            )

    def _validate_same_dims(self, dataset: Dataset, reference: Dataset) -> None:
        """Require spatial axis names to match the first node."""
        if dataset.gs.spatial_dims != reference.gs.spatial_dims:
            raise RasterValidationError(
                "Nodes use different spatial dimensions. Rename their spatial axes "
                "consistently, for example with to_yx, before combining them.",
                variable=str(next(iter(dataset.data_vars))),
            )

    def _validate_same_grid(self, dataset: Dataset, reference: Dataset) -> None:
        """Require equal known ODC placement."""
        if dataset.gs.geobox != reference.gs.geobox:
            raise RasterValidationError(
                "Nodes do not share one grid. Pass grid= to to_dataset with a node "
                "path or GeoBox to reproject them onto it.",
                variable=str(next(iter(dataset.data_vars))),
            )

    @property
    def groups(self) -> tuple[str, ...]:
        """Return the collection names in tree order, or a scene's grid node names.

        Returns:
            Direct child names.
        """
        return tuple(self._data.children)

    @property
    def variables(self) -> tuple[str, ...]:
        """Name every grid node's data variable by its path, as `"s2/r10m/red"`."""
        return tuple(
            f"{group}/{variable}"
            for group, raster in self.datasets.items()
            for variable in raster.gs.variables
        )

    @property
    def timespan(self) -> DateRange | None:
        """Read inclusive temporal coverage across every group.

        Groups need not share a cadence, so the tree runs from the earliest
        instant any group reaches to the latest, a timeless group widening
        nothing. Read one group's own span with `rasters["<group>"].gs.timespan`.

        Returns:
            First and last covered instant, or None where no group is dated.

        Examples:
            >>> scene.gs.datasets["dem"].gs.timespan is None
            True
            >>> scene.gs.timespan == scene.gs.datasets["sentinel-2-l2a"].gs.timespan
            True
        """
        spans = [
            span
            for span in (raster.gs.timespan for raster in self.datasets.values())
            if span is not None
        ]
        if not spans:
            return None
        return (min(start for start, _ in spans), max(end for _, end in spans))

    @property
    def times(self) -> pd.DatetimeIndex | None:
        """Read every instant any group reaches, in order.

        Returns:
            Sorted labels across the groups, each instant once, or None where
            no group is dated.
        """
        dated = [
            times
            for times in (raster.gs.times for raster in self.datasets.values())
            if times is not None
        ]
        if not dated:
            return None
        return pd.DatetimeIndex(dated[0].append(dated[1:]).unique().sort_values())

    def write_extent(self, extent: Geometry | None) -> DataTree:
        """Record the declared extent throughout the tree without changing pixels.

        Args:
            extent: CRS-aware geometry to preserve, or None to clear it.

        Returns:
            Tree with the declaration on each node and its spatial variables,
            so detached nodes and selected bands retain their extent.

        Raises:
            ValueError: The extent is invalid, empty, lacks a CRS or is not 2D.
        """
        return cast(
            "DataTree",
            self._data.map_over_datasets(
                lambda dataset: dataset.gs.write_extent(extent)
            ),
        )

    @property
    def region(self) -> GeoRegion:
        """Return the declared extent or union of raster extents, with their time span.

        Different grids need no alignment. This inspects metadata without
        reading pixels; missing geographic coverage raises ValueError.
        """
        from geosave.geodata import attrs
        from geosave.geodata.core.region import GeoRegion

        extent = attrs.Extent.from_attrs(self._data.attrs)
        if extent is not None:
            return GeoRegion(extent.to_geometry(), timespan=self.timespan)
        if not self._data.children:
            return self._data.to_dataset(inherit="all_coords").gs.region
        regions = []
        for raster in self.datasets.values():
            regions.append(raster.gs.region)
        if not regions:
            raise ValueError("an empty tree has no geographic coverage")
        first = regions[0]
        geometry = first.extent
        for member in regions[1:]:
            geometry = geometry.union(member.extent.to_crs(first.crs))
        return GeoRegion(geometry, timespan=self.timespan)

    @property
    def datasets(self) -> dict[str, Dataset]:
        """Read every data-bearing node as a Dataset, keyed by its relative path.

        Returns:
            {
                grid node path, as `"s2/r10m"`: that node's raster, carrying
                    the grid and time coordinates it inherits from above,
            }

        Examples:
            Rebuild a tree to add a collection:

            >>> list(scene.gs.datasets)
            ['sentinel-2-l2a/r10m', 'dem/r10m']
            >>> xr.merge([scene, xr.DataTree.from_dict({"ndvi/r10m": ndvi})]).gs.groups
            ('sentinel-2-l2a', 'dem', 'ndvi')
        """
        rasters = {}
        for path, node in self._data.subtree_with_keys:
            if not node.data_vars:
                continue
            # .dataset would return a DatasetView whose attrs still write through here.
            dataset = node.to_dataset(inherit="all_coords")
            # DataTree inherits coordinates, not their bounds data variables.
            # Resolve only the bounds explicitly referenced by inherited coords.
            for coordinate in list(dataset.coords.values()):
                reference = coordinate.attrs.get(
                    "bounds", coordinate.encoding.get("bounds")
                )
                if reference is None or reference in dataset.variables:
                    continue
                for ancestor in node.parents:
                    parent = ancestor.to_dataset(inherit=False)
                    if reference in parent.variables:
                        dataset = dataset.assign_coords({reference: parent[reference]})
                        break
            if dataset.gs.raster_variables:
                rasters[path] = cast("Dataset", dataset)
        return rasters

    def group_grid(self) -> DataTree:
        """Merge sibling raster nodes on one grid into one node named by its pixel size.

        Reads coordinates only. Nodes without data keep their coordinates and
        attrs, so asset nodes still inherit what their collection node states.

        Returns:
            Tree with one data node per grid under each parent, named like
            `r10m`.

        Raises:
            ValueError: Two different grids under one parent share a pixel
                size, or merged nodes name one variable twice.

        Examples:
            >>> list(tree.gs.datasets)
            ['s2/B04', 's2/B08', 's2/B11']
            >>> list(tree.gs.group_grid().gs.datasets)
            ['s2/r10m', 's2/r20m']
        """
        from geosave.geodata.io.paths import format_resolution

        nodes = {}
        grids = {}
        for path, node in self._data.subtree_with_keys:
            own = node.to_dataset(inherit=False)
            if path == "." or not own.data_vars:
                nodes["/" if path == "." else path] = own
                continue
            geobox = node.to_dataset(inherit="all_coords").gs.geobox
            name = format_resolution(geobox)
            parent = path.rsplit("/", 1)[0] if "/" in path else None
            target = name if parent is None else f"{parent}/{name}"
            if target not in grids:
                grids[target] = geobox
                nodes[target] = own
                continue
            if grids[target] != geobox:
                raise ValueError(
                    f"{target!r} would hold two different grids; group them by hand"
                )
            repeated = set(nodes[target].data_vars) & set(own.data_vars)
            if repeated:
                raise ValueError(f"{target!r} would hold {sorted(repeated)} twice")
            nodes[target] = xr.merge(
                [nodes[target], own], join="exact", combine_attrs="drop_conflicts"
            )
        return cast("DataTree", xr.DataTree.from_dict(nodes))

    def _map_variables(
        self, change: Callable[[xr.DataArray], xr.DataArray]
    ) -> DataTree:
        """Apply one per-variable change to every raster node's variables."""
        # No pixel moves, so the tree keeps its root and its coordinates.
        tree = self._data.map_over_datasets(
            lambda raster: raster.gs._map_variables(change)
        )
        return cast("DataTree", tree)

    def crop(
        self, region: Geometry | gpd.GeoSeries | gpd.GeoDataFrame, *, mask: bool = True
    ) -> DataTree:
        """Cut every raster node down to the extent of a region.

        Args:
            region: Geometry, or geometries, to cut against, reprojected onto
                each node's CRS where they sit in another.
            mask: Also make nodata the pixels outside the geometries.

        Returns:
            New tree; collection nodes, which hold no pixels, are unchanged.

        Examples:
            >>> tree.gs.crop(field_boundaries)
        """
        tree = self._data.map_over_datasets(
            lambda raster: (
                raster.gs.crop(region, mask=mask)
                if raster.gs.raster_variables
                else raster
            )
        )
        return cast("DataTree", tree)

    def to_dataset(
        self,
        grid: str | GeoBox | None = None,
        *,
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        combine_attrs: CombineAttrsOptions = "drop_conflicts",
    ) -> Dataset:
        """Merge this node and its raster descendants into one Dataset.

        Without `grid`, the nodes must already share one affine georeferenced
        grid. With it, every node is first reprojected onto that grid.
        Variable names must be distinct and coordinates compatible.

        Args:
            grid: Node path relative to this node, whose grid every node lands
                on, or an exact GeoBox. None keeps the shared grid.
            resampling: One kernel, or a mapping from variable name to kernel
                where `"*"` answers the rest. Used only with `grid`.
            combine_attrs: Native attr policy or callable; omitted keys warn.

        Returns:
            Dataset with merged variables. Each raster's attrs take precedence
            over this node's attrs.

        Raises:
            KeyError: `grid` names no raster node, or a `resampling` mapping
                names neither a variable nor `"*"`.
            RasterValidationError: No raster data exists, a raster has no usable
                grid, or, without `grid`, grids differ. The error names the node.
            ValueError: Coordinates or dimensions differ, variable names
                collide, or `resampling` would blend class codes.

        Examples:
            >>> scene = tree["s2"].gs.to_dataset(grid="r10m", resampling={"scl": "mode", "*": "bilinear"})
            >>> scene.gs.resolution.x
            10.0
        """
        from functools import partial
        from geosave.geodata.conventions import CRS_COORDINATE

        rasters = self.datasets
        try:
            self._validate_nonempty(rasters)
        except RasterValidationError as error:
            error.operation = "Cannot combine the selected rasters into a Dataset"
            raise
        if grid is not None:
            target = grid
            if isinstance(grid, str):
                rasters[grid].gs.validate(georeferenced=True, affine=True)
                target = cast("GeoBox", rasters[grid].gs.geobox)
            for path, raster in rasters.items():
                warped = raster.gs.reproject(target, resampling=resampling)
                rasters[path] = cast("Dataset", warped)
        try:
            for path, raster in rasters.items():
                raster.gs.validate(georeferenced=True, affine=True)
            first = next(iter(rasters.values()))
            for path, raster in rasters.items():
                if raster.gs.geobox != first.gs.geobox:
                    raise RasterValidationError(
                        "Nodes do not share one grid. Pass grid= with a node path "
                        "or GeoBox to reproject them onto it.",
                        path=path,
                    )
        except RasterValidationError as error:
            error.operation = "Cannot combine the selected rasters into a Dataset"
            raise
        nodes = []
        names = set()
        dimensions = first[first.gs.raster_variables[0]].dims
        for path, raster in rasters.items():
            duplicates = names.intersection(raster.gs.raster_variables)
            if duplicates:
                raise ValueError(
                    f"duplicate variables {sorted(duplicates)} in {path!r}"
                )
            names.update(raster.gs.raster_variables)
            if any(
                raster[name].dims != dimensions for name in raster.gs.raster_variables
            ):
                raise ValueError(
                    f"{path!r} requires equal variable dimensions; no broadcasting"
                )
            # Grid equality was checked above. Normalize equivalent CRS placeholders.
            if CRS_COORDINATE in raster.coords and CRS_COORDINATE in first.coords:
                raster = raster.assign_coords(
                    {CRS_COORDINATE: first[CRS_COORDINATE].variable}
                )
            raster = raster.gs.write_crs()
            if not raster.coords.to_dataset().equals(first.coords.to_dataset()):
                raise ValueError(
                    f"{path!r} requires equal coordinates; select observations first"
                )
            nodes.append(raster.assign_attrs({**self._data.attrs, **raster.attrs}))
        merged = xr.merge(
            nodes,
            join="exact",
            compat="equals",
            combine_attrs=partial(attrs.combine_attrs, policy=combine_attrs),
        )
        return cast("Dataset", merged)

    def plot(self, *, cols: int = 1, xlabel: str | None = None) -> hv.Layout:
        """Draw every group down the page.

        Each group draws through `Dataset.gs.plot`. Its group name, time, and
        location appear below the axes. A group spanning `time` contributes
        one panel per step.

        Args:
            cols: Panels per row. Defaults to one, stacking them in a column.
            xlabel: Location caption below each group's name. None uses each
                group's reverse-geocoded location; an empty string suppresses it.

        Returns:
            Layout of every panel, or the sole panel when the tree holds one
            group with no time axis.

        Raises:
            ValueError: A group names nothing to draw; name its channels with
                `gs.write_rgb` first or plot it alone with
                `tree["<collection>"].gs.to_dataset().gs.plot()`.

        Examples:
            >>> hv.save(scene.gs.plot(), "scene.png")
        """
        import holoviews as hv

        panels: list[hv.Element] = []
        for name, raster in self.datasets.items():
            place = raster.gs.region.locate() if xlabel is None else None
            where = place.to_address() if place is not None else xlabel
            caption = f"{name}\n{where}" if where else name
            drawn = raster.gs.plot(xlabel=caption)
            panels.extend(drawn.values() if isinstance(drawn, hv.Layout) else [drawn])

        layout = hv.Layout(panels).cols(cols)
        # Room between columns for each panel's colorbar.
        layout.opts(hspace=0.9)
        return layout

    def to_numpy(self, *, dtype: DTypeLike | None = None) -> dict[str, np.ndarray]:
        """Stack each group's variables into one model-input array.

        Groups share a grid but not their variables, non-spatial axes, or
        dtypes, so each keeps its own array rather than being joined. Cast a
        single group afterwards, which is one call on the array it returns.

        Args:
            dtype: Cast applied to every group. None keeps each group's source
                dtype, which every variable of that group must then share.

        Returns:
            {
                grid node path: array shaped `(*axes, band, y, x)`,
            }

        Raises:
            ValueError: A group's variables carry different non-spatial
                dimensions, or differ in dtype while `dtype` is None.

        Examples:
            >>> {name: a.shape for name, a in scene.gs.to_numpy().items()}
            {'sentinel-2-l2a': (2, 4, 256, 256), 'dem': (1, 256, 256)}
        """
        return {
            name: raster.gs.to_numpy(dtype=dtype)
            for name, raster in self.datasets.items()
        }

    def to_tensor(
        self, *, dtype: str | torch.dtype | None = None
    ) -> dict[str, torch.Tensor]:
        """Stack each group's variables into one model-input tensor.

        Args:
            dtype: Torch dtype or its string name, applied to every group.
                None keeps each group's own dtype.

        Returns:
            {
                grid node path: tensor shaped `(*axes, band, y, x)`,
            }

        Raises:
            ValueError: A dtype name is unknown, or a group's `to_numpy`
                refuses it.
            TypeError: `dtype` is neither a torch dtype nor a name, or a pixel
                dtype has no tensor equivalent.

        Examples:
            >>> {name: t.shape for name, t in scene.gs.to_tensor().items()}
            {'sentinel-2-l2a': torch.Size([2, 4, 256, 256]), 'dem': torch.Size([1, 256, 256])}
        """
        return {
            name: raster.gs.to_tensor(dtype=dtype)
            for name, raster in self.datasets.items()
        }

    def _read_rasters_by_path(self) -> dict[str, Dataset]:
        """Read every raster node with the attrs its ancestors state for it."""
        rasters = {}
        for path, dataset in self.datasets.items():
            node = self._data if path == "." else self._data[path]
            metadata = dict(dataset.attrs)
            # Collection metadata sits on parent nodes; a node's own attrs win.
            while node is not self._data and node.parent is not None:
                node = node.parent
                metadata = {**node.attrs, **metadata}
            rasters[path] = dataset.assign_attrs(metadata)
        return rasters

    def to_item(
        self,
        path: Location,
        *,
        split_item: Sequence[str] = ("time",),
        split_asset: Sequence[str] = ("variable",),
        id: str | None = None,
        format_id: str | None = None,
        format_paths: str | None = None,
        collection: pystac.Collection | None = None,
        **options: Any,
    ) -> pystac.Item:
        """Save COG assets and return one native PySTAC Item.

        Args:
            path: Output folder or filesystem URL.
            split_item: Observation dimensions; default time, ignored when absent.
            split_asset: Dimensions partitioning assets within an Item;
                "variable" gives each data variable its own asset.
            id: Explicit ID for one observation, overriding format_id.
            format_id: ID template using scalar observation coordinates.
            format_paths: Relative asset path template; defaults to {id}/{asset}.tif.
            collection: Native Collection overriding the stored stac_collection.
            **options: overwrite, storage_options, cog_options, and
                create_stac_item metadata options.

        Returns:
            One Item; more than one observation raises before writing.
        """
        from geosave.geodata.stac.publish import split_items
        from geosave.geodata.stac.publish import publish_items

        self.validate(same_footprint=True, shared_time=True)
        items = split_items(self._read_rasters_by_path(), split_item, split_asset)
        if len(items) != 1:
            raise ValueError(
                "to_item requires one observation; use to_catalog for multiple Items"
            )
        published = publish_items(
            items,
            path,
            id=id,
            format_id=format_id,
            format_paths=format_paths,
            collection=collection,
            split_item=split_item,
            **options,
        )
        return published[0]

    def to_catalog(
        self,
        path: Location,
        *,
        split_item: Sequence[str] = ("time",),
        split_asset: Sequence[str] = ("variable",),
        id: str | None = None,
        format_id: str | None = None,
        format_paths: str | None = None,
        collection: pystac.Collection | None = None,
        **options: Any,
    ) -> GeoDataFrame:
        """Save COG assets and return a native Item GeoDataFrame.

        Args:
            path: Output folder or filesystem URL.
            split_item: Observation dimensions; default time, ignored when absent.
            split_asset: Dimensions partitioning assets within an Item;
                "variable" gives each data variable its own asset.
            id: Explicit ID for one observation, overriding format_id.
            format_id: ID template using scalar observation coordinates.
            format_paths: Relative asset path template; defaults to {id}/{asset}.tif.
            collection: Native Collection overriding the stored stac_collection.
            **options: overwrite, storage_options, cog_options, and
                create_stac_item metadata options.

        Returns:
            GeoDataFrame encoded through stac-geoparquet and Arrow.
        """
        from geosave.geodata.stac.publish import split_items
        from geosave.geodata.stac.publish import publish_items
        from geosave.geodata.stac.catalog import create_catalog

        self.validate(same_footprint=True, shared_time=True)
        items = split_items(self._read_rasters_by_path(), split_item, split_asset)
        published = publish_items(
            items,
            path,
            id=id,
            format_id=format_id,
            format_paths=format_paths,
            collection=collection,
            split_item=split_item,
            **options,
        )
        return create_catalog(published)

    def to_cog(
        self,
        path: Location,
        *,
        split_item: Sequence[str] = ("time",),
        split_asset: Sequence[str] = ("variable",),
        format_paths: str | None = None,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[UPath] | Delayed:
        """Write explicitly partitioned rasters; templates only name files.

        Args:
            path: Output folder or filesystem URL.
            split_item: Dimensions partitioning observations; default time.
            split_asset: Dimensions partitioning assets within an Item;
                "variable" gives each data variable its own asset.
            format_paths: Relative filename template using scalar coordinates,
                asset, var, and path. None uses time folders when dated.
            overwrite: Replace existing files.
            compute: False returns one delayed batch without reading pixels.
            storage_options: Filesystem options.
            **options: Raster creation options.

        Returns:
            Written paths, or a native delayed batch.
        """
        from geosave.geodata.stac.publish import split_items
        from geosave.geodata.io.formats import geotiff

        self.validate(shared_time=True)
        items = split_items(self._read_rasters_by_path(), split_item, split_asset)
        return geotiff.write_items(
            items,
            path,
            format_paths=format_paths,
            writer=geotiff.write_cog,
            compute=compute,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )

    def to_zarr(
        self,
        path: Location,
        *,
        compute: bool = True,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> UPath | Delayed:
        """Write the tree as one Zarr store with a group per collection and grid node.

        Args:
            path: The store, ending in .zarr.
            compute: False returns one native `dask.Delayed`; computing it
                writes the store and returns its path.
            overwrite: Replace an existing store.
            storage_options: Filesystem options merged into the destination UPath.
            **write_options: Supported xarray Zarr write options; `encoding` is keyed by variable name and reaches every node holding it.

        Returns:
            Path of the store written, or one delayed result.

        Raises:
            ValueError: The destination does not end in .zarr.
            KeyError: An encoding name is no variable of the tree.
            FileExistsError: The store exists and overwrite is false.

        Examples:
            >>> open_raster(tree.gs.to_zarr("scene.zarr")).gs.groups
        """
        from geosave.geodata.io.formats import zarr

        return zarr.write_zarr(
            self._data,
            path,
            compute=compute,
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )

    def to_netcdf(
        self,
        path: Location,
        *,
        compute: bool = True,
        engine: NetCDFEngine = "netcdf4",
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> UPath | Delayed:
        """Write the tree as one netCDF file with a group per collection and grid node.

        Args:
            path: The file, ending in .nc, .nc4 or .cdf.
            compute: False returns one native `dask.Delayed`; computing it
                writes the file and returns its path.
            engine: Concrete xarray netCDF backend.
            overwrite: Replace an existing file.
            storage_options: Filesystem options merged into the destination UPath.
            **write_options: Supported xarray netCDF write options; `encoding` is keyed by variable name and reaches every node holding it.

        Returns:
            Path of the file written, or one delayed result.

        Raises:
            ValueError: The destination does not end in a netCDF suffix.
            KeyError: An encoding name is no variable of the tree.
            FileExistsError: The file exists and overwrite is false.

        Examples:
            >>> open_raster(tree.gs.to_netcdf("scene.nc")).gs.groups
        """
        from geosave.geodata.io.formats import netcdf

        return netcdf.write_netcdf(
            self._data,
            path,
            compute=compute,
            engine=engine,
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )
