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
        >>> sentinel.gs.reproject(bands_10m.gs.geobox).gs.geobox.resolution.x
        10.0
"""

from __future__ import annotations

from collections.abc import Mapping
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
    from odc.geo import SomeCRS, SomeResolution
    from odc.geo.geom import Geometry
    import holoviews as hv
    import torch
    from numpy.typing import DTypeLike

    from geosave.geodata import Dataset, DataTree, GeoDataFrame
    from geosave.geodata.core.region import GeoRegion
    from dask.delayed import Delayed
    from geosave.geodata.io.formats.geotiff import COGWriteOptions
    from geosave.geodata.io.formats.netcdf import NetCDFEngine
    from geosave.geodata.io.formats.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.formats.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from .transform.warp import Resampling
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
    def geobox(self) -> GeoBox | None:
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
        same_grid: bool = False,
        stac_pattern: bool = False,
    ) -> None:
        """Check every raster node, and the STAC layout when an operation needs it.

        Each data node needs a grid ODC recognizes, with a CRS. Empty inputs fail.

        Args:
            same_grid: Require every data node on one grid, so they merge
                into one Dataset.
            stac_pattern: Require the STAC roles counted from the leaves:
                every node holding data is a leaf (an asset), every leaf sits
                at one depth below this node, and that depth is at most 3
                (asset ← Item ← collection), so a whole tree, one collection
                or one Item validates, also after an operation re-roots it.

        Raises:
            RasterValidationError: A raster is invalid or the layout differs.
                The error names the node and how to fix it.

        Examples:
            >>> tree.gs.validate(stac_pattern=True)
        """
        datasets = self.datasets
        self._validate_nonempty(datasets)
        for path, dataset in datasets.items():
            try:
                dataset.gs.validate()
            except RasterValidationError as error:
                error.path = path
                raise
        if same_grid:
            self._validate_same_grid(datasets)
        if stac_pattern:
            self._validate_stac_pattern()

    def _validate_stac_pattern(self) -> None:
        """Require STAC roles counted from the leaves: asset ← Item ← collection."""
        hint = (
            "A STAC tree holds data only on its leaves, one asset each, at most "
            "collection/item/asset deep; load one with stac.load_datatree(items) "
            "or catalog.gs.to_datatree()."
        )
        first_by_depth: dict[int, str] = {}
        for path, node in self._data.subtree_with_keys:
            if node.children:
                if node.data_vars:
                    raise RasterValidationError(
                        f"This node holds data but is not a leaf. {hint}", path=path
                    )
                continue
            if path == ".":
                continue
            depth = path.count("/") + 1
            if depth > 3:
                raise RasterValidationError(
                    f"This leaf sits deeper than collection/item/asset. {hint}",
                    path=path,
                )
            first_by_depth.setdefault(depth, path)
        if len(first_by_depth) > 1:
            depths = " and ".join(str(depth) for depth in sorted(first_by_depth))
            raise RasterValidationError(
                f"Leaves sit at depths {depths}, so their roles differ. {hint}",
                path=first_by_depth[max(first_by_depth)],
            )

    def _validate_same_grid(self, datasets: Mapping[str, Dataset]) -> None:
        """Require one grid across the data nodes, naming every grid when they differ."""
        nodes_by_grid: dict[GeoBox | None, list[str]] = {}
        for path, dataset in datasets.items():
            nodes_by_grid.setdefault(dataset.gs.geobox, []).append(path)
        if len(nodes_by_grid) == 1:
            return
        lines = []
        for paths in nodes_by_grid.values():
            lines.append(", ".join(paths))
        first_other = list(nodes_by_grid.values())[1][0]
        raise RasterValidationError(
            f"Nodes are on {len(nodes_by_grid)} grids ({'; '.join(lines)}). Align them "
            "with tree.gs.reproject(geobox).",
            path=first_other,
        )

    def _validate_nonempty(self, datasets: Mapping[str, Dataset]) -> None:
        """Require at least one data-bearing node."""
        if not datasets:
            raise RasterValidationError(
                "There are no raster variables to validate. Select or load raster data first."
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
        from geosave.geodata.core.region import GeoRegion

        extent = attrs.ACDDExtent.from_attrs(self._data.attrs)
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
            if dataset.gs.variables:
                rasters[path] = cast("Dataset", dataset)
        return rasters

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
                raster.gs.crop(region, mask=mask) if raster.gs.variables else raster
            )
        )
        return cast("DataTree", tree)

    def reproject(
        self,
        how: GeoBox | SomeCRS,
        *,
        resolution: SomeResolution | None = None,
        resampling: Resampling = "nearest",
    ) -> DataTree:
        """Reproject every raster node with `Dataset.gs.reproject`.

        A GeoBox puts every node on that one grid; a CRS sizes each node's own
        grid, as `Dataset.gs.reproject` does.

        Args:
            how: GeoBox or CRS, as for `Dataset.gs.reproject`.
            resolution: Output pixel size for a CRS, as for `Dataset.gs.reproject`.
            resampling: One ODC resampling method for every raster node.

        Returns:
            Tree with unchanged hierarchy and reprojected raster nodes.

        Examples:
            >>> aligned = tree.gs.reproject(labels.gs.geobox)
        """

        def move(raster: xr.Dataset) -> xr.Dataset:
            if not raster.gs.variables:
                return raster
            return raster.gs.reproject(
                how, resolution=resolution, resampling=resampling
            )

        return cast("DataTree", self._data.map_over_datasets(move))

    def combine_stac(self) -> Dataset:
        """Combine a STAC Item, or a collection on one grid, into one Dataset.

        An Item merges its asset leaves into one observation. A collection
        merges each Item that way, then concatenates the Items along `time`
        in tree order. The selected node's attrs, such as its
        Collection, sit beneath the merged ones. Several collections keep
        independent timelines, so the tree holds them.

        Returns:
            Lazy Dataset of the Item, or of the collection along `time`.

        Raises:
            RasterValidationError: The tree is not one Item or collection on one
                grid (align with `tree.gs.reproject`), or an Item lacks a
                variable another holds.

        Examples:
            >>> tree["s2"].gs.combine_stac()
            <xarray.Dataset> Dimensions: (time: 2, y: 64, x: 64) ...
            >>> tree["s2/S2B_33UUP_20250115"].gs.combine_stac()
            <xarray.Dataset> Dimensions: (y: 64, x: 64) ...
        """
        self.validate(same_grid=True, stac_pattern=True)
        depth = self._data.leaves[0].level - self._data.level
        if depth == 3:
            raise RasterValidationError(
                "Collections keep independent timelines, so the tree holds them. "
                "Combine one collection, such as tree['s2'].gs.combine_stac()."
            )
        items = [self._data] if depth < 2 else list(self._data.children.values())

        observations = []
        for item in items:
            assets = []
            for leaf in item.leaves:
                assets.append(leaf.to_dataset(inherit="all_coords"))
            observation = xr.merge(
                assets,
                join="exact",
                compat="no_conflicts",
                combine_attrs="drop_conflicts",
            )
            if observations and set(observation.data_vars) != set(
                observations[0].data_vars
            ):
                raise RasterValidationError(
                    "Items of one collection hold the same variables; this Item holds "
                    f"{sorted(observation.data_vars)}.",
                    path=item.name,
                )
            observations.append(observation)

        combined = observations[0]
        if depth == 2:
            # "different" lays each Item's own scalars, as stac_properties, along time.
            combined = xr.concat(
                observations,
                dim=TIME_COORDINATE,
                join="exact",
                coords="different",
                compat="no_conflicts",
                combine_attrs="drop_conflicts",
            )
        return cast(
            "Dataset", combined.assign_attrs({**self._data.attrs, **combined.attrs})
        )

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
                `tree["<collection>"].gs.combine_stac().gs.plot()`.

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

    def to_cog(
        self,
        path: Location,
        *,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[UPath] | Delayed:
        """Write every leaf as one COG at its node path.

        A leaf holds one instant of one grid, as a COG does, and its path names
        the file: ``/s2/20250601T000000/red`` becomes
        ``{path}/s2/20250601T000000/red.tif``. Rename nodes to rename files.

        Args:
            path: Output folder or filesystem URL.
            overwrite: Replace existing files.
            compute: False returns one delayed batch without reading pixels.
            storage_options: Filesystem options.
            **options: COG creation options.

        Returns:
            Written paths in leaf order, or one native delayed batch.

        Raises:
            ValueError: A leaf cannot be one COG; nothing is written.
            FileExistsError: A destination exists and overwrite is false.

        Examples:
            >>> scene.gs.expand_stac().gs.to_cog("out")
            [UPath('/abs/out/20250601T000000/red.tif'), ...]
        """
        from geosave.geodata.io import storage
        from geosave.geodata.io.formats import geotiff

        root = storage.location(path, storage_options)
        files = {}
        for leaf in self._data.leaves:
            name = leaf.relative_to(self._data)
            files[root / f"{name}.tif"] = leaf.to_dataset(inherit="all_coords")
        return geotiff.write_files(
            files, geotiff.write_cog, compute=compute, overwrite=overwrite, **options
        )

    def to_item(self, path: Location, **options: Any) -> pystac.Item:
        """STAC Item publication is being rebuilt on `expand_stac` and `to_cog`.

        Raises:
            NotImplementedError: Always, until publication is rebuilt.
        """
        raise NotImplementedError(
            "STAC publication is being rebuilt on expand_stac and tree.gs.to_cog"
        )

    def to_catalog(self, path: Location, **options: Any) -> GeoDataFrame:
        """STAC catalog publication is being rebuilt on `expand_stac` and `to_cog`.

        Raises:
            NotImplementedError: Always, until publication is rebuilt.
        """
        raise NotImplementedError(
            "STAC publication is being rebuilt on expand_stac and tree.gs.to_cog"
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
