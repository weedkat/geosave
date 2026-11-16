"""General geospatial helpers for native xarray DataTrees."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack, cast

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox


from .base import GeoRaster, Rule
from geosave.geodata.core.conventions import TIME_COORDINATE
from geosave.geodata.errors import RasterValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import geopandas as gpd
    import torch
    from odc.geo import SomeCRS, SomeResolution
    from odc.geo.geom import Geometry

    from geosave.geodata import DataTree
    from geosave.geodata.core.region import GeoRegion
    from dask.delayed import Delayed
    from geosave.geodata.io.raster.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.raster.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from upath import UPath
    from .transform.extent import Padding
    from .transform.warp import Resampling
    from .utils.windows import Keep
    from geosave.geodata.utils.datetime import DateRange, Freq


# Rules a tree can be asked to meet; `tree.gs.validate(same_geobox)` runs one.


def same_grid(tree: xr.DataTree) -> None:
    """Require one grid across the data nodes: CRS, resolution and pixel alignment.

    Extents may differ: nodes on one grid stack by their pixel labels, the
    union of their extents, when datasets are assembled.

        a: x 500000–500040    b: x 500020–500080    10 m, alignment (0, 0)   → one grid
        c: x 500005–500045                          10 m, alignment (5, 0)   → another

    Raises:
        RasterValidationError: Nodes sit on several grids; the error names each.
    """
    nodes_by_grid: dict[tuple | None, list[str]] = {}
    for node in tree.subtree:
        if not node.data_vars:
            continue
        path = node.relative_to(tree)
        geobox = node.to_dataset(inherit="all_coords").gs.geobox
        grid = None
        if geobox is not None:
            grid = (geobox.crs, geobox.resolution, geobox.alignment)
        nodes_by_grid.setdefault(grid, []).append(path)
    if len(nodes_by_grid) <= 1:
        return

    lines = []
    for paths in nodes_by_grid.values():
        lines.append(", ".join(paths))
    raise RasterValidationError(
        f"Nodes are on {len(nodes_by_grid)} grids ({'; '.join(lines)}). Resample "
        "them onto one with tree.gs.reproject(geobox), or select one grid's nodes.",
        path=list(nodes_by_grid.values())[1][0],
    )


def same_geobox(tree: xr.DataTree) -> None:
    """Require one GeoBox across the data nodes: the same grid and the same extent.

    Raises:
        RasterValidationError: Nodes sit on several GeoBoxes; the error names each.
    """
    nodes_by_geobox: dict[GeoBox | None, list[str]] = {}
    for node in tree.subtree:
        if not node.data_vars:
            continue
        geobox = node.to_dataset(inherit="all_coords").gs.geobox
        nodes_by_geobox.setdefault(geobox, []).append(node.relative_to(tree))
    if len(nodes_by_geobox) <= 1:
        return

    lines = []
    for paths in nodes_by_geobox.values():
        lines.append(", ".join(paths))
    raise RasterValidationError(
        f"Nodes are on {len(nodes_by_geobox)} GeoBoxes ({'; '.join(lines)}). Align "
        "them with tree.gs.reproject(geobox).",
        path=list(nodes_by_geobox.values())[1][0],
    )


class GeoDataTree(GeoRaster["DataTree"]):
    """Operate on raster-bearing nodes without prescribing a hierarchy."""

    @property
    def spatial_dims(self) -> tuple[str, str] | None:
        """Return the spatial axes of the tree's one GeoBox, or None without one."""
        if self.geobox is None:
            return None

        first = next(node for node in self._data.subtree if node.data_vars)
        return first.to_dataset(inherit="all_coords").gs.spatial_dims

    @property
    def geobox(self) -> GeoBox | None:
        """Return the GeoBox every raster shares, or None for none or several."""
        try:
            self.validate(same_geobox)
        except RasterValidationError:
            return None

        first = next(node for node in self._data.subtree if node.data_vars)
        return first.to_dataset(inherit="all_coords").gs.geobox

    def validate(self, *rules: Rule[DataTree]) -> None:
        """Require every raster georeferenced, then every rule, without reading pixels.

        Rasters may sit on different grids, so each needs its own grid ODC
        recognizes, with a CRS. A tree without rasters fails.

        Args:
            *rules: Tree rules the operation needs, such as `same_grid` or
                `same_geobox`.

        Raises:
            RasterValidationError: A raster or a rule is broken. The error
                names the node and how to fix it.

        Examples:
            >>> tree.gs.validate(same_geobox)
        """
        rasters = [node for node in self._data.subtree if node.data_vars]
        if not rasters:
            raise RasterValidationError(
                "There are no raster variables to validate. Select or load raster data first."
            )

        for node in rasters:
            try:
                node.to_dataset(inherit="all_coords").gs.validate()
            except RasterValidationError as error:
                error.path = node.relative_to(self._data)
                raise

        for rule in rules:
            rule(self._data)

    @property
    def timespan(self) -> DateRange | None:
        """Read the coverage of every dataset, from the earliest start to the latest end.

        Each dataset states its time on its own node; datasets on their own
        timelines widen the span together.

        Returns:
            First and last covered instant, or None where no dataset is dated.

        Examples:
            >>> tree.gs.timespan == tree["s2"].to_dataset().gs.timespan
            True
        """
        starts = []
        ends = []
        for node in self._data.subtree:
            span = node.to_dataset(inherit=False).gs.timespan
            if span is not None:
                starts.append(span[0])
                ends.append(span[1])
        if not starts:
            return None
        return min(starts), max(ends)

    @property
    def times(self) -> pd.DatetimeIndex | None:
        """Read every dataset's time label, each instant once, in order.

        Returns:
            Sorted labels across datasets, or None where no dataset is dated.
        """
        labels = []
        for node in self._data.subtree:
            times = node.to_dataset(inherit=False).gs.times
            if times is not None:
                labels.append(times)
        if not labels:
            return None
        return labels[0].append(labels[1:]).unique().sort_values()

    @property
    def region(self) -> GeoRegion:
        """Return the union of the rasters' extents, with the datasets' time span.

        Rasters on different grids need no alignment. This inspects metadata
        without reading pixels.

        Raises:
            RasterValidationError: The tree holds no raster, or a raster has no
                grid with a CRS.
        """
        from odc.geo import CRS

        from geosave.geodata.core.region import GeoRegion

        self.validate()

        # Many datasets share a few GeoBoxes; each extent is added once.
        geoboxes: dict[GeoBox, None] = {}
        for node in self._data.subtree:
            if node.data_vars:
                geobox = cast("GeoBox", node.to_dataset(inherit="all_coords").gs.geobox)
                geoboxes[geobox] = None

        first, *others = geoboxes
        extent = first.extent
        crs = cast("CRS", first.crs)
        for geobox in others:
            extent = extent.union(geobox.extent.to_crs(crs))
        return GeoRegion(extent, timespan=self.timespan)

    def crop(
        self, region: Geometry | gpd.GeoSeries | gpd.GeoDataFrame, *, mask: bool = True
    ) -> DataTree:
        """Cut every raster node down to the extent of a region.

        Args:
            region: Geometry, or geometries, to cut against, reprojected onto
                each node's CRS where they sit in another.
            mask: Also make nodata the pixels outside the geometries.

        Returns:
            New tree; empty nodes, which hold no pixels, are unchanged.

        Examples:
            >>> tree.gs.crop(field_boundaries)
        """
        tree = self._data.map_over_datasets(
            lambda raster: (
                raster.gs.crop(region, mask=mask) if raster.gs.variables else raster
            )
        )
        return cast("DataTree", tree)

    def crop_to_grid(
        self,
        geobox: GeoBox,
        *,
        mode: Padding = "constant",
        constant_value: int | float = 0,
    ) -> DataTree:
        """Take the pixels of an aligned grid from every raster node, padding past the edges.

        Args:
            geobox: Grid sharing the rasters' CRS and pixel size, offset by
                whole pixels. It may extend past the raster edges.
            mode: How pixels outside a raster are filled, as in `numpy.pad`.
            constant_value: Fill for `"constant"`, keeping each dtype.

        Returns:
            New lazy tree with every raster node on exactly `geobox`; empty
            nodes are unchanged.

        Raises:
            ValueError: A raster has no regular grid, or `geobox` is not
                pixel-aligned with it or does not overlap it.

        Examples:
            >>> chip = tree.gs.crop_to_grid(tree.gs.geobox[0:256, 0:256])
        """

        def cut(raster: xr.Dataset) -> xr.Dataset:
            if not raster.gs.variables:
                return raster
            return raster.gs.crop_to_grid(
                geobox, mode=mode, constant_value=constant_value
            )

        return cast("DataTree", self._data.map_over_datasets(cut))

    def sample(
        self,
        freq: Freq,
        *,
        size: int = 1,
        stride: int = 1,
        keep: Keep = "last",
    ) -> list[DataTree]:
        """Sample the shared time axis into windows of consecutive slots, lazily.

        Each window is this tree at `size` slots of length `freq` in a row,
        as `Dataset.gs.sample` cuts one raster. A node without a time
        axis, such as a static label, is in every window whole.

        Args:
            freq: Length of one slot, as a pandas offset alias such as "MS".
            size: Slots per window, the length of each window's time axis.
            stride: Slots between the starts of consecutive windows.
            keep: The observation a slot keeps when several fall in it.

        Returns:
            One tree per window, in time order.

        Raises:
            ValueError: No node has a datetime `time` dimension, the dated
                nodes do not share one time axis, `size` or `stride` is below
                1, or `keep` is unknown.

        Examples:
            >>> windows = tree.gs.sample("MS", size=2)
            >>> chipper = RasterChipper(windows, size=256)
        """
        from .utils.windows import window_times

        times = self.times
        if times is None:
            raise ValueError("sample needs a node with a datetime `time` dimension")
        for node in self._data.subtree:
            node_times = node.to_dataset(inherit=False).gs.times
            if node_times is not None and not node_times.equals(times):
                raise ValueError(
                    f"node {node.path!r} has its own time axis; windows need "
                    "every dated node on the same time labels"
                )

        windows = []
        for positions in window_times(times, freq, size=size, stride=stride, keep=keep):
            window = self._data.isel({TIME_COORDINATE: positions}, missing_dims="ignore")
            windows.append(cast("DataTree", window))
        return windows

    def to_tensor(
        self, *, grid: bool = False, context: Sequence[str] = ()
    ) -> dict[str, torch.Tensor]:
        """Convert every raster node to a tensor named by its path, with the grid and context.

        Each node becomes one tensor laid out as `Dataset.gs.to_numpy`,
        keeping its own dtype. The default DataLoader collation batches every
        entry, and a model chain takes each by name.

        Args:
            grid: Add `transform` and `crs`, which place the pixels, so a
                prediction can be placed without the tree and a scene can be
                chipped by `TensorChipper`.
            context: Names in `TENSOR_CONTEXT` whose readers add one entry each,
                such as `"centroid"` or `"day_of_year"`.

        Returns:
            {
                "<node path>": tensor shaped `(*axes, band, y, x)`,
                "transform": float64 tensor of the six affine coefficients,
                "crs": int64 EPSG code,
                "<context name>": the reader's tensor,
            }

        Raises:
            RasterValidationError: The tree holds no raster, or its rasters
                sit on several GeoBoxes.
            KeyError: A context name is not in `TENSOR_CONTEXT`.
            ValueError: A node takes the name of an entry, a node's variables
                differ in dtype or dimensions, the CRS has no EPSG code, or a
                reader refuses the tree.

        Examples:
            >>> sample = tree.gs.to_tensor(grid=True, context=["centroid"])
            >>> sample["image"].shape, sample["crs"], sample["centroid"]
            (torch.Size([2, 3, 256, 256]), tensor(32633), tensor([51.0000, 11.0000]))
            >>> GeoBox(sample["image"].shape[-2:], Affine(*sample["transform"]), int(sample["crs"]))
        """
        from .utils.context import build_sample
        from .utils.tensor import as_tensor

        self.validate(same_geobox)
        if self.geobox is None:
            raise RasterValidationError("The tree holds no georeferenced raster.")

        layers = {}
        for node in self._data.subtree:
            if not node.data_vars:
                continue
            node_path = node.relative_to(self._data)
            raster = node.to_dataset(inherit="all_coords")
            layers[node_path] = as_tensor(raster.gs.to_numpy, None)
        return build_sample(layers, self._data, grid=grid, context=context)

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

    def to_zarr(
        self,
        path: Location,
        *,
        compute: bool = True,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[ZarrWriteOptions],
    ) -> UPath | Delayed:
        """Write the tree as one Zarr store with a group per node.

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
            >>> open_raster_tree(tree.gs.to_zarr("scene.zarr")).children.keys()
        """
        from geosave.geodata.io.raster import zarr

        return zarr.write_zarr_tree(
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
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **write_options: Unpack[NetCDFWriteOptions],
    ) -> UPath | Delayed:
        """Write the tree as one netCDF file with a group per node.

        Args:
            path: The file, ending in .nc, .nc4 or .cdf.
            compute: False returns one native `dask.Delayed`; computing it
                writes the file and returns its path.
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
            >>> open_raster_tree(tree.gs.to_netcdf("scene.nc")).children.keys()
        """
        from geosave.geodata.io.raster import netcdf

        return netcdf.write_netcdf_tree(
            self._data,
            path,
            compute=compute,
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )
