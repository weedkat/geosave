"""Read raster trees through the `gs` DataTree accessor.

A tree holds collections, and each collection its grid nodes: one Dataset
per grid, named after ESA's EOPF products (`r10m`, `r20m`). Data that cannot
align, such as one collection's 10 m and 20 m bands, sits in sibling nodes.
What the root carries says whether every grid node is co-registered, and
`GeoDataTree.align` is what puts them there.

Examples:
    Co-registered, so every grid node shares one grid and `gs.geobox` reads it::

        >>> scene = xr.merge([optical.gs.to_datatree("sentinel-2-l2a"), dem.gs.to_datatree("dem")])
        /
        ├── /sentinel-2-l2a
        │   └── /r10m                red, nir            (time, y, x)
        └── /dem
            └── /r10m                elevation           (y, x)
        >>> scene.gs.geobox == optical.gs.geobox
        True

    Native resolutions, so each grid node keeps its own grid and `gs.geobox` is
    None::

        >>> sentinel = xr.merge([bands_10m.gs.to_datatree("s2"), bands_20m.gs.to_datatree("s2")])
        >>> sentinel.gs.geobox is None
        True
        >>> sentinel["s2"].gs.to_dataset().gs.resolution.x
        10.0
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Unpack, cast

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from upath import UPath

from geosave.geodata.conventions import ITEM_ID_FORMAT
from geosave.geodata.transform import warp

from .base import GeoRaster

if TYPE_CHECKING:
    import holoviews as hv
    import numpy as np
    import torch
    from numpy.typing import DTypeLike
    from odc.geo import SomeResolution

    from geosave.geodata import Dataset, DataTree, GeoDataFrame
    from geosave.geodata.stac.item import ItemTime
    from geosave.geodata.core.anchor import GeoAnchor
    from dask.delayed import Delayed
    from geosave.geodata.io.formats.geotiff import COGWriteOptions
    from geosave.geodata.io.formats.netcdf import NetCDFEngine
    from geosave.geodata.io.formats.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.formats.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from geosave.geodata.transform.warp import Resampling
    from geosave.geodata.utils.datetime import DateRange


class GeoDataTree(GeoRaster["DataTree"]):
    """Read and persist one raster-tree DataTree.

    The root carries the tree's own attrs, a collection node its STAC
    metadata, and each grid node's raster its own, read through
    `rasters["<collection>/<res>"].gs`.

    Args:
        data: DataTree of collections over grid nodes, `/{collection}/{res}`,
            or a scene tree whose children are grid nodes.

    Examples:
        >>> scene.gs.groups
        ('sentinel-2-l2a', 'dem')
        >>> scene.gs.geobox.shape
        (512, 512)
    """

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
            for group, raster in self.rasters.items()
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
            >>> scene.gs.rasters["dem"].gs.timespan is None
            True
            >>> scene.gs.timespan == scene.gs.rasters["sentinel-2-l2a"].gs.timespan
            True
        """
        spans = [
            span
            for span in (raster.gs.timespan for raster in self.rasters.values())
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
            for times in (raster.gs.times for raster in self.rasters.values())
            if times is not None
        ]
        if not dated:
            return None
        return pd.DatetimeIndex(dated[0].append(dated[1:]).unique().sort_values())

    @property
    def anchor(self) -> GeoAnchor:
        """Return the union of leaf footprints and their temporal envelope.

        Different grids need no alignment. This inspects metadata without
        reading pixels; missing geographic coverage raises ValueError.
        """
        from geosave.geodata.core.anchor import GeoAnchor

        if not self._data.children:
            return self._data.to_dataset(inherit="all_coords").gs.anchor
        geometries = []
        for raster in self.rasters.values():
            geometries.append(raster.gs.anchor.geometry)
        if not geometries:
            raise ValueError("an empty tree has no geographic coverage")
        geometry = geometries[0]
        for member in geometries[1:]:
            geometry = geometry.union(member.to_crs(geometry.crs))
        return GeoAnchor(geometry, timespan=self.timespan)

    @property
    def rasters(self) -> dict[str, Dataset]:
        """Read every grid node as a raster Dataset, keyed by its path.

        Returns:
            {
                grid node path, as `"s2/r10m"`: that node's raster, carrying
                    the grid and time coordinates it inherits from above,
            }

        Examples:
            Rebuild a tree to add a collection:

            >>> list(scene.gs.rasters)
            ['sentinel-2-l2a/r10m', 'dem/r10m']
            >>> xr.merge([scene, ndvi.gs.to_datatree("ndvi")]).gs.groups
            ('sentinel-2-l2a', 'dem', 'ndvi')
        """
        # Grid nodes are the leaves: a collection node holds none of their pixels.
        rasters = {}
        for node in self._data.leaves:
            if node is self._data:
                continue
            path = node.relative_to(self._data)
            # .dataset would return a DatasetView whose attrs still write through here.
            rasters[path] = cast("Dataset", node.to_dataset(inherit="all_coords"))
        return rasters

    def _read_collection_metadata(self, name: str) -> Dataset:
        """Return a pixel-free Dataset carrying a collection's attrs over its first grid node's."""
        node = self._data[name]
        first = node if not node.children else next(iter(node.leaves))
        return cast("Dataset", xr.Dataset(attrs={**first.attrs, **node.attrs}))

    def to_dataset(
        self,
        *,
        res: str | None = None,
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
    ) -> Dataset:
        """Convert one collection to one Dataset on one of its grids.

        A collection's grid nodes are one grid at several zoom levels: one CRS,
        origin and extent, pixel sizes in whole-number ratios. Every node is
        warped onto the chosen one, then their variables join. Up to a finer
        grid `"nearest"` repeats each pixel; down to a coarser one `"average"`
        takes each block's mean and `"mode"` its most common value.

        Args:
            res: The grid node every node is brought onto, as `"r20m"`. None
                takes the finest, the first on a tie, which keeps every value.
            resampling: One GDAL kernel, or one per variable with `"*"` for
                the rest, as `warp.reproject` takes.

        Returns:
            Dataset of every grid node's variables on the chosen grid, with
            the attrs every node shares after taking those of its collection.

        Raises:
            ValueError: The tree holds several collections, or a node does not
                nest in the chosen grid, which `align` reprojects instead.
            KeyError: `res` names no grid node.

        Examples:
            >>> tree["s2"].gs.to_dataset().gs.resolution.x
            10.0
            >>> tree["s2"].gs.to_dataset(res="r20m", resampling="average").gs.resolution.x
            20.0
        """
        from geosave.geodata.transform.combine import merge

        # A grid node is a Dataset already.
        if not self._data.children:
            return cast("Dataset", self._data.to_dataset(inherit="all_coords"))

        # 1. One collection: every child is a grid node
        collections = []
        for name, child in self._data.children.items():
            if child.children:
                collections.append(name)
        if collections:
            raise ValueError(
                f"the tree holds collections {collections}; convert one, as "
                f"tree[{collections[0]!r}].gs.to_dataset()"
            )

        # 2. The grid every node is brought onto
        rasters = self.rasters
        if res is None:
            res = min(
                rasters, key=lambda name: abs(rasters[name].odc.geobox.resolution.x)
            )
        elif res not in rasters:
            raise KeyError(f"{res!r} is no grid node; the nodes are {list(rasters)}")
        target = rasters[res].odc.geobox

        # 3. Each node onto it, carrying the attrs of this node above it
        nodes = []
        for name, raster in rasters.items():
            grid = raster.odc.geobox
            if not _nests(grid, target):
                raise ValueError(
                    f"{name} does not nest in {res}: no whole-number zoom of one "
                    f"grid gives the other; reproject it with "
                    f"gs.align(target=...) before converting"
                )
            if grid != target:
                raster = warp.reproject(raster, target, resampling=resampling)
            nodes.append(raster.assign_attrs({**self._data.attrs, **raster.attrs}))
        return merge(nodes)

    def align(
        self,
        *,
        target: warp.Target,
        extent: warp.Extent = "match",
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
        resolution: SomeResolution | warp.Native | None = None,
    ) -> DataTree:
        """Warp every grid node onto one grid.

        Grid nodes on their own grids share no axes, so nothing indexes them
        together. Once aligned they share one grid, which `gs.geobox` reads,
        and one `isel` on `y` and `x` cuts every node at once.

        Args:
            target: Grid to align onto, a raster already on one, or a CRS.
                Required. Select a grid node when a reference tree has
                different grids.
            extent: Ground the aligned nodes cover. `"union"` reaches every
                node, holding fill where one does not; `"intersection"` drops
                every edge outside the ground they all cover; `"match"` adopts
                the reference's own.
            resampling: One GDAL kernel for every variable, or a mapping naming
                each variable's own as `"<collection>/<variable>"`, which `"*"`
                answers the rest of. Pixels move only where a node does not
                already sit on the grid it aligns onto.
            resolution: Pixel size every node lands at. None takes the
                reference's own. `"native"` keeps each node's instead, so the
                nodes cover one extent on nesting grids.

        Returns:
            New DataTree whose grid nodes share one exact GeoBox, or cover one
            extent at their own pixel sizes under `"native"`.

        Raises:
            KeyError: A `resampling` mapping names neither a variable nor `"*"`.
            ValueError: A node sits on no locatable grid, `target` names no
                grid or CRS, `extent` names no ground, `"intersection"` leaves
                none in common, or `resampling` would blend class codes.

        Examples:
            A grid node names its grid by the raster it is, since a bare
            string is read as a CRS:

            >>> onto = sentinel.gs.rasters["s2/r10m"]
            >>> aligned = sentinel.gs.align(target=onto, extent="intersection")
            >>> aligned.gs.resolution.x
            10.0
            >>> aligned.isel(y=slice(0, 256), x=slice(0, 256)).gs.geobox.shape
            Shape2d(x=256, y=256)

            Imagery and labels in one tree warp with different kernels:

            >>> scene.gs.align(target=imagery, resampling={"*": "bilinear", "labels/mask": "mode"})
        """
        (aligned,) = warp.align(
            [self._data],
            target=target,
            extent=extent,
            resampling=resampling,
            resolution=resolution,
        )
        return aligned

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
        for name, raster in self.rasters.items():
            place = raster.gs.anchor.locate() if xlabel is None else None
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
            for name, raster in self.rasters.items()
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
            for name, raster in self.rasters.items()
        }

    def to_cog(
        self,
        path: Location,
        *,
        path_format: str | Mapping[str, str] | None = None,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[UPath] | Delayed:
        """Write the tree as Cloud Optimized GeoTIFFs under the folder `path`.

        A GeoTIFF holds one grid at one date, so `path_format` lays the tree
        out: `{collection}` splits the collections, `{res}` a collection's grid
        nodes, `{time}` dates and `{band}` variables. A field left out keeps its
        axis in one file, which several collections, grids or dates cannot
        share. Without `path_format` every band of every date is its own file,
        `{collection}/{time:%Y%m%dT%H%M%S}/{band}.tif`, which needs no `{res}`
        since band names are unique within a collection.

        Args:
            path: Folder or URL prefix the files go under.
            path_format: Relative path filled with `str.format`, ending in
                .tif, or one per collection. `{time}` takes strftime directives.
            compute: False returns one native `dask.Delayed`; computing it
                writes all outputs and returns their paths.
            overwrite: Replace existing destinations.
            storage_options: Filesystem options merged into the destination UPath.
            **options: GeoTIFF creation options.

        Returns:
            Written paths in collection, grid, time then variable order, or one
            delayed result.

        Raises:
            ValueError: `path` names a file rather than a folder, a path format
                misses a field the tree needs, as `{res}` for two grids sharing
                a band name, or a raster cannot be one GeoTIFF.
            KeyError: A path format field has no value.
            FileExistsError: A destination exists and overwrite is false.

        Examples:
            >>> paths = tree.gs.to_cog("out", path_format="{collection}/{res}/{time:%Y%m%d}_{band}.tif")
        """
        from geosave.geodata.io.formats import geotiff
        from geosave.geodata.io.paths import expand_paths
        from geosave.geodata.io.storage import location

        files = expand_paths(self._data, location(path, storage_options), path_format)
        return geotiff.write_files(
            files, geotiff.write_cog, compute=compute, overwrite=overwrite, **options
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

    def to_catalog(
        self,
        path: Location,
        *,
        path_format: str | Mapping[str, str] | None = None,
        id_format: str = ITEM_ID_FORMAT,
        datetime: ItemTime | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> GeoDataFrame:
        """Write the tree as COGs and return their STAC Items as one item table.

        Each collection is one STAC Collection named after it, and each of its
        dates one Item, whose assets are that date's files across every grid
        node, each stating its own grid in `proj:*`. A collection carrying a
        `stac_collection` gives the table that Collection's metadata under the
        collection's name.

        Args:
            path: Folder the files go under.
            path_format: Relative path ending in .tif, as `to_cog` lays it out,
                or one per collection. None writes one file per band and date.
            id_format: Item id template, as `Dataset.gs.to_catalog` takes it.
            datetime: Date or interval for collections stating no time.
            overwrite: Replace files that exist.
            storage_options: Credentials and connection options for the files.
            **options: GeoTIFF creation options.

        Returns:
            Item table, one row per collection and date.

        Raises:
            ValueError: A file does not end in .tif, a mapping misses a
                collection, destinations collide, or a collection cannot be
                saved as `Dataset.gs.to_catalog` explains.
            FileExistsError: A file exists and `overwrite` is false.

        Examples:
            >>> catalog = tree.gs.to_catalog("out", path_format="{collection}/{time:%Y%m%d}/{band}.tif")
            >>> sorted(catalog["collection"].unique())
            ['dem', 's2']
        """
        from geosave.geodata.io.stac import write_assets

        collections = []
        for name in self.groups:
            carried = self._read_collection_metadata(name).gs.stac_collection
            if carried is not None:
                # In a tree's catalog the collection's name is its id.
                renamed = carried.clone()
                renamed.id = name
                collections.append(renamed)
        return write_assets(
            self._data,
            path,
            path_format=path_format,
            collections=collections,
            id_format=id_format,
            datetime=datetime,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )


def _nests(grid: GeoBox, target: GeoBox) -> bool:
    """Say whether one grid is the other zoomed out by a whole number."""
    if grid.crs != target.crs:
        return False
    if grid == target:
        return True
    fine, coarse = (
        (grid, target)
        if abs(grid.resolution.x) < abs(target.resolution.x)
        else (target, grid)
    )
    factor = abs(coarse.resolution.x) / abs(fine.resolution.x)
    return factor == round(factor) and fine.zoom_out(round(factor)) == coarse
