"""Read STAC trees through the `gs` DataTree accessor.

A STAC tree is catalog / collection / Item / asset, as
`geosave.geodata.core.conventions` describes. Only assets, the leaves, hold
data; each carries its own grid. `ds.gs.expand()` builds one, and
`tree["<collection>"].gs.combine()` returns the collection as a Dataset.

Examples:
    >>> tree = scene.gs.expand()
    >>> list(tree.children)
    ['default']
    >>> tree["default"].gs.combine()
    <xarray.Dataset> Dimensions: (time: 2, y: 64, x: 64) ...
    >>> tree.gs.reproject(scene.gs.geobox).gs.geobox == scene.gs.geobox
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Unpack, cast

import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox


from .base import GeoRaster, Rule
from .utils.stack import stack_rasters
from geosave.geodata.core.conventions import TIME_COORDINATE
from geosave.geodata.errors import RasterValidationError

if TYPE_CHECKING:
    import geopandas as gpd
    from odc.geo import SomeCRS, SomeResolution
    from odc.geo.geom import Geometry

    from geosave.geodata import Dataset, DataTree, GeoDataFrame
    from geosave.geodata.core.region import GeoRegion
    from dask.delayed import Delayed
    from geosave.geodata.io.formats.geotiff import COGWriteOptions
    from geosave.geodata.io.formats.netcdf import NetCDFWriteOptions
    from geosave.geodata.io.formats.zarr import ZarrWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from upath import UPath
    from .transform.warp import Resampling
    from geosave.geodata.utils.datetime import DateRange


# Rules a tree can be asked to meet; `tree.gs.validate(stac_catalog)` runs one.


def stac_collection(tree: xr.DataTree) -> None:
    """Require a STAC collection: Items below it, assets below each Item.

    Only assets, the leaves, hold data, and every leaf sits exactly at
    Item/asset below this node.

        /                  collection   ← this node
        ├── /2025-06-01    Item
        │   ├── /red       asset
        │   └── /nir       asset
        └── /2025-06-06    Item

    Raises:
        RasterValidationError: The layout differs; the error names the node.
    """
    hint = (
        "A STAC collection holds Items, each holding one asset per leaf; "
        "select one with tree['<collection>'] from ds.gs.expand() or "
        "stac.load_datatree(items)."
    )
    for path, node in tree.subtree_with_keys:
        if node.children:
            if node.data_vars:
                raise RasterValidationError(
                    f"This node holds data but is not a leaf. {hint}", path=path
                )
            continue

        if path == ".":
            continue

        if path.count("/") != 1:
            raise RasterValidationError(
                f"This leaf does not sit at Item/asset. {hint}", path=path
            )


def stac_catalog(tree: xr.DataTree) -> None:
    """Require a STAC catalog: every child a STAC collection, side by side as layers.

        /                      catalog   ← this node
        ├── /s2                collection
        └── /labels            collection

    Raises:
        RasterValidationError: The catalog holds data, or a collection's
            layout differs; the error names the node.
    """
    if tree.data_vars:
        raise RasterValidationError(
            "A STAC catalog holds collections, not data.", path=tree.path
        )

    for name, collection in tree.children.items():
        try:
            stac_collection(collection)
        except RasterValidationError as error:
            error.path = f"{name}/{error.path}"
            raise


def same_grid(tree: xr.DataTree) -> None:
    """Require one grid across the data nodes: CRS, resolution and pixel alignment.

    Extents may differ: nodes on one grid stack by their pixel labels, the
    union of their extents, as `Dataset.gs.combine` concatenates them.

        a: x 500000–500040    b: x 500020–500080    10 m, alignment (0, 0)   → one grid
        c: x 500005–500045                          10 m, alignment (5, 0)   → another

    Raises:
        RasterValidationError: Nodes sit on several grids; the error names each.
    """
    nodes_by_grid: dict[tuple | None, list[str]] = {}
    for leaf in tree.leaves:
        if not leaf.data_vars:
            continue
        path = leaf.relative_to(tree)
        geobox = leaf.to_dataset(inherit="all_coords").gs.geobox
        grid = None
        if geobox is not None:
            grid = (geobox.crs, geobox.resolution, geobox.alignment)
        nodes_by_grid.setdefault(grid, []).append(path)
    if len(nodes_by_grid) == 1:
        return

    lines = []
    for paths in nodes_by_grid.values():
        lines.append(", ".join(paths))
    raise RasterValidationError(
        f"Nodes are on {len(nodes_by_grid)} grids ({'; '.join(lines)}). Resample "
        "them onto one with tree.gs.reproject(geobox), or select one grid's nodes.",
        path=list(nodes_by_grid.values())[1][0],
    )


def same_pixel_meaning(tree: xr.DataTree) -> None:
    """Require each asset of a collection to state one pixel meaning across its Items.

    An asset stacks along the Item dim as one variable, which states one
    `CFPacking`, `CFFlags` and units for every step. A provider changing the
    encoding mid-collection, as Sentinel-2's 2022 offset did, would otherwise
    decode some steps wrong without error.

    Raises:
        RasterValidationError: An asset states another pixel meaning in an Item.
    """
    first_by_variable: dict[str, xr.DataArray] = {}
    for item_name, item in tree.children.items():
        for leaf in item.leaves:
            for name, band in leaf.data_vars.items():
                first = first_by_variable.setdefault(str(name), band)
                if (
                    band.gs.packing != first.gs.packing
                    or band.gs.flags != first.gs.flags
                    or band.attrs.get("units") != first.attrs.get("units")
                ):
                    raise RasterValidationError(
                        "It states another CF packing, flags or units than in the "
                        "collection's first Item, so one variable cannot hold both. "
                        "Decode each Item with mask_and_scale() first, or combine "
                        "the Items of each encoding apart.",
                        path=f"{item_name}/{leaf.name}",
                    )


def same_geobox(tree: xr.DataTree) -> None:
    """Require one GeoBox across the data nodes: the same grid and the same extent.

    Raises:
        RasterValidationError: Nodes sit on several GeoBoxes; the error names each.
    """
    nodes_by_geobox: dict[GeoBox | None, list[str]] = {}
    for leaf in tree.leaves:
        if not leaf.data_vars:
            continue
        geobox = leaf.to_dataset(inherit="all_coords").gs.geobox
        nodes_by_geobox.setdefault(geobox, []).append(leaf.relative_to(tree))
    if len(nodes_by_geobox) == 1:
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
    """Read, reproject and combine one STAC tree.

    Works on a catalog, a collection or an Item node: the members read the
    assets below it, each on its own grid.

    Args:
        data: STAC tree, as `ds.gs.expand()` or `stac.load_datatree` builds it.

    Examples:
        >>> tree["s2"].gs.combine(["B02", "B03", "B04"])
        <xarray.Dataset> ...
    """

    @property
    def spatial_dims(self) -> tuple[str, str] | None:
        """Return the spatial axes of the tree's one GeoBox, or None without one."""
        if self.geobox is None:
            return None

        first = next(leaf for leaf in self._data.leaves if leaf.data_vars)
        return first.to_dataset(inherit="all_coords").gs.spatial_dims

    @property
    def geobox(self) -> GeoBox | None:
        """Return the GeoBox every asset shares, or None for none or several."""
        try:
            self.validate(same_geobox)
        except RasterValidationError:
            return None

        first = next(leaf for leaf in self._data.leaves if leaf.data_vars)
        return first.to_dataset(inherit="all_coords").gs.geobox

    def validate(self, *rules: Rule[DataTree]) -> None:
        """Require every asset georeferenced, then every rule, without reading pixels.

        Assets may sit on different grids, so each needs its own grid ODC
        recognizes, with a CRS. A tree without assets fails.

        Args:
            *rules: Tree rules the operation needs, such as `same_grid` or
                `stac_catalog`.

        Raises:
            RasterValidationError: An asset or a rule is broken. The error
                names the node and how to fix it.

        Examples:
            >>> tree.gs.validate(stac_catalog)
        """
        assets = [leaf for leaf in self._data.leaves if leaf.data_vars]
        if not assets:
            raise RasterValidationError(
                "There are no raster variables to validate. Select or load raster data first."
            )

        for leaf in assets:
            try:
                leaf.to_dataset(inherit="all_coords").gs.validate()
            except RasterValidationError as error:
                error.path = leaf.relative_to(self._data)
                raise

        for rule in rules:
            rule(self._data)

    @property
    def timespan(self) -> DateRange | None:
        """Read the coverage of every Item, from the earliest start to the latest end.

        Each Item states its time on its own node; collections on their own
        timelines widen the span together.

        Returns:
            First and last covered instant, or None where no Item is dated.

        Examples:
            >>> tree.gs.timespan == tree["s2"].gs.combine().gs.timespan
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
        """Read every Item's time label, each instant once, in order.

        Returns:
            Sorted labels across collections, or None where no Item is dated.
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
        """Return the union of the assets' extents, with the Items' time span.

        Assets on different grids need no alignment. This inspects metadata
        without reading pixels.

        Raises:
            RasterValidationError: The tree holds no asset, or an asset has no
                grid with a CRS.
        """
        from odc.geo import CRS

        from geosave.geodata.core.region import GeoRegion

        self.validate()

        # Many Items share a few GeoBoxes; each extent is added once.
        geoboxes: dict[GeoBox, None] = {}
        for leaf in self._data.leaves:
            if leaf.data_vars:
                geobox = cast("GeoBox", leaf.to_dataset(inherit="all_coords").gs.geobox)
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

    def combine(
        self,
        assets: Sequence[str] | None = None,
        item_dim: str = TIME_COORDINATE,
    ) -> Dataset:
        """Combine a STAC collection node into one Dataset, the inverse of `expand`.

        Each Item merges its assets into one raster, inheriting the Item's
        coordinates; the rasters then stack along `item_dim`, and only the
        Item's own coordinates gain that dimension. Collections are layers on
        their own timelines and grids, so each combines on its own.

            /2025-06-01/red, /nir   →   merge   →   raster (y, x)  ┐
            /2025-06-02/red, /nir   →   merge   →   raster (y, x)  ┴ stack → (time, y, x)

        Items on one grid may cover different extents: the Dataset spans their
        union, NaN where an Item has no pixels or lacks an asset. Pixels stack
        as one dask array per variable, as odc-stac builds them, so many Items
        cost no per-Item alignment.

        Args:
            assets: Asset names to combine, such as the 10 m bands of a
                collection whose assets sit on several grids. None combines
                every asset.
            item_dim: Dimension the Items stack along.

        Returns:
            Lazy Dataset of the collection along `item_dim`.

        Raises:
            RasterValidationError: The node is not a STAC collection, or the
                chosen assets sit on several grids (resample with
                `tree.gs.reproject`, or choose one grid's `assets`).

        Examples:
            >>> tree["s2"].gs.combine(["B02", "B03", "B04", "B08"])
            <xarray.Dataset> Dimensions: (time: 2, y: 64, x: 64) ...
        """
        tree = self._data

        # Keep only the chosen assets, so their grids alone must agree.
        if assets is not None:
            tree = tree.filter(lambda node: not node.is_leaf or node.name in assets)

        tree.gs.validate(stac_collection, same_grid, same_pixel_meaning)

        # Item: its assets merge into one raster, with the Item's coordinates
        # and attrs. Every asset inherits the same Item coordinates and
        # same_grid checked the CRS, so nothing needs comparing.
        rasters = []
        for item in tree.children.values():
            item_assets = []
            for leaf in item.leaves:
                item_assets.append(leaf.to_dataset(inherit="all_coords"))
            raster = xr.merge(item_assets, join="outer", compat="override")
            rasters.append(raster.assign_attrs(item.attrs))

        # One grid, so the Items' GeoBoxes union exactly: the extent they all
        # fit in.
        geobox = rasters[0].odc.geobox
        for raster in rasters[1:]:
            geobox = geobox | raster.odc.geobox

        combined = stack_rasters(rasters, geobox, item_dim)

        # The Collection describes the whole Dataset.
        return cast("Dataset", combined.assign_attrs(tree.attrs))

    def to_cog(
        self,
        path: Location,
        *,
        overwrite: bool = False,
        compute: bool = True,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[UPath] | Delayed:
        """Write every asset of a STAC catalog as one COG at its node path.

        Node names are collection ids, Item ids and asset keys, so the tree
        lays the files out: ``/s2/{item id}/red`` becomes
        ``{path}/s2/{item id}/red.tif``.

        Args:
            path: Output folder or filesystem URL.
            overwrite: Replace existing files.
            compute: False returns one delayed batch without reading pixels.
            storage_options: Filesystem options.
            **options: COG creation options.

        Returns:
            Written paths in asset order, or one native delayed batch.

        Raises:
            RasterValidationError: The tree is not a STAC catalog.
            ValueError: An asset cannot be one COG; nothing is written.
            FileExistsError: A destination exists and overwrite is false.

        Examples:
            >>> scene.gs.expand().gs.to_cog("out")
            [UPath('/abs/out/default/20250601T000000_49LEL_-8.9999_111.0001/red.tif'), ...]
        """
        from geosave.geodata.io import storage
        from geosave.geodata.io.formats import geotiff

        self.validate(stac_catalog)

        root = storage.create_upath(path, storage_options)
        files = {}
        for leaf in self._data.leaves:
            name = leaf.relative_to(self._data)
            files[root / f"{name}.tif"] = leaf.to_dataset(inherit="all_coords")
        return geotiff.write_files(
            files, compute=compute, overwrite=overwrite, **options
        )

    def to_catalog(
        self,
        path: Location,
        *,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> GeoDataFrame:
        """Write every asset as a COG and describe each Item, as an item table.

        Args:
            path: Output folder or filesystem URL; files lay out as `to_cog` writes them.
            overwrite: Replace existing files.
            storage_options: Filesystem options.
            **options: COG creation options.

        Returns:
            One row per Item, each asset's href the COG written for it.
        """
        from geosave.geodata.io import storage
        from geosave.geodata.stac.catalog import create_catalog
        from geosave.geodata.stac.publish import StacItem, create_stac_collection

        root = storage.create_upath(path, storage_options)
        items = []
        collections = []
        for collection_id, collection in self._data.children.items():
            collection_items = []
            for item_node in collection.children.values():
                item = StacItem.from_datatree(item_node, collection=collection_id)
                for key, asset in item_node.children.items():
                    href = root / f"{asset.relative_to(self._data)}.tif"
                    raster = asset.to_dataset(inherit="all_coords")
                    item.add_raster_asset(key, raster, str(href.absolute()))
                collection_items.append(item)
            collections.append(create_stac_collection(collection, collection_items))
            items.extend(collection_items)

        self.to_cog(root, overwrite=overwrite, **options)
        return create_catalog(items, collections=collections)

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
            >>> open_raster(tree.gs.to_zarr("scene.zarr")).children.keys()
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
            >>> open_raster(tree.gs.to_netcdf("scene.nc")).children.keys()
        """
        from geosave.geodata.io.formats import netcdf

        return netcdf.write_netcdf(
            self._data,
            path,
            compute=compute,
            overwrite=overwrite,
            storage_options=storage_options,
            **write_options,
        )
