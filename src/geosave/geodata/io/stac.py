"""Read raster assets and write rasters with native STAC Item metadata."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Unpack

import pystac
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave.geodata.io.formats.geotiff import COGWriteOptions
from geosave.geodata.io.storage import location
from geosave.geodata.stac import band, item


def open_scene(entry: pystac.Item, **options: Any) -> xr.DataTree:
    """Open one Item's data assets as a scene tree, one node per grid, at the Item's date.

    The assets are the scene's files: each opens with the GDAL reader on its
    own grid, takes its band names and STAC band metadata, and merges with
    the files sharing that grid into one grid node, named after EOPF (`r10m`).
    No pixel is resampled.

    Args:
        entry: Native STAC Item with resolvable asset hrefs.
        **options: Forwarded to each asset's reader, such as `chunks` or
            `mask_and_scale`. `chunks` defaults to `{}`: each file's own
            Dask chunks.

    Returns:
        DataTree of grid nodes, each holding the Item's variables on that
        grid, its date as a `time` coordinate and its properties on
        `stac_properties`; closing it closes every file.

    Raises:
        ValueError: A file has no grid, in itself or in its asset's `proj:*`
            fields, files on one grid differ in CRS, or a file's band count
            disagrees with its asset.
    """
    from geosave.geodata.core.raster.grid import name_grid
    from geosave.geodata.transform.combine import merge

    # Scenes join along time with concat, which loads arrays Dask does not hold.
    options.setdefault("chunks", {})
    names = item.read_variable_names(entry)
    with ExitStack() as opened:
        # 1. Each file on its own grid, dated alike so their coordinates agree
        files_by_grid: dict[str, list[xr.Dataset]] = {}
        for key, asset in item.select_data_assets(entry).items():
            raster = _open_asset(asset, names[key], key, opened, **options)
            grid = raster.odc.geobox
            if grid is None or grid.crs is None:
                raise ValueError(
                    f"{entry.id}'s asset {key!r} has no grid in its file; "
                    "its proj:code, proj:shape and proj:transform fields place it"
                )
            files_by_grid.setdefault(name_grid(grid), []).append(
                item.stamp_item_time(raster, entry)
            )

        # 2. One node per grid, its files merged
        nodes = {}
        for name, files in files_by_grid.items():
            nodes[name] = merge(files) if len(files) > 1 else files[0]
        scene = xr.DataTree.from_dict(nodes)
        scene.set_close(opened.pop_all().close)
        return scene


def read_scene_pixels(
    entry: pystac.Item,
    geobox: GeoBox,
    variables: Sequence[str] | None = None,
) -> xr.Dataset:
    """Read an Item's pixels inside `geobox`, each file from the overview it needs.

    Only the window of each file that `geobox` overlaps is read, from the
    coarsest overview still at least as fine as `geobox`, then reprojected onto it.

    Args:
        entry: Native STAC Item.
        geobox: Grid to read onto, such as a map's view.
        variables: Variables to read. None reads every data asset.

    Returns:
        Dataset on `geobox`, one float32 variable per requested name, nodata
        and pixels outside the files as NaN.

    Raises:
        KeyError: A requested variable is not in the Item.
    """
    import numpy as np

    from geosave.geodata.transform.combine import merge

    names_by_asset = item.read_variable_names(entry)
    known_names = []
    for names in names_by_asset.values():
        known_names.extend(names)
    requested_names = known_names if variables is None else list(variables)
    for name in requested_names:
        if name not in known_names:
            raise KeyError(f"{entry.id} has no variable {name!r}")

    rasters = []
    with ExitStack() as opened:
        for key, asset in item.select_data_assets(entry).items():
            asset_names = []
            for name in names_by_asset[key]:
                if name in requested_names:
                    asset_names.append(name)
            if not asset_names:
                continue
            raster = _read_asset_pixels(
                asset, names_by_asset[key], key, asset_names, geobox, opened
            )
            if raster is not None:
                rasters.append(raster)

    # Variables whose files miss the view are NaN, without a read.
    read_names = set()
    for raster in rasters:
        read_names.update(raster.data_vars)
    missing = {}
    for name in requested_names:
        if name not in read_names:
            missing[name] = (("y", "x"), np.full(geobox.shape, np.nan, "float32"))
    if missing:
        rasters.append(xr.Dataset(missing).gs.write_grid(geobox))
    pixels = merge(rasters) if len(rasters) > 1 else rasters[0]
    return pixels[requested_names]


def _read_asset_pixels(
    asset: pystac.Asset,
    names: Sequence[str],
    key: str,
    asset_names: list[str],
    geobox: GeoBox,
    opened: ExitStack,
) -> xr.Dataset | None:
    """Read `asset_names` of one asset onto `geobox`; None when the file misses it."""
    import numpy as np
    from odc.geo.overlap import compute_reproject_roi
    from odc.geo.roi import roi_is_empty
    from odc.geo.xr import xr_coords, xr_reproject

    from geosave.geodata.conventions import CRS_COORDINATE
    from geosave.geodata.io import rasterio
    from geosave.geodata.stac.extensions import projection

    # 1. A grid the asset states answers a miss without opening the file
    stated = projection.read_geobox(asset)
    if stated is not None and stated.crs is not None:
        if roi_is_empty(compute_reproject_roi(stated, geobox).roi_src):
            return None

    # 2. The base image's header: where the view falls, and how much it shrinks
    raster = _open_asset(asset, names, key, opened, chunks=None)
    roi = compute_reproject_roi(raster.odc.geobox, geobox)
    if roi_is_empty(roi.roi_src):
        return None

    # 3. The coarsest overview whose pixels are still no larger than the view's
    href = asset.get_absolute_href() or asset.href
    overview_level = None
    for level, factor in enumerate(rasterio.read_overview_factors(location(href))):
        if factor <= roi.read_shrink:
            overview_level = level
    if overview_level is not None:
        raster = _open_asset(
            asset, names, key, opened, chunks=None, overview_level=overview_level
        )
        roi = compute_reproject_roi(raster.odc.geobox, geobox)

    # 4. Read only the window, then put it on the view's grid
    window = raster[asset_names].isel(y=roi.roi_src[0], x=roi.roi_src[1])
    window = window.gs.to_nan().astype("float32").load()
    pixels = xr_reproject(window, geobox, resampling="nearest", dst_nodata=np.nan)
    # The file's own CRS attrs survive the warp; the pixels now sit on `geobox`.
    return pixels.odc.assign_crs(geobox.crs).assign_coords(
        {CRS_COORDINATE: xr_coords(geobox)[CRS_COORDINATE]}
    )


def _open_asset(
    asset: pystac.Asset,
    names: Sequence[str],
    key: str,
    opened: ExitStack,
    **options: Any,
) -> xr.Dataset:
    """Open one asset and apply its band identities and published metadata.

    `options` reach the file's reader, such as `chunks` or `overview_level`.
    """
    from geosave.geodata.io.openers import open_raster

    href = asset.get_absolute_href() or asset.href
    raster = open_raster(location(href), **options)
    opened.callback(raster.close)

    raster = _place_on_stated_grid(raster, asset, href)
    raster = rename_bands(raster, names, source=href, fallback=key)
    for index, name in enumerate(raster.data_vars):
        variable = raster[name]
        metadata = band.read_attrs(variable, asset, index)
        raster[name] = variable.assign_attrs({**metadata, **variable.attrs})

    return raster


def _place_on_stated_grid(
    raster: xr.Dataset, asset: pystac.Asset, href: str
) -> xr.Dataset:
    """Give a file without a grid the one its asset states; warn when the two disagree.

    The file's own grid is the truth, as publication reads written headers. A
    stated grid whose shape differs from the file's belongs to other pixels and
    is not assigned.
    """
    import warnings

    from odc.geo.xr import xr_coords

    from geosave.geodata.stac.extensions import projection

    stated = projection.read_geobox(asset)
    if stated is None or stated.crs is None:
        return raster
    own = raster.odc.geobox
    if own is not None and own.crs is not None:
        if own != stated:
            warnings.warn(
                f"{href} states a grid in its file and another in its asset; "
                f"the file's is kept: file {own}, asset {stated}",
                UserWarning,
                stacklevel=3,
            )
        return raster
    if tuple(stated.shape) != (raster.sizes["y"], raster.sizes["x"]):
        return raster
    return raster.assign_coords(xr_coords(stated))


def write_assets(
    raster: xr.Dataset | xr.DataTree,
    path: Location,
    *,
    path_format: str | Mapping[str, str] | None = None,
    collection: str | None = None,
    collections: Sequence[pystac.Collection] = (),
    id_format: str,
    datetime: ItemTime | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[COGWriteOptions],
) -> GeoDataFrame:
    """Write a raster's COGs and return the item table of what was written.

    The files are the ones `to_cog` writes for the same `path_format`. Each
    collection's dates become Items whose assets are that date's files across
    every grid node, each stating its own grid. Item metadata, identities and
    file requirements are checked before any pixel is written.

    Args:
        raster: A tree of collections, one collection's tree of grid nodes,
            or a Dataset.
        path: Folder the files go under.
        path_format: Relative path ending in .tif, or one per collection. None
            takes `paths.default_path_format`.
        collection: Collection id of a raster that is one collection.
        collections: Collections the table carries; each one's extent is
            set to cover its written Items.
        id_format: Item id template, as `stac.item.create_item`
            fills it.
        datetime: Date or interval for selections stating no time.
        overwrite: Replace files that exist.
        storage_options: Credentials and connection options for the files.
        **options: GeoTIFF creation options.

    Returns:
        In-memory Item table, one row per collection and date. Persist the
        table separately with its `gs.to_geoparquet` method.

    Raises:
        ValueError: A destination is not a GeoTIFF, a mapping misses a
            collection, destinations collide, or a selection is not one
            observation on a locatable grid.
        FileExistsError: A destination exists and `overwrite` is false.
    """
    import pystac

    from geosave.geodata.io.formats import geotiff
    from geosave.geodata.io.paths import expand_paths, split_by_collection
    from geosave.geodata.io.storage import absolute_location, location
    from geosave.geodata.stac import item as stac_item
    from geosave.geodata.stac.catalog import from_items

    # 1. Each collection laid out as COGs, one destination per file
    root = location(path, storage_options)
    files_by_collection = {}
    files = {}
    for name, member in split_by_collection(raster, collection):
        located = expand_paths(member, root, path_format, collection=name)
        for destination, selected in located.items():
            if destination.suffix.lower() not in geotiff.FILE_SUFFIXES:
                raise ValueError(
                    f"catalog assets are COGs, so {destination.name} must end in .tif; "
                    "write a Zarr or netCDF store with to_zarr or to_netcdf"
                )
            if destination in files:
                raise ValueError(
                    f"path selections collide at {destination}; add {{collection}} to the path format"
                )
            files[destination] = selected
        files_by_collection[name] = located

    # 2. One Item per collection and date, from the selections about to be written
    items = []
    for name, located in files_by_collection.items():
        for observation in group_files_by_date(located):
            items.append(
                stac_item.create_item(
                    observation, id_format=id_format, collection=name, datetime=datetime
                )
            )
    covering = []
    for covered_collection in collections:
        collection_items = [
            entry for entry in items if entry.collection_id == covered_collection.id
        ]
        covered = covered_collection.clone()
        if collection_items:
            covered.extent = pystac.Extent.from_items(collection_items)
        covering.append(covered)
    catalog = from_items(items, collections=covering)

    # 3. Every file written, then rows of what the writer actually wrote
    written = geotiff.write_files(files, compute=True, overwrite=overwrite, **options)
    hrefs = {}
    for destination, saved in zip(files, written, strict=True):
        original = absolute_location(destination)
        href = absolute_location(saved)
        if original != href:
            hrefs[original] = href
    if not hrefs:
        return catalog
    for entry in items:
        for asset in entry.assets.values():
            asset.href = hrefs.get(asset.href, asset.href)
    return from_items(items, collections=covering)


if TYPE_CHECKING:
    from upath import UPath

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.io.storage import Location, StorageOptions
    from geosave.geodata.stac.item import ItemTime


def group_files_by_date(
    files: Mapping[UPath, xr.Dataset],
) -> list[dict[str, tuple[UPath, xr.Dataset]]]:
    """Group written files into observations, one per date, keyed by asset.

    A file holding one variable is the asset of that name; one holding
    several is `"image"`.

    Args:
        files: Written paths mapped to the selections they hold.

    Returns:
        One mapping of asset key to path and selection per date, in the
        order the dates first appear.

    Raises:
        ValueError: Two files of one date name the same asset.
    """
    observations: dict[object, dict[str, tuple[UPath, xr.Dataset]]] = {}
    for path, raster in files.items():
        variables = list(raster.data_vars)
        key = str(variables[0]) if len(variables) == 1 else "image"
        date = (
            raster.gs.times[0].to_datetime64() if raster.gs.times is not None else None
        )
        assets = observations.setdefault(date, {})
        if key in assets:
            raise ValueError(
                f"two files of one date hold the asset {key!r}; a file holding "
                "several variables is the Item's one image, so split them with "
                "{band} in the template"
            )
        assets[key] = (path, raster)
    return list(observations.values())


def rename_bands(
    raster: xr.Dataset,
    names: Sequence[Hashable],
    *,
    source: str,
    fallback: str | None = None,
) -> xr.Dataset:
    """Name file bands by position, expanding a lone name for a multiband file.

    Args:
        raster: Opened file with variables in stored band order.
        names: Names stated by the file header or catalog.
        source: File location for a band-count error.
        fallback: Prefix for unnamed bands; defaults to the lone stated name.

    Returns:
        Dataset with named variables and unchanged pixels.

    Raises:
        ValueError: The file and stated band counts disagree.
    """
    found = list(raster.data_vars)
    stated = list(names)
    if len(stated) == 1 and len(found) > 1:
        prefix = stated[0] if fallback is None else fallback
        stated = [f"{prefix}.{index}" for index in range(1, len(found) + 1)]
    if len(found) != len(stated):
        raise ValueError(
            f"{source} holds {len(found)} bands but its metadata names {len(stated)}: {stated}"
        )

    return raster.rename(dict(zip(found, stated, strict=True)))
