"""Build native STAC Items and Collections from saved rasters."""

from __future__ import annotations

import os

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from datetime import UTC
from datetime import datetime as DateTime
from os import PathLike
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pystac
import xarray as xr
from shapely import union_all
from shapely.geometry import mapping

from geosave_engine.geodata.io.raster.cogs import CogFile
from geosave_engine.geodata.io.readers import open_raster
from geosave_engine.geodata.io.storage import absolute_location

from geosave_engine.geodata.io.vector import stac_geoparquet

from . import extensions

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata import GeoDataFrame

type ItemTime = DateTime | tuple[DateTime, DateTime]
type Saved = Mapping[str, xr.Dataset]

# A grid spelled as a hashable key, as `("EPSG:32633", (10.0, 0.0, ...), (512, 512))`.
type GridKey = tuple[str, tuple[float, ...], tuple[int, ...]]

# What makes files one Item: their collection, their instant and their grid.
type SceneKey = tuple[str | None, pd.Timestamp | None, GridKey | None]

# What STAC calls a saved raster, by its suffix.
_MEDIA_TYPES = {
    ".tif": pystac.MediaType.GEOTIFF,
    ".tiff": pystac.MediaType.GEOTIFF,
    ".jp2": pystac.MediaType.JPEG2000,
    ".png": pystac.MediaType.PNG,
}


def create_collection(
    id: str,
    *,
    description: str,
    license: str = "other",
    title: str | None = None,
    providers: Sequence[pystac.Provider] | None = None,
) -> pystac.Collection:
    """Build a Collection that Items are created into.

    Args:
        id: Collection identity, which its Items carry.
        description: What the Collection holds.
        license: SPDX license identifier, or `"other"`.
        title: Short human-readable name.
        providers: Organizations that produced, processed or host the data.

    Returns:
        Collection covering the whole globe over an unbounded time range,
        until a table states what its rows cover.

    Examples:
        >>> forest = create_collection("forest", description="Forest samples")
        >>> create_items(paths, collection=forest)["collection"].unique().tolist()
        ['forest']
    """
    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
        pystac.TemporalExtent([[None, None]]),
    )
    return pystac.Collection(
        id=id,
        description=description,
        extent=extent,
        title=title,
        license=license,
        providers=None if providers is None else list(providers),
    )


def create_item(
    assets: Mapping[str, str | PathLike[str]],
    *,
    id: str,
    collection: pystac.Collection | None = None,
    datetime: ItemTime | None = None,
    **options: Any,
) -> GeoDataFrame:
    """Build one Item from saved rasters the caller names, as a one-row table.

    Args:
        assets: Asset key mapped to the file or store holding it, as
            `{"label": "s0/label.tif"}`.
        id: Item identity.
        collection: Collection the Item belongs to; the Item takes its id.
            None states none.
        datetime: Time for timeless files, as one instant or a start and end.
            None takes the instants of the files' `time` coordinate, widened
            to what a `TimeSpec` on it says each label covers.
        **options: Read options passed to `open_raster` for every asset.

    Returns:
        Item table of one row, over the files' footprint, dated by one instant
        or by a start and end, declaring the extensions its assets use. No
        pixel is read.

    Raises:
        ValueError: An asset has no locatable grid or names no raster format,
            or no asset is dated and `datetime` is None.

    Examples:
        >>> create_item({"label": "s0/label.tif"}, id="s0")["id"].tolist()
        ['s0']
    """
    with ExitStack() as opened:
        named = {key: _open(opened, path, options) for key, path in assets.items()}
        item = _item(named, id=id, collection=_identity(collection), when=datetime)
    return stac_geoparquet.from_items([item])


def create_items(
    files: Iterable[CogFile],
    *,
    id: str | None = None,
    collection: pystac.Collection | None = None,
    datetime: ItemTime | None = None,
    **options: Any,
) -> GeoDataFrame:
    """Build the item table of the files one or more `to_cog` calls wrote.

    Files sharing a collection, an instant and a grid are one Item, one Asset
    each: the band's name for a single-band file, `"image"` for several bands.

    Args:
        files: Records `to_cog` returned.
        id: `GeoAnchor.format` template filled from each Item's raster, as
            `"{lat:.2f}N_{start:%Y%m%d}"`, or, for files naming their
            collections, a prefix every id takes. None names each Item after
            its file, or the stem its files share.
        collection: Collection of files that name none; they take its id.
        datetime: Time for undated files. None gives them the span the dated
            files cover.
        **options: Read options passed to `open_raster` for every file.

    Returns:
        Item table, one row per Item, in the order the files first name them.
        No pixel is read.

    Raises:
        TypeError: An entry is not a `CogFile`.
        ValueError: No file is given; `collection` is given for files naming
            their own; two files of one Item share an asset key; ids repeat;
            or a file is undated, none is dated and `datetime` is None.

    Examples:
        >>> create_items(ds.gs.to_cog("samples/forest_{time:%Y%m%d}.tif"))["id"].tolist()
        ['forest_20250601', 'forest_20250611']
        >>> files = sample.gs.to_cog("samples/s0/{collection}_{time:%Y%m%d}.tif")
        >>> create_items(files, id="s0")["collection"].tolist()
        ['optical', 'optical', 'label']
    """
    return stac_geoparquet.from_items(
        _records_items(
            files, id=id, collection=collection, when=datetime, options=options
        )
    )


def cog_files(item: Mapping[str, Any]) -> list[CogFile]:
    """Spell the data assets of one Item as the records a write produced.

    Args:
        item: One STAC Item as a dict, its asset hrefs absolute.

    Returns:
        One record per data asset, its bands named as the asset's `eo:bands`
        lists them, dated by the Item's `datetime` or `start_datetime`.

    Examples:
        >>> [f.bands for f in cog_files(item.to_dict())]
        [('red',), ('nir',)]
    """
    when = instant(item)
    records = []
    for key, asset in item["assets"].items():
        roles = asset.get("roles")
        if roles is not None and "data" not in roles:
            continue
        names = [band["name"] for band in asset.get("eo:bands") or []] or [key]
        records.append(
            CogFile(
                href=asset["href"],
                bands=tuple(names),
                datetime=when,
                collection=item.get("collection"),
            )
        )
    return records


def instant(item: Mapping[str, Any]) -> pd.Timestamp | None:
    """Return an Item dict's instant: its `datetime`, else `start_datetime`.

    Args:
        item: One Item as a dict, as a window's `sources` holds it.

    Returns:
        The instant, naive UTC, or None when the Item states neither.

    Examples:
        >>> instant({"properties": {"datetime": "2025-06-01T00:00:00Z"}})
        Timestamp('2025-06-01 00:00:00')
    """
    properties = item.get("properties", {})
    stamp = properties.get("datetime") or properties.get("start_datetime")
    if stamp is None:
        return None
    return pd.to_datetime(stamp, utc=True).tz_convert(None)


def default_key(raster: xr.Dataset) -> str:
    """Name the asset holding a whole raster.

    Args:
        raster: Raster one file or store holds.

    Returns:
        The raster's variable where it has exactly one, else `"image"`.

    Examples:
        >>> default_key(scene[["red"]]), default_key(scene)
        ('red', 'image')
    """
    variables = raster.gs.variables
    if len(variables) == 1:
        return variables[0]
    return "image"


def _media_type(href: str) -> str:
    """Name the media type of the raster file an href points at."""
    found = _MEDIA_TYPES.get(PurePosixPath(href).suffix.lower())
    if found is None:
        raise ValueError(
            f"{href} names no raster file an Item lists; save the rasters with "
            f"to_cog and pass the files it returned"
        )
    return found


def _identity(collection: pystac.Collection | None) -> str | None:
    """Return the id an Item of this Collection carries, None for no Collection."""
    return None if collection is None else collection.id


def _open(
    opened: ExitStack, path: str | PathLike[str], options: Mapping[str, Any]
) -> tuple[str, xr.Dataset]:
    """Open one saved raster lazily, to be closed when `opened` exits."""
    raster = opened.enter_context(open_raster(path, **options))
    return absolute_location(path), raster


def _records_items(
    files: Iterable[CogFile],
    *,
    id: str | None,
    collection: pystac.Collection | None,
    when: ItemTime | None,
    options: Mapping[str, Any],
) -> list[pystac.Item]:
    """Build one Item per collection and instant the records name."""
    records = list(files)
    if not records:
        raise ValueError("create_items needs the CogFile records to_cog returned")
    if not all(isinstance(file, CogFile) for file in records):
        raise TypeError(
            "create_items takes the CogFile records to_cog returned; a path "
            "states neither its bands nor its instant"
        )
    named = any(file.collection is not None for file in records)
    if named and collection is not None:
        raise ValueError(
            "these files already name their collections, the layers they were "
            "written from; drop collection="
        )

    with ExitStack() as opened:
        rasters = {file.href: _open(opened, file.href, options)[1] for file in records}
        # One Item is one scene: one layer, one instant, one grid.
        scenes: dict[SceneKey, list[CogFile]] = {}
        for file in records:
            grid = rasters[file.href].gs.geobox
            key = (
                file.collection,
                file.datetime,
                None if grid is None else _grid_key(grid),
            )
            scenes.setdefault(key, []).append(file)
        spans = [r.gs.timespan for r in rasters.values() if r.gs.timespan is not None]
        fallback = when
        if fallback is None and spans:
            fallback = (min(start for start, _ in spans), max(end for _, end in spans))

        items = []
        for (layer, stamp, _), members in scenes.items():
            assets: dict[str, tuple[str, xr.Dataset]] = {}
            for file in members:
                key = file.bands[0] if len(file.bands) == 1 else "image"
                if key in assets:
                    raise ValueError(
                        f"{file.href} and {assets[key][0]} would both be the asset "
                        f"{key!r} of one Item; split them with {{band}} or {{time}}"
                    )
                assets[key] = (file.href, rasters[file.href])
            # One file is named after itself; several after the stem they share,
            # as `forest_20250601` for `forest_20250601_red` and `_nir`.
            stems = [PurePosixPath(file.href).stem for file in members]
            shared = os.path.commonprefix(stems).rstrip("_-.")
            stem = (
                stems[0]
                if len(stems) == 1
                else shared or PurePosixPath(members[0].href).parent.name
            )
            if id is None:
                item_id = stem
            elif named:
                item_id = f"{id}/{stem}"
            else:
                item_id = next(iter(assets.values()))[1].gs.anchor.format(id)
            items.append(
                _item(
                    assets,
                    id=item_id,
                    collection=layer if named else _identity(collection),
                    when=None if stamp is not None else fallback,
                )
            )

    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"id {id!r} gives several Items the same id {ids}; add a time field "
            f"such as {{start:%Y%m%d}}"
        )
    return items


def _grid_key(grid: GeoBox) -> GridKey:
    """Spell a grid as a hashable key: CRS, transform and shape."""
    return (str(grid.crs), tuple(grid.transform)[:6], tuple(grid.shape))


def _item(
    named: Mapping[str, tuple[str, xr.Dataset]],
    *,
    id: str,
    collection: str | None,
    when: ItemTime | None,
) -> pystac.Item:
    """Build one Item from opened rasters keyed by asset name.

    Args:
        named: Asset key mapped to the href and the raster opened from it.
        id: Item identity.
        collection: Collection id the Item carries, or None.
        when: Time for timeless rasters.

    Returns:
        Item with one asset per raster, over their joined footprint.

    Raises:
        ValueError: A raster has no locatable grid, or none is dated and
            `when` is None.
    """
    rasters = [raster for _, raster in named.values()]
    if isinstance(when, tuple):
        start, end = when
    elif when is not None:
        start, end = when, when
    else:
        start, end = _covered(rasters, list(named))

    grids = [raster.gs.geobox for raster in rasters]
    if any(grid is None or grid.crs is None for grid in grids):
        footprint = None
    else:
        footprint = union_all([grid.geographic_extent.geom for grid in grids])

    # STAC dates an Item by one instant, or by a start and an end, never both.
    ranged = start != end
    item = pystac.Item(
        id,
        None if footprint is None else mapping(footprint),
        None if footprint is None else list(footprint.bounds),
        None if ranged else start,
        {},
        start_datetime=start if ranged else None,
        end_datetime=end if ranged else None,
        collection=collection,
    )
    for key, (href, raster) in named.items():
        # An extension declares its schema on the asset's owner, so add first.
        item.add_asset(
            key, pystac.Asset(href, media_type=_media_type(href), roles=["data"])
        )
        for extension in extensions.ASSET:
            extension.write(item.assets[key], raster)
    return item


def _covered(
    rasters: Sequence[xr.Dataset], keys: Sequence[str]
) -> tuple[DateTime, DateTime]:
    """Return the first and last instant the rasters of one Item cover.

    Args:
        rasters: Rasters of one Item, as opened from their files.
        keys: Their asset keys, for the error.

    Returns:
        (first, last) in UTC. Rasters holding one time label between them
        cover that instant; rasters holding several cover what `gs.timespan`
        says, so a monthly label covers its month.

    Raises:
        ValueError: No raster is dated.
    """
    labels = [raster.gs.times for raster in rasters if raster.gs.times is not None]
    spans = [raster.gs.timespan for raster in rasters if raster.gs.timespan is not None]
    if not spans:
        raise ValueError(
            f"none of the assets {keys} is dated, and a STAC Item needs a "
            f"time; pass datetime="
        )
    instants = pd.DatetimeIndex(
        np.concatenate([each.values for each in labels])
    ).unique()
    if len(instants) == 1:
        stamp = instants[0].tz_localize("UTC").to_pydatetime()
        return stamp, stamp
    start = min(start for start, _ in spans).replace(tzinfo=UTC)
    end = max(end for _, end in spans).replace(tzinfo=UTC)
    return start, end
