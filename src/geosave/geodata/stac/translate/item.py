"""A STAC Item as the scene its raster files hold.

A scene is what remote sensing calls one acquisition: one collection, at one
instant, on one grid. Files written from one scene are one Item, one Asset
each.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime as DateTime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pystac
import xarray as xr
from shapely import union_all
from shapely.geometry import mapping

from geosave.geodata.io.raster.cogs import CogFile

from . import asset

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox

type ItemTime = DateTime | tuple[DateTime, DateTime]

# A grid spelled as a hashable key, as `("EPSG:32633", (10.0, 0.0, ...), (512, 512))`.
type GridKey = tuple[str, tuple[float, ...], tuple[int, ...]]


# An Asset's key within its Item, as "label", "B04" or "image".
type AssetKey = str
# Absolute path or URL of a saved file, as `absolute_location` spells it.
type Href = str


@dataclass(frozen=True)
class Scene:
    """The saved files of one collection, at one instant, on one grid: one Item.

    Args:
        collection: Collection id the files belong to, or None.
        datetime: Instant the files hold, or None for timeless files.
        grid: Grid the files share, or None where it is not locatable.
        assets: Each Asset's key mapped to the file it is, as
            {
                "image": "/data/s0/optical_20250601.tif",
                "label": "/data/s0/label.tif",
            }

    Examples:
        >>> [scene.stem for scene in scenes(headers)]
        ['forest_20250601', 'forest_20250611']
    """

    collection: str | None
    datetime: pd.Timestamp | None
    grid: GeoBox | None
    assets: dict[AssetKey, Href]

    @property
    def stem(self) -> str:
        """Name the scene after its file, or the stem its files share.

        `forest_20250601_red` and `forest_20250601_nir` share
        `forest_20250601`; files sharing none take their folder's name.
        """
        hrefs = list(self.assets.values())
        stems = [PurePosixPath(href).stem for href in hrefs]
        if len(stems) == 1:
            return stems[0]
        shared = os.path.commonprefix(stems).rstrip("_-.")
        return shared or PurePosixPath(hrefs[0]).parent.name


def scenes(
    headers: Mapping[Href, xr.Dataset], *, collection: str | None = None
) -> list[Scene]:
    """Group saved files into the scenes they make: one per collection, instant and grid.

    Args:
        headers: Each file's href mapped to its raster as `open_raster` opens
            it. No pixel is read.
        collection: Collection of every file. None takes each file's ACDD `id`.

    Returns:
        One scene per collection, instant and grid, in the order the files
        first name them. A file is the asset of its one variable, or
        `"image"` for several.

    Raises:
        ValueError: A file holds several dates, or two files of one scene
            would be the same asset.

    Examples:
        >>> [list(scene.assets) for scene in scenes(headers)]
        [['red', 'nir'], ['red', 'nir']]
    """
    grouped: dict[tuple[str | None, pd.Timestamp | None, GridKey | None], Scene] = {}
    for href, header in headers.items():
        times = header.gs.times
        if times is not None and len(times) != 1:
            raise ValueError(f"{href} holds {len(times)} dates; an Item is one date")
        when = None if times is None else times[0]

        acdd = header.gs.acdd
        stated = collection or (None if acdd is None else acdd.id)
        grid = header.gs.geobox
        found = grouped.setdefault(
            (stated, when, None if grid is None else _grid_key(grid)),
            Scene(stated, when, grid, {}),
        )

        variables = header.gs.variables
        key = variables[0] if len(variables) == 1 else "image"
        if key in found.assets:
            raise ValueError(
                f"{href} and {found.assets[key]} would both be the asset {key!r} "
                f"of one Item; split them with {{band}} or {{time}}"
            )
        found.assets[key] = href
    return list(grouped.values())


def span(headers: Iterable[xr.Dataset]) -> tuple[DateTime, DateTime] | None:
    """Return the first and last instant dated rasters cover, None where none is dated."""
    spans = [raster.gs.timespan for raster in headers if raster.gs.timespan is not None]
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans)


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


def files(item: pystac.Item) -> list[CogFile]:
    """Return the data assets of an Item as named raster file records.

    A variable is named by its asset key, unique within an Item: a one-band
    asset is its key, and band `i` of a several-band asset is `key.i`, as
    odc-stac names them. Band names are metadata, not names.

    Args:
        item: Native STAC Item with absolute asset hrefs.

    Returns:
        One record per data asset, dated by the Item's datetime or interval
        start.

    Examples:
        >>> [file.bands for file in files(item)]
        [('B04',), ('visual.1', 'visual.2', 'visual.3')]
    """
    stamp = item.datetime or item.common_metadata.start_datetime
    when = None if stamp is None else pd.to_datetime(stamp, utc=True).tz_convert(None)
    return [
        CogFile(
            href=entry.href,
            bands=asset.names(key, entry),
            datetime=when,
            collection=item.collection_id,
        )
        for key, entry in item.assets.items()
        if entry.roles is None or "data" in entry.roles
    ]


def attrs(item: pystac.Item) -> dict[str, dict[str, Any]]:
    """Return the CF attrs each variable of an Item takes from its bands.

    Args:
        item: Native STAC Item.

    Returns:
        Variable name, as `files` names it, mapped to the packing, units,
        spectral facts and legend its band states. Variables stating none are
        absent.

    Examples:
        >>> attrs(item)["B04"]
        {'scale_factor': 0.0001, 'add_offset': 0.0, 'common_name': 'red'}
    """
    found = {}
    for key, entry in item.assets.items():
        found |= asset.attrs(key, entry)
    return found


def write(
    scene: Scene,
    headers: Mapping[str, xr.Dataset],
    *,
    id: str,
    when: ItemTime | None = None,
) -> pystac.Item:
    """Build the Item describing one scene.

    Args:
        scene: Files of the scene, keyed by asset.
        headers: Each file's href mapped to its raster as `open_raster`
            opens it. No pixel is read.
        id: Item identity.
        when: Time for a timeless scene, as one instant or a start and end.
            None takes what the rasters' `time` coordinates cover.

    Returns:
        Item with one asset per file, over their joined footprint.

    Raises:
        ValueError: A raster has no locatable grid, or none is dated and
            `when` is None.
    """
    rasters = [headers[href] for href in scene.assets.values()]
    if isinstance(when, tuple):
        start, end = when
    elif when is not None:
        start, end = when, when
    else:
        start, end = covered(rasters, list(scene.assets))

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
        collection=scene.collection,
    )
    for key, href in scene.assets.items():
        asset.write(item, key, href, headers[href])
    return item


def covered(
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
            f"none of the assets {list(keys)} is dated, and a STAC Item needs a "
            f"time; pass datetime="
        )
    instants = cast(
        pd.DatetimeIndex,
        pd.DatetimeIndex(np.concatenate([each.values for each in labels])).unique(),
    )
    if len(instants) == 1:
        stamp = cast(pd.Timestamp, instants[0]).tz_localize("UTC").to_pydatetime()
        return stamp, stamp
    start = min(start for start, _ in spans).replace(tzinfo=UTC)
    end = max(end for _, end in spans).replace(tzinfo=UTC)
    return start, end


def _grid_key(grid: GeoBox) -> GridKey:
    """Spell a grid as a hashable key: CRS, transform and shape."""
    return (str(grid.crs), tuple(grid.transform)[:6], tuple(grid.shape))
