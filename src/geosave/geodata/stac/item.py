"""Interpret and construct native STAC Item metadata without raster I/O."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as DateTime
from typing import TYPE_CHECKING, Any, TypedDict

import pandas as pd
import pystac

from geosave.geodata.stac import band

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odc.geo.geom import Geometry


# The time of an Item whose rasters state none. STAC dates every Item, so a
# timeless one, as an elevation model, spans all of time and so overlaps every
# window; `read_timespan` and `read_instant` read it back as None.
TIMELESS = (
    DateTime(1, 1, 1, tzinfo=UTC),
    DateTime(9999, 12, 31, 23, 59, 59, tzinfo=UTC),
)

# Item properties a raster states in its own structure, so they are never kept
# as copied properties: the time coordinate states the dates, the grid states
# `proj:*`, each variable's attrs state its band, and a new Item has its own
# lifecycle dates.
_STATED_BY_RASTER = (
    "datetime",
    "start_datetime",
    "end_datetime",
    "created",
    "updated",
    "bands",
    "eo:bands",
    "raster:bands",
)


def select_item_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Keep the Item properties about its date and acquisition, not its pixels.

    Args:
        properties: An Item's properties.

    Returns:
        New mapping without the dates, grid and band fields a raster states
        in its own coordinates and attrs.

    Examples:
        >>> select_item_properties({"datetime": "2025-01-15", "platform": "sentinel-2a"})
        {'platform': 'sentinel-2a'}
    """
    kept = {}
    for key, value in properties.items():
        if key in _STATED_BY_RASTER or key.startswith("proj:"):
            continue
        kept[key] = value
    return kept


def read_timespan(
    instant: DateTime | str | None,
    start: DateTime | str | None = None,
    end: DateTime | str | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the first and last instant an Item covers.

    Args:
        instant: Nominal observation timestamp, or None.
        start: Explicit coverage start, or None to use the instant.
        end: Explicit coverage end, or None to use the instant.

    Returns:
        (start, end), naive UTC: its `start_datetime` and `end_datetime`,
        else its `datetime` twice. None for a timeless Item, which spans
        `TIMELESS` or states no time.

    Examples:
        >>> read_timespan("2025-06-01T00:00:00Z")
        (Timestamp('2025-06-01 00:00:00'), Timestamp('2025-06-01 00:00:00'))
    """
    start = start if start is not None else instant
    end = end if end is not None else instant
    if start is None or end is None:
        return None
    # Compared as Timestamps first, since TIMELESS lies past pandas' nanosecond range.
    if (pd.Timestamp(start), pd.Timestamp(end)) == TIMELESS:
        return None
    return (
        pd.to_datetime(start, utc=True).tz_convert(None),
        pd.to_datetime(end, utc=True).tz_convert(None),
    )


def read_item_period(item: pystac.Item) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the period an Item covers: its start and end, else its instant twice.

    Args:
        item: Native STAC Item.

    Returns:
        (start, end), naive UTC, or None for a timeless Item.

    Examples:
        >>> read_item_period(composite)
        (Timestamp('2025-06-01 00:00:00'), Timestamp('2025-07-01 00:00:00'))
    """
    return read_timespan(
        item.datetime,
        item.properties.get("start_datetime"),
        item.properties.get("end_datetime"),
    )


def read_instant(item: pystac.Item) -> pd.Timestamp | None:
    """Return the instant an Item is dated by: its `datetime`, else its start.

    Args:
        item: Native STAC Item.

    Returns:
        The instant, naive UTC, or None for a timeless Item.

    Examples:
        >>> read_instant(item)
        Timestamp('2025-06-01 00:00:00')
    """
    covered = read_item_period(item)
    if covered is None:
        return None
    nominal = item.datetime
    return pd.to_datetime(nominal, utc=True).tz_convert(None) if nominal else covered[0]


def read_footprint(items: Sequence[pystac.Item]) -> Geometry | None:
    """Return the area the Items' geometries cover, as one lon/lat geometry.

    Args:
        items: Items whose `geometry` states their region.

    Returns:
        Union of their geometries in EPSG:4326, or None where none states one.

    Examples:
        >>> read_footprint([item]).crs
        CRS('EPSG:4326')
    """
    from odc.geo.geom import Geometry
    from shapely.geometry import shape
    from shapely.ops import unary_union

    shapes = []
    for entry in items:
        if entry.geometry is not None:
            shapes.append(shape(entry.geometry))
    if not shapes:
        return None
    return Geometry(unary_union(shapes), "EPSG:4326")


def select_data_assets(item: pystac.Item) -> dict[str, pystac.Asset]:
    """Select data assets in published order, including assets without stated roles.

    Explicit roles without `data` exclude an asset from native raster reading,
    grid selection, and variable naming. The Item and its assets are not changed.
    """
    return {
        key: entry
        for key, entry in item.assets.items()
        if entry.roles is None or "data" in entry.roles
    }


def read_variable_names(item: pystac.Item) -> dict[str, tuple[str, ...]]:
    """Name eligible data assets for native raster reading.

    These rules are independent of ODC loader configuration and aliases.
    A one-band asset is its key. Band `i` of a several-band asset takes the
    first name it states, `name` then `common_name`, that no asset key spells
    and no asset of fewer bands, or of an earlier key, claims first; a band
    claiming none is `key.i`.

    Args:
        item: Native STAC Item.

    Returns:
        Asset key mapped to its variable names, in band order.

    Examples:
        >>> read_variable_names(item)
        {'B04': ('B04',), 'visual': ('visual.1', 'visual.2', 'visual.3')}
    """
    assets = select_data_assets(item)
    listed = {key: band.read_aliases(entry) for key, entry in assets.items()}

    # Each name goes to the band of the asset with the fewest bands, then the
    # first key; names spelling an asset key stay with that asset.
    claims: dict[str, list[tuple[int, str, int]]] = {}
    for key, bands in listed.items():
        for index, aliases in enumerate(bands, start=1):
            for alias in aliases:
                if alias not in assets:
                    claims.setdefault(alias, []).append((len(bands), key, index))
    owner = {alias: min(claimants)[1:] for alias, claimants in claims.items()}

    found = {}
    for key, bands in listed.items():
        count = len(bands)
        if count <= 1:
            found[key] = (key,)
            if count == 1:
                for alias in bands[0]:
                    if alias == key or owner.get(alias) == (key, 1):
                        found[key] = (alias,)
                        break
            continue
        variables = []
        for index in range(1, count + 1):
            # Prefer the published band name before its common name.
            taken = [alias for alias, band in owner.items() if band == (key, index)]
            variables.append(taken[0] if taken else f"{key}.{index}")
        found[key] = tuple(variables)
    return found


class ItemTime(TypedDict):
    """UTC observation instant and optional paired coverage boundaries."""

    datetime: DateTime | None
    start_datetime: DateTime | None
    end_datetime: DateTime | None


def normalize_datetime(value: DateTime | str) -> DateTime:
    """Convert a finite timestamp to UTC; interpret naive timestamps as UTC."""
    timestamp = pd.Timestamp(value)
    if not isinstance(timestamp, pd.Timestamp):
        raise ValueError("Item datetime must be finite")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def normalize_time(
    instant: DateTime | str | None,
    start: DateTime | str | None,
    end: DateTime | str | None,
) -> ItemTime:
    """Normalize an instant, an interval, or an instant within an interval.

    Args:
        instant: Nominal observation timestamp, or None for interval-only data.
        start: Inclusive coverage start; requires end.
        end: Coverage end; requires start.

    Returns:
        Named UTC datetime fields; absent instant or boundaries remain None.

    Raises:
        ValueError: Time is missing, nonfinite, incomplete, reversed, or the
            nominal instant lies outside the interval.
    """
    if (start is None) != (end is None):
        raise ValueError("provide both start_datetime and end_datetime")
    if instant is None and start is None:
        raise ValueError(
            "supply input_datetime or both start_datetime and end_datetime"
        )

    instant = None if instant is None else normalize_datetime(instant)
    start = None if start is None else normalize_datetime(start)
    end = None if end is None else normalize_datetime(end)
    if start is not None and end is not None:
        if start > end:
            raise ValueError("Item interval must have ordered boundaries")
        if instant is not None and not start <= instant <= end:
            raise ValueError("Item datetime lies outside the stated coverage")

    return {"datetime": instant, "start_datetime": start, "end_datetime": end}


def create_item(
    *,
    id: str,
    time: ItemTime,
    properties: dict[str, Any],
    collection: str | None = None,
) -> pystac.Item:
    """Construct an Item from normalized time and acquisition properties.

    Geometry and Assets can be attached with native PySTAC methods. Properties
    are copied; time is stated exclusively by the explicit time argument.
    """
    from copy import deepcopy

    fields = deepcopy(properties)
    for key in ("datetime", "start_datetime", "end_datetime"):
        if key in fields:
            raise ValueError(f"provide {key} through time, not properties")
    start = time["start_datetime"]
    end = time["end_datetime"]
    if start is not None and end is not None:
        fields["start_datetime"] = start.isoformat()
        fields["end_datetime"] = end.isoformat()
    return pystac.Item(id, None, None, time["datetime"], fields, collection=collection)
