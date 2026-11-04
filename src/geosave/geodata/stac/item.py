"""Translate between STAC Items and their rasters.

An Item is one observation: one date, its files as assets. Writing builds it
from rasters about to be saved, with the STAC metadata they carry: the
`stac_properties` coordinate and the `StacMetadata` root attrs. Reading names
an Item's assets and dates for opening them as xarray.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from datetime import UTC
from datetime import datetime as DateTime
from typing import Any, cast

import numpy as np
import pandas as pd
import pystac
import xarray as xr
from odc.geo.geobox import GeoBox
from shapely import union_all
from shapely.geometry import mapping
from upath import UPath

import geosave.geodata.attrs as attrs
from geosave import __path__ as _package_paths
from geosave.geodata.conventions import STAC_PROPERTIES, TIME_COORDINATE
from geosave.geodata.core.anchor import GeoAnchor
from geosave.geodata.core.raster.time import assign_bounds, read_bounds
from geosave.geodata.stac import asset, band
from geosave.geodata.utils.template import fields_named
from geosave.geodata.warnings import TimelessItemWarning

# An Item's time: one instant, or a start and an end.
type ItemTime = DateTime | tuple[DateTime, DateTime]


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


def select_item_properties(properties: Mapping[str, Any]) -> dict[str, Any]:
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


def stamp_properties_coordinate[T: xr.Dataset | xr.DataArray](
    raster: T, properties: Sequence[Mapping[str, Any]]
) -> T:
    """Record each date's Item properties as JSON on the `stac_properties` coordinate.

    Args:
        raster: Raster carrying a `time` coordinate.
        properties: One mapping per `time` label, in label order.

    Returns:
        Raster with `stac_properties` along `time`, or scalar for a scalar time.

    Raises:
        ValueError: The mappings do not match the `time` labels one to one.

    Examples:
        >>> stamp_properties_coordinate(scene, [{"eo:cloud_cover": 20.0}]).stac_properties.item()
        '{"eo:cloud_cover": 20.0}'
    """
    texts = [json.dumps(dict(entry)) for entry in properties]
    labels = raster[TIME_COORDINATE]
    if labels.ndim == 0:
        if len(texts) != 1:
            raise ValueError("a scalar time takes one properties mapping")
        return raster.assign_coords({STAC_PROPERTIES: texts[0]})
    if len(texts) != labels.size:
        raise ValueError(
            f"{len(texts)} properties mappings for {labels.size} time labels"
        )
    return raster.assign_coords({STAC_PROPERTIES: (labels.dims[0], texts)})


def read_properties_coordinate(
    raster: xr.Dataset | xr.DataArray,
) -> list[dict[str, Any]]:
    """Read each date's Item properties from the `stac_properties` coordinate.

    Args:
        raster: Raster, with or without the coordinate.

    Returns:
        One mapping per `time` label, in label order; empty without the
        coordinate.
    """
    if STAC_PROPERTIES not in raster.coords:
        return []
    texts = np.atleast_1d(raster.coords[STAC_PROPERTIES].values)
    return [json.loads(str(text)) for text in texts]


def check_single_date(raster: xr.Dataset) -> None:
    """Refuse a raster that cannot be saved as one STAC observation.

    Args:
        raster: Raster about to be written as an Item asset.

    Raises:
        ValueError: The raster lacks a locatable grid or holds several dates.
            Only coordinates are inspected.
    """
    if raster.gs.geobox is None:
        raise ValueError(
            "an Item asset needs a locatable grid; assign a CRS before writing"
        )
    if TIME_COORDINATE in raster.coords and raster[TIME_COORDINATE].size > 1:
        raise ValueError(
            "one Item is one observation; split the dates with {time} in the "
            "path template"
        )


def create_item(
    assets: Mapping[str, tuple[UPath, xr.Dataset]],
    *,
    id_format: str,
    collection: str | None = None,
    datetime: ItemTime | None = None,
) -> pystac.Item:
    """Build the Item of one date's written files, with the STAC metadata they carry.

    Args:
        assets: Asset keys mapped to written paths and the selections written
            there, all of one date.
        id_format: Item id template, filled from `lat`, `lon`, `start`, `end`,
            and, for assets on one grid, `res`, `width`, `height`, `epsg`.
        collection: Collection id the Item belongs to.
        datetime: Date or interval for selections stating no time.

    Returns:
        Item whose assets are the files, and whose properties are the
        selections' `StacMetadata` common properties overlaid with their
        date's `stac_properties` mapping.

    Raises:
        ValueError: A selection lacks a locatable grid or holds several
            dates, the selections disagree on their date, or an id format
            field is unavailable. Grid fields require all assets on one grid.

    Warns:
        TimelessItemWarning: No selection states a time and `datetime` is None.

    Examples:
        >>> item = create_item({"red": (path, scene)}, id_format="{start:%Y%m%d}")
        >>> item.properties["platform"]
        'sentinel-2a'
    """
    rasters = [raster for _, raster in assets.values()]
    grids = []
    for raster in rasters:
        check_single_date(raster)
        grid = raster.gs.geobox
        if not isinstance(grid, GeoBox):
            raise ValueError("an Item asset needs a locatable grid with a transform")
        grids.append(grid)
    nominal, start, end = _read_time(rasters, datetime)
    footprint = union_all([grid.geographic_extent.geom for grid in grids])
    anchor = GeoAnchor.from_geometry(footprint, timespan=(start, end))
    geobox = None
    if set(fields_named(id_format)) & {"res", "width", "height", "epsg"}:
        geobox = grids[0]
        if any(other != geobox for other in grids[1:]):
            raise ValueError("grid fields of an Item id need its assets on one grid")
    ranged = start != end
    item = pystac.Item(
        anchor.format(id_format, geobox=geobox),
        mapping(footprint),
        list(footprint.bounds),
        nominal,
        _read_carried_properties(rasters[0]),
        start_datetime=start if ranged else None,
        end_datetime=end if ranged else None,
        collection=collection,
    )
    stated = attrs.StacMetadata.from_attrs(rasters[0].attrs)
    if stated is not None and stated.extensions:
        item.stac_extensions = list(
            dict.fromkeys([*item.stac_extensions, *stated.extensions])
        )
    for key, (path, raster) in assets.items():
        item.add_asset(key, asset.build_asset(path, raster, owner=item))
    return item


def _read_carried_properties(raster: xr.Dataset) -> dict[str, Any]:
    """Overlay the date's `stac_properties` mapping on the common properties.

    Fields the raster states in its own structure are left to it, even where
    `with_properties` recorded them, so a copied date never contradicts the
    Item's own.
    """
    stated = attrs.StacMetadata.from_attrs(raster.attrs)
    common = {} if stated is None else dict(stated.common_properties or {})
    dates = read_properties_coordinate(raster)
    # A selection about to be one Item holds one date, as check_single_date keeps it.
    carried = {**common, **dates[0]} if dates else common
    return select_item_properties(carried)


def _read_label(raster: xr.Dataset) -> np.datetime64 | None:
    """Return the one `time` label a selection holds, or None for a timeless one."""
    if TIME_COORDINATE not in raster.coords:
        return None
    return np.atleast_1d(raster[TIME_COORDINATE].values)[0]


def _read_time(
    rasters: Sequence[xr.Dataset], datetime: ItemTime | None
) -> tuple[DateTime | None, DateTime, DateTime]:
    """Return the nominal instant, start and end of one observation's selections."""
    labels = []
    for raster in rasters:
        label = _read_label(raster)
        if label is not None:
            labels.append(label)
    if not labels:
        if datetime is None:
            warnings.warn(
                "the rasters state no time; pass datetime= to date them",
                TimelessItemWarning,
                skip_file_prefixes=tuple(_package_paths),
            )
            return None, *TIMELESS
        if isinstance(datetime, tuple):
            start, end = datetime
            return (start if start == end else None), start, end
        return datetime, datetime, datetime

    dates = pd.DatetimeIndex(labels).unique()
    if len(dates) != 1 or pd.isna(dates[0]):
        raise ValueError("the assets of one Item must share one date")
    stamp = _to_utc(dates.to_numpy()[0])

    # A composite states the period it covers; an instant covers itself.
    starts = [stamp]
    ends = [stamp]
    for raster in rasters:
        bounds = read_bounds(raster.coords)
        if bounds is not None:
            first, last = bounds
            if np.isnat(first).any() or np.isnat(last).any():
                raise ValueError("start_datetime and end_datetime must not hold NaT")
            starts.append(_to_utc(first.min()))
            ends.append(_to_utc(last.max()))
    return stamp, min(starts), max(ends)


def _to_utc(value: np.datetime64) -> DateTime:
    """Read a naive UTC `datetime64`, already checked not NaT, as an aware datetime."""
    return cast("DateTime", pd.Timestamp(value).tz_localize(UTC).to_pydatetime())


def read_timespan(
    item: pystac.Item | Mapping[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the first and last instant an Item covers.

    Args:
        item: Item, or its dict as a window's `sources` holds it.

    Returns:
        (start, end), naive UTC: its `start_datetime` and `end_datetime`,
        else its `datetime` twice. None for a timeless Item, which spans
        `TIMELESS` or states no time.

    Examples:
        >>> read_timespan({"properties": {"datetime": "2025-06-01T00:00:00Z"}})
        (Timestamp('2025-06-01 00:00:00'), Timestamp('2025-06-01 00:00:00'))
    """
    properties = (
        item.properties if isinstance(item, pystac.Item) else item.get("properties", {})
    )
    start = properties.get("start_datetime") or properties.get("datetime")
    end = properties.get("end_datetime") or properties.get("datetime")
    if start is None or end is None:
        return None
    # Compared as Timestamps first, since TIMELESS lies past pandas' nanosecond range.
    if (pd.Timestamp(start), pd.Timestamp(end)) == TIMELESS:
        return None
    return (
        pd.to_datetime(start, utc=True).tz_convert(None),
        pd.to_datetime(end, utc=True).tz_convert(None),
    )


def read_instant(item: pystac.Item | Mapping[str, Any]) -> pd.Timestamp | None:
    """Return the instant an Item is dated by: its `datetime`, else its start.

    Args:
        item: Item, or its dict as a window's `sources` holds it.

    Returns:
        The instant, naive UTC, or None for a timeless Item.

    Examples:
        >>> read_instant({"properties": {"datetime": "2025-06-01T00:00:00Z"}})
        Timestamp('2025-06-01 00:00:00')
    """
    covered = read_timespan(item)
    if covered is None:
        return None
    properties = (
        item.properties if isinstance(item, pystac.Item) else item.get("properties", {})
    )
    nominal = properties.get("datetime")
    return pd.to_datetime(nominal, utc=True).tz_convert(None) if nominal else covered[0]


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
            continue
        variables = []
        for index in range(1, count + 1):
            # Prefer the published band name before its common name.
            taken = [alias for alias, band in owner.items() if band == (key, index)]
            variables.append(taken[0] if taken else f"{key}.{index}")
        found[key] = tuple(variables)
    return found


def stamp_item_time(raster: xr.Dataset, item: pystac.Item) -> xr.Dataset:
    """Date a raster opened from an Item, and record the Item's properties.

    Time coordinates already in the file are kept; composite bounds and the
    `stac_properties` coordinate come from the Item. A timeless Item adds no
    dates and no properties.
    """
    when = read_instant(item)
    if when is None:
        return raster

    if raster.gs.times is None:
        raster = raster.assign_coords(time=when)

    if "start_datetime" in item.properties and "end_datetime" in item.properties:
        edges = []
        for key in ("start_datetime", "end_datetime"):
            edges.append(
                pd.to_datetime(item.properties[key], utc=True)
                .tz_convert(None)
                .to_datetime64()
            )
        raster = assign_bounds(raster, [tuple(edges)])

    properties = select_item_properties(item.properties)
    labels = np.atleast_1d(raster[TIME_COORDINATE].values)
    return stamp_properties_coordinate(raster, [properties] * labels.size)


def build_stac_metadata(
    items: Sequence[pystac.Item], collection: pystac.Collection | None
) -> attrs.StacMetadata:
    """Read Collection metadata and the properties and schemas shared by Items.

    Args:
        items: Nonempty sequence of source Items.
        collection: Collection to retain without its catalog links, or None.

    Returns:
        Metadata for a Dataset root or a DataTree's collection node. Only
        nonstructural properties present and equal on every Item are retained.
    """
    fields = None
    if collection is not None:
        fields = collection.to_dict(include_self_link=False, transform_hrefs=False)
        fields["links"] = []
    shared = select_item_properties(items[0].properties)
    for entry in items[1:]:
        for key in list(shared):
            if key not in entry.properties or entry.properties[key] != shared[key]:
                del shared[key]
    extensions = []
    for entry in items:
        extensions.extend(entry.stac_extensions)
    return attrs.StacMetadata(
        collection=fields,
        common_properties=shared,
        extensions=list(dict.fromkeys(extensions)),
    )
