"""A STAC Item as the `Scene` it lists: one collection at one instant.

This module dates an Item from its scene's files, and reads that time back,
for every caller.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime as DateTime
from typing import Any, cast

import numpy as np
import pandas as pd
import pystac
from odc.geo.geobox import GeoBox
from shapely import union_all
from shapely.geometry.base import BaseGeometry
from shapely.geometry import mapping

from geosave import __path__ as _package_paths
from geosave.geodata.io.raster.file import AssetKey, Scene
from geosave.geodata.stac.extensions import eo
from geosave.geodata.utils.template import fields_named
from geosave.geodata.utils.scalar import coordinate_value
import xarray as xr
from geosave.geodata import attrs
from geosave.geodata.stac.extensions import projection
from geosave.geodata.warnings import MissingGridWarning, TimelessItemWarning

from . import asset, band

# An Item's time: one instant, or a start and an end.
type ItemTime = DateTime | tuple[DateTime, DateTime]

# The time of an Item whose files state none. STAC dates every Item, so a
# timeless one, as an elevation model, spans all of time and so overlaps every
# window; `read_timespan` and `read_instant` read it back as None.
TIMELESS = (
    DateTime(1, 1, 1, tzinfo=UTC),
    DateTime(9999, 12, 31, 23, 59, 59, tzinfo=UTC),
)


def from_scene(
    scene: Scene,
    *,
    id_format: str,
    collection: str | None = None,
    datetime: ItemTime | None = None,
) -> pystac.Item:
    """Build the Item listing a scene's files.

    Args:
        scene: Written scene.
        id_format: `GeoAnchor.format` template the Item's id is filled from,
            its fields taken from the Item's own grid and time.
        collection: Collection id; None takes the scene's.
        datetime: Time for a scene whose files state none, as one instant or
            a start and end.

    Returns:
        Item with one data Asset per file over their joined footprint, dated
        by its files' one shared time label as `datetime`, or the span their
        labels cover as `start_datetime` and `end_datetime`.

    Raises:
        RuntimeError: The scene is still pending.
        ValueError: A file has no locatable grid.

    Warns:
        TimelessItemWarning: No file is dated and `datetime` is None, so the
            Item spans `TIMELESS`.
    """
    grids = [f.header.gs.geobox for f in scene.written.values()]
    if not grids or any(grid is None for grid in grids):
        raise ValueError("a scene needs files with locatable grids")

    nominal, start, end = _read_scene_time(scene, datetime)
    footprint = union_all([grid.geographic_extent.geom for grid in grids])
    ranged = start != end
    item = pystac.Item(
        _format_id(id_format, grids, footprint, start, end),
        mapping(footprint),
        list(footprint.bounds),
        nominal,
        {},
        start_datetime=start if ranged else None,
        end_datetime=end if ranged else None,
        collection=collection or scene.collection,
    )

    for key, file in scene.written.items():
        item.add_asset(key, asset.from_file(file, owner=item))

    coords = (
        scene.coords
        if scene.coords is not None
        else next(iter(scene.written.values())).header.coords
    )
    _write_properties(item, coords)
    return item


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


def select_data_assets(item: pystac.Item) -> dict[AssetKey, pystac.Asset]:
    """Select data assets in published order, including assets without stated roles.

    Explicit roles without `data` exclude an asset from native raster reading,
    grid selection, and variable naming. The Item and its assets are not changed.
    """
    return {
        key: entry
        for key, entry in item.assets.items()
        if entry.roles is None or "data" in entry.roles
    }


def read_grid(item: pystac.Item) -> GeoBox | None:
    """Return the grid of the first eligible data asset that states one.

    Returns:
        Stated grid, or None when no data asset supplies a grid and CRS.
        This metadata lookup does not open the raster files.

    Warns:
        MissingGridWarning: No stated grid; native image reading remains available.

    Examples:
        >>> read_grid(item).shape
        Shape2d(x=64, y=64)
    """
    for entry in select_data_assets(item).values():
        found = projection.to_geobox(entry)
        if found is not None and found.crs is not None:
            return found

    warnings.warn(
        f"{item.id} states no georeferenced grid on any data asset; it cannot "
        "define a spatial sampling window. Read its images with row.gs.to_xarray().",
        MissingGridWarning,
        skip_file_prefixes=tuple(_package_paths),
    )
    return None


def read_variable_names(item: pystac.Item) -> dict[AssetKey, tuple[str, ...]]:
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
    claims: dict[str, list[tuple[int, AssetKey, int]]] = {}
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


def assign_time(raster: xr.Dataset, item: pystac.Item) -> xr.Dataset:
    """Apply Item dates without replacing time coordinates already in the file.

    Composite bounds are taken from the Item; timeless Items add no dates.
    """
    when = read_instant(item)
    if when is None:
        return raster

    if raster.gs.times is None:
        raster = raster.assign_coords(time=when)

    if "start_datetime" in item.properties and "end_datetime" in item.properties:
        coords = {}
        for key in ("start_datetime", "end_datetime"):
            coords[key] = (
                pd.to_datetime(item.properties[key], utc=True)
                .tz_convert(None)
                .to_datetime64()
            )
        raster = raster.assign_coords(coords)

    return raster


def _format_id(
    template: str,
    grids: list[GeoBox],
    footprint: BaseGeometry,
    start: DateTime,
    end: DateTime,
) -> str:
    """Name the union footprint; grid fields require one shared grid."""
    fields = {
        "lat": footprint.centroid.y,
        "lon": footprint.centroid.x,
        "start": start,
        "end": end,
    }
    if set(fields_named(template)) & {"res", "width", "height", "epsg"}:
        grid = grids[0]
        if any(other != grid for other in grids[1:]):
            raise ValueError("grid-specific Item id fields require a common grid")
        if grid.crs is None:
            raise ValueError("grid carries no CRS")
        fields.update(
            res=abs(grid.resolution.x),
            width=grid.boundingbox.span_x,
            height=grid.boundingbox.span_y,
            epsg=grid.crs.epsg,
        )

    return template.format(**fields)


def _write_properties(item: pystac.Item, coords: xr.Coordinates) -> None:
    """Translate scalar observation properties and their extensions."""
    eo.write_item(item, coords)
    for name in ("platform", "constellation", "gsd"):
        value = coordinate_value(coords, name)
        if value is not None:
            if name == "gsd":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value <= 0
                ):
                    raise ValueError("gsd must be a positive finite number")
            elif not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            item.properties[name] = value


def _read_scene_time(
    scene: Scene, datetime: ItemTime | None
) -> tuple[DateTime | None, DateTime, DateTime]:
    """Prefer observation coordinates, then caller dates, then timeless coverage."""
    coords = [file.header.coords for file in scene.written.values()]
    if scene.coords is not None and "time" in scene.coords:
        coords = [scene.coords]
    dated = [entry for entry in coords if "time" in entry]
    if dated:
        return _read_observation_time(dated)

    if datetime is not None:
        if isinstance(datetime, tuple):
            start, end = datetime
            return (start if start == end else None), start, end
        return datetime, datetime, datetime

    warnings.warn(
        "scene states no time; pass datetime= to date it",
        TimelessItemWarning,
        skip_file_prefixes=tuple(_package_paths),
    )
    return None, *TIMELESS


def _read_observation_time(
    coords: list[xr.Coordinates],
) -> tuple[DateTime, DateTime, DateTime]:
    """Keep one nominal label separate from explicit or resampled coverage."""
    labels = []
    for entry in coords:
        labels.extend(np.atleast_1d(entry["time"].values))
    times = pd.DatetimeIndex(labels).unique()
    if len(times) != 1:
        raise ValueError(
            "one Item represents one observation or composite; split multiple dates with a {time} path template"
        )
    label = cast(pd.Timestamp, times[0])
    if label is pd.NaT:
        raise ValueError("an observation time must not be NaT")
    stamp = pd.to_datetime(label, utc=True).to_pydatetime()

    spans = [_read_coverage(entry, stamp) for entry in coords]
    return stamp, min(start for start, _ in spans), max(end for _, end in spans)


def _read_coverage(
    coords: xr.Coordinates, stamp: DateTime
) -> tuple[DateTime, DateTime]:
    """Read explicit composite bounds, calendar coverage, or a point observation."""
    if "start_datetime" in coords or "end_datetime" in coords:
        start = coordinate_value(coords, "start_datetime")
        end = coordinate_value(coords, "end_datetime")
        if start is None or end is None:
            raise ValueError(
                "composite coordinates need both start_datetime and end_datetime"
            )
        first = pd.to_datetime(start, utc=True).to_pydatetime()
        last = pd.to_datetime(end, utc=True).to_pydatetime()
        if first > last:
            raise ValueError("start_datetime must not follow end_datetime")
        return first, last

    spec = attrs.TimeSpec.from_attrs(coords["time"].attrs)
    if spec is None:
        return stamp, stamp

    start, end = spec.timespan(coords["time"].values)
    return start.replace(tzinfo=UTC), end.replace(tzinfo=UTC)
