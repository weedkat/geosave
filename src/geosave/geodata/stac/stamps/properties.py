"""Stamp that writes the Item properties of a load as coordinates."""

from typing import Any

import pystac
import xarray as xr

from .registry import register_stamp
from .source import LoadSource


@register_stamp("properties")
def stamp_properties(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Write each scalar property every Item states alike, marked `stac_property`.

    A property the Items differ on is left out; request it with
    `with_properties` to keep ODC's per-step value. List- and dict-valued
    properties are left out.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items that were loaded.

    Returns:
        `loaded` with one scalar coordinate per shared property.
    """
    readings = []
    for item in source.items:
        readings.append(_read_scalars(item))

    for key, value in readings[0].items():
        if _agree(key, value, readings):
            loaded = loaded.gs.write_stac_property(key, value)
    return loaded


# Properties the loaded raster states in its own structure: `time` and its
# period state the dates, the grid states `proj:*`, each variable's attrs its
# band; `created` and `updated` describe the source Item's record, not the data.
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


def _read_scalars(item: pystac.Item) -> dict[str, Any]:
    """Read the acquisition properties of an Item that a scalar coordinate can hold."""
    scalars = {}
    for key, value in item.properties.items():
        if key in _STATED_BY_RASTER or key.startswith("proj:"):
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            scalars[key] = value
    return scalars


def _agree(key: str, value: Any, readings: list[dict[str, Any]]) -> bool:
    """Check that every reading states `key` as `value`, of the same type.

    The type check keeps `1`, `1.0` and `True`, which compare equal, apart.
    """
    for reading in readings:
        if key not in reading:
            return False
        if type(reading[key]) is not type(value) or reading[key] != value:
            return False
    return True
