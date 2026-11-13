"""Stamp that writes Item properties as coordinates marked `stac_property`."""

from typing import Any

import numpy as np
import pystac
import xarray as xr

from geosave.geodata.stac.item import select_item_properties

from .source import LoadSource


def stamp_properties(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Write each scalar Item property as a coordinate marked `stac_property`.

    A property every Item states alike is a scalar coordinate. When every
    `time` step is one Item, a property they differ on follows `time`,
    missing where an Item lacks it; otherwise it is left out. List- and
    dict-valued properties are left out.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items that were loaded.

    Returns:
        `loaded` with one coordinate per property.
    """
    stepped = source.match_steps(loaded)
    items = source.items if stepped is None else stepped

    readings = []
    keys = {}
    for item in items:
        reading = _read_scalars(item)
        readings.append(reading)
        for key in reading:
            keys[key] = None

    for key in keys:
        values = []
        for reading in readings:
            values.append(reading.get(key))

        if _agree(key, readings):
            loaded = loaded.gs.write_stac_property(key, values[0])
        elif stepped is not None:
            loaded = loaded.gs.write_stac_property(key, _build_values(values))
    return loaded


def _read_scalars(item: pystac.Item) -> dict[str, Any]:
    """Read the acquisition properties of an Item that a coordinate can hold."""
    scalars = {}
    for key, value in select_item_properties(item.properties).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            scalars[key] = value
    return scalars


def _agree(key: str, readings: list[dict[str, Any]]) -> bool:
    """Check that every reading states `key` with one value of one type.

    The type check keeps `1`, `1.0` and `True`, which compare equal, apart.
    """
    first = readings[0].get(key)
    for reading in readings:
        if key not in reading:
            return False
        value = reading[key]
        if type(value) is not type(first) or value != first:
            return False
    return True


def _build_values(values: list[Any]) -> np.ndarray:
    """Build one value per step, None where an Item lacks the property.

    Numbers keep a numeric dtype, with NaN for a missing value; anything else
    is an object array, so strings are neither truncated nor coerced.
    """
    numeric = True
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            numeric = False

    if numeric and None not in values:
        return np.array(values)
    if numeric:
        filled = []
        for value in values:
            filled.append(np.nan if value is None else value)
        return np.array(filled, dtype=float)

    stored = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        stored[index] = value
    return stored
