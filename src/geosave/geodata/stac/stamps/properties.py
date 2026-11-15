"""Stamps that make the Item properties ODC loaded into marked coordinates.

They are not registered: ODC loads every requested property as a variable,
so `load_dataset` always runs them, before the registered stamps.
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import geosave.geodata.attrs as attrs
from geosave.geodata.core.conventions import (
    END_DATETIME,
    START_DATETIME,
    TIME_COORDINATE,
)

from .source import LoadSource


def fuse_start(values: list[str]) -> np.datetime64:
    """Return the earliest of one step's `start_datetime` values, naive UTC.

    ODC passes the values STAC states, without the Items that lack one; a
    value without a UTC offset is read as UTC.
    """
    if not values:
        return np.datetime64("NaT")
    instants = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="ISO8601"))
    return instants.tz_convert(None).to_numpy().min()


def fuse_end(values: list[str]) -> np.datetime64:
    """Return the latest of one step's `end_datetime` values, naive UTC."""
    if not values:
        return np.datetime64("NaT")
    instants = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="ISO8601"))
    return instants.tz_convert(None).to_numpy().max()


# The period every load requests: a step covers its Items' earliest start to
# latest end. ODC fills a step without Items with nodata.
PERIOD_REQUESTS: list[dict[str, Any]] = [
    {
        "key": START_DATETIME,
        "dtype": "datetime64[ns]",
        "nodata": "NaT",
        "fuser": fuse_start,
    },
    {
        "key": END_DATETIME,
        "dtype": "datetime64[ns]",
        "nodata": "NaT",
        "fuser": fuse_end,
    },
]


def stamp_properties(ds: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Move each property ODC loaded as a variable to a coordinate marked with its key.

    ODC names a property variable `eo_cloud_cover` for the key
    `eo:cloud_cover`, or as the request's `name` says; the `stac_property`
    mark keeps the key.

    Args:
        ds: Dataset returned by the ODC load.
        source: The load's property requests.

    Returns:
        `ds` with each requested property a `(time,)` coordinate.
    """
    for request in source.properties:
        name = request.output_name
        ds = ds.set_coords(name)
        marked = attrs.stamp_attrs(
            ds[name].variable, attrs.StacProperty(name=request.key)
        )
        ds = ds.assign_coords({name: marked})
    return ds


def stamp_period(ds: xr.Dataset) -> xr.Dataset:
    """Make the loaded period the time cells, filling steps whose Items state only `datetime`.

    The period is the time axis's cells, found by name as the conventions
    name them, so it carries no `stac_property` mark. pystac keeps `datetime`
    out of an Item's properties, so ODC cannot load it; such a step's cell
    is its label, the earliest Item's instant. That is exact for one Item;
    for several, the end falls short by the seconds between acquisitions.

    Args:
        ds: Dataset whose period coordinates `stamp_properties` wrote.

    Returns:
        `ds` with `start_datetime` and `end_datetime` free of NaT and
        attrs, which would clash with CF time encoding.
    """
    labels = ds[TIME_COORDINATE]
    filled = {}
    for name in (START_DATETIME, END_DATETIME):
        period = ds[name].where(ds[name].notnull(), labels)
        filled[name] = period.drop_attrs().variable
    return ds.assign_coords(filled)
