"""Write the observation period of each time plane loaded by ODC."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pystac
import xarray as xr
from odc.stac import ConversionConfig

from geosave.geodata.core.conventions import END_DATETIME, START_DATETIME


def fuse_start(values: list[str]) -> np.datetime64:
    """Read the earliest stated start in one ODC group as naive UTC."""
    if not values:
        return np.datetime64("NaT")

    dates = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="ISO8601"))
    return dates.tz_convert(None).to_numpy().min()


def fuse_end(values: list[str]) -> np.datetime64:
    """Read the latest stated end in one ODC group as naive UTC."""
    if not values:
        return np.datetime64("NaT")

    dates = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="ISO8601"))
    return dates.tz_convert(None).to_numpy().max()


# Pass these to ODC when loading Items with interval coverage.
PERIOD_REQUESTS = [
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


def stamp_observation_period(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
) -> xr.Dataset:
    """Write the start and end of each observation as time coordinates.

    ODC's `with_properties=PERIOD_REQUESTS` provides the grouped endpoints
    for interval Items. For acquisitions stating only an instant, the time
    plane's timestamp supplies both endpoints.
    """
    coordinates = {}

    for key in (START_DATETIME, END_DATETIME):
        if key in ds:
            period = ds[key].fillna(ds.time)
        elif any(item.properties.get(key) is not None for item in items):
            raise ValueError(
                f"Load {key!r} with with_properties=PERIOD_REQUESTS to stamp interval Items"
            )
        else:
            period = ds.time
        coordinates[key] = period.drop_attrs().variable

    return ds.assign_coords(coordinates)
