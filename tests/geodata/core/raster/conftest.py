from __future__ import annotations

import numpy as np
import xarray as xr

from geosave.geodata.core.raster.create import raster
from tests.geodata.conftest import geobox


def optical(shape: tuple[int, int] = (8, 8), *, times: int = 2) -> xr.Dataset:
    """Two uint16 bands whose constant values expose their stacking order."""
    box = geobox(shape=shape)
    labels = np.array(
        [f"2024-0{month + 1}-01" for month in range(times)], dtype="datetime64[ns]"
    )
    return raster(
        {
            "B04": (
                ("time", *("y", "x")),
                np.full((times, *shape), 4, "uint16"),
            ),
            "B08": (
                ("time", *("y", "x")),
                np.full((times, *shape), 8, "uint16"),
            ),
        },
        box,
        time=labels,
    ).gs.write_nodata(0)
