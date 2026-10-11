"""Read ordered timestamps from the raster an encoder receives."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


def time_labels(raster: xr.Dataset | xr.DataArray) -> list[datetime]:
    """Return this raster's ordered acquisition timestamps.

    Args:
        raster: Raster whose frames the encoder receives; one date may be a
            scalar `time` coordinate.

    Returns:
        Datetimes in the same order as the model's temporal input.

    Raises:
        ValueError: The raster has no time labels, or one is missing.
    """
    if "time" not in raster.coords:
        raise ValueError("model context requires the raster's time labels")
    values = np.atleast_1d(raster["time"].values).astype("datetime64[us]")
    if not values.size or np.isnat(values).any():
        raise ValueError("model context requires non-empty, finite time labels")
    return [pd.Timestamp(value).to_pydatetime() for value in values]
