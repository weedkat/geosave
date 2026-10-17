"""Scalar observation values from xarray coordinates."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import xarray as xr


def coordinate_value(coords: xr.Coordinates, name: str) -> Any:
    """Read a scalar or one time-indexed value, omitting missing values."""
    if name not in coords:
        return None
    coord = coords[name]
    if coord.size != 1 or any(dim != "time" for dim in coord.dims):
        raise ValueError(f"{name} must be scalar or have one time value")
    value = coord.values.reshape(-1)[0]
    if pd.isna(value):
        return None
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
