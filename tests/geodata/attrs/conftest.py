from __future__ import annotations

import numpy as np
import xarray as xr


def _band(dtype="uint8", **attributes) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((2, 2), dtype),
        dims=("y", "x"),
        coords={"x": xr.DataArray([0, 1], dims="x", attrs={"axis": "X"})},
        attrs={"units": "1", **attributes},
    )
