from __future__ import annotations

import dask.array as da
import xarray as xr


def _empty(name: str) -> xr.DataArray:
    """Return a pixel-free 0-d variable, as a catalog read starts one."""
    return xr.DataArray(da.empty((), "float64"), name=name)
