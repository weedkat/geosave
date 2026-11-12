"""Mask missing pixels or apply boolean masks to one raster array."""

from __future__ import annotations

import numpy as np
import xarray as xr


def mask_with_np(
    raster: xr.DataArray, mask: np.ndarray, *, fill: float | int | None = None
) -> xr.DataArray:
    """Apply a positional mask whose shape follows the raster's spatial axes."""
    dims = raster.gs.spatial_dims
    if dims is None:
        raise ValueError("a NumPy mask requires known spatial dimensions")
    shape = tuple(raster.sizes[dim] for dim in dims)
    if mask.shape != shape:
        raise ValueError(
            f"mask is shaped {mask.shape} but the spatial shape is {shape}; "
            "pass a DataArray to mask along named axes"
        )
    return mask_with_da(raster, xr.DataArray(mask, dims=dims), fill=fill)


def mask_with_da(
    raster: xr.DataArray, mask: xr.DataArray, *, fill: float | int | None = None
) -> xr.DataArray:
    """Keep True pixels using native named alignment and the packing fill value."""
    if mask.dtype != bool:
        raise ValueError(f"mask is {mask.dtype}, not boolean; pass a comparison")
    if (
        raster.gs.crs is not None
        and mask.gs.crs is not None
        and raster.gs.crs != mask.gs.crs
    ):
        raise ValueError("mask and raster have different CRSs; align the mask first")

    # Equivalent CRS carriers can have different scalar placeholder values.
    # Keep the raster's carrier rather than merging it with the condition's.
    if mask.odc.crs_coord is not None:
        mask = mask.drop_vars(mask.odc.crs_coord.name)

    if fill is None:
        packing = raster.gs.packing
        fill = packing.fill_value if packing is not None else None
        if fill is None:
            raise ValueError(f"{str(raster.name)!r} carries no fill value in packing")
    else:
        raster = raster.gs.write_nodata(fill)

    return raster.where(mask, fill)


def mask_missing(array: xr.DataArray) -> xr.DataArray:
    """Decode CF missing values for display without changing stored-value units."""
    from xarray.coding.variables import CFMaskCoder

    variable = array.variable.copy(deep=False)
    variable.encoding.setdefault("dtype", array.dtype)
    masked = CFMaskCoder().decode(variable, name=array.name)
    result = xr.DataArray(masked, coords=array.coords, name=array.name)
    result.encoding = dict(masked.encoding)
    return result
