"""This project's band selection and reflectance preparation."""

import torch
import xarray as xr


def prepare(raster: xr.Dataset) -> xr.Dataset:
    """Select Sentinel-2 bands in model order and decode stored reflectance."""
    selected = raster[["B02", "B03", "B04", "B08"]]
    for variable in selected.data_vars.values():
        if variable.dims != ("time", "y", "x"):
            raise ValueError(
                "Select one observation with the recipe's expected dimensions"
            )
    return selected.gs.mask_and_scale()


def accept_tensor(value: torch.Tensor) -> torch.Tensor:
    """Use an already encoded tensor without copying or detaching gradients."""
    return value
