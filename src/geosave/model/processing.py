"""Explicit raster/tensor boundaries and model normalization."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import torch


def to_tensor(
    raster: xr.Dataset | xr.DataArray, *, dims: Sequence[str], dtype: str = "float32"
) -> torch.Tensor:
    """Materialize a raster in an explicitly declared dimension order.

    Args:
        raster: Native raster. Dataset variables become bands in their current order.
        dims: Every output dimension, including band for a Dataset.
        dtype: PyTorch dtype name.

    Returns:
        CPU tensor in the declared order. Conversion reads lazy pixels.

    Raises:
        ValueError: Dataset variables have different dimension sets. Broadcast
            explicitly before encoding if repeating a variable is intended.
    """
    import torch

    if isinstance(raster, xr.Dataset):
        dimensions = {
            frozenset(variable.dims) for variable in raster.data_vars.values()
        }
        if len(dimensions) != 1:
            raise ValueError(
                "tensor bands must have the same dimensions; broadcast explicitly first"
            )
        array = raster.to_array(dim="band")
    else:
        array = raster
    values = array.transpose(*dims).to_numpy()
    # Views may have negative strides or be read-only, which torch cannot safely use.
    return torch.as_tensor(np.array(values, copy=True), dtype=getattr(torch, dtype))


def normalize(
    value: torch.Tensor, *, mean: Sequence[float], std: Sequence[float], dim: int
) -> torch.Tensor:
    """Normalize one tensor axis with fixed statistics, preserving gradients.

    Args:
        value: Floating tensor on any device, batched or unbatched.
        mean: One mean per element of the selected axis.
        std: One finite positive standard deviation per axis element.
        dim: Axis to normalize; negative axes remain stable when batching.

    Returns:
        Tensor with the same shape, dtype, and device. Missing values are preserved.

    Raises:
        ValueError: Statistics or axis lengths are invalid.
        TypeError: The tensor is not floating point.
    """
    import torch

    if not value.is_floating_point():
        raise TypeError("normalize requires a floating tensor")
    if not -value.ndim <= dim < value.ndim:
        raise ValueError(
            f"normalization axis {dim} is absent from shape {tuple(value.shape)}"
        )
    if len(mean) != value.shape[dim] or len(std) != value.shape[dim]:
        raise ValueError("normalization statistics must match the selected axis")
    center = torch.as_tensor(mean, dtype=value.dtype, device=value.device)
    scale = torch.as_tensor(std, dtype=value.dtype, device=value.device)
    if not bool(
        torch.isfinite(center).all() & torch.isfinite(scale).all() & (scale > 0).all()
    ):
        raise ValueError(
            "normalization needs finite means and positive finite standard deviations"
        )
    shape = [1] * value.ndim
    shape[dim] = len(mean)
    return (value - center.reshape(shape)) / scale.reshape(shape)


def to_raster(
    value: torch.Tensor | np.ndarray,
    *,
    reference: xr.Dataset | xr.DataArray,
    dims: Sequence[str],
    name: str,
) -> xr.DataArray:
    """Place model values on an unchanged reference grid.

    Args:
        value: Output values. Tensors are detached and copied to CPU.
        reference: Raster defining exact spatial placement, not output band meaning.
        dims: Output axes; spatial axes must be y and x.
        name: Output variable name.

    Returns:
        Native DataArray on the reference grid, without inherited input packing.
        Assign output units, class labels, and nodata explicitly in the recipe.

    Raises:
        ValueError: The output does not cover the reference grid exactly.
    """
    import torch

    values = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    grid = reference.gs.geobox
    if grid is None or "y" not in dims or "x" not in dims:
        raise ValueError("output requires a reference grid and y/x dimensions")
    if len(dims) != values.ndim:
        raise ValueError("output dimensions must match its rank")
    if (values.shape[dims.index("y")], values.shape[dims.index("x")]) != tuple(
        grid.shape
    ):
        raise ValueError(
            "output shape differs from the reference grid; supply its actual grid explicitly"
        )
    return xr.DataArray(values, dims=dims, name=name).gs.write_grid(grid)
