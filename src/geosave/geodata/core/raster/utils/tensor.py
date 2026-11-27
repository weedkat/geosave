"""Conversions from raster pixels to PyTorch tensors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def resolve_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Return the torch dtype given, or the one named.

    Raises:
        ValueError: A dtype name is unknown.
        TypeError: `dtype` is neither a torch dtype nor a name.
    """
    # Imported here so that importing geodata never pays for torch.
    import torch

    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError(f"{dtype!r} is not a torch dtype or the name of one")

    target = getattr(torch, dtype, None)
    if not isinstance(target, torch.dtype):
        raise ValueError(f"Unknown torch dtype {dtype!r}")
    return target


def find_numpy_dtype(dtype: torch.dtype | None) -> np.dtype | None:
    """Return the NumPy dtype that reads pixels for a tensor of `dtype`.

    A dtype NumPy cannot hold, such as bfloat16, reads as float32. None keeps
    the pixel dtype.
    """
    import torch

    if dtype is None:
        return None
    try:
        return torch.empty(0, dtype=dtype).numpy().dtype
    except TypeError:
        return np.dtype("float32")


def as_tensor(array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert an array to a writable, contiguous tensor.

    Read-only buffers are copied, so tensor operations can safely write.

    Args:
        array: Array to convert.
        dtype: Tensor dtype. None keeps the array dtype.

    Raises:
        TypeError: The array dtype has no tensor equivalent.
    """
    import torch

    writable = np.require(array, requirements=["C", "W"])
    try:
        return torch.as_tensor(writable, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Cannot convert numpy dtype {array.dtype} to a torch tensor"
        ) from error


def as_tensors(arrays: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert named arrays to tensors as `as_tensor` does, keeping their names.

    Raises:
        TypeError: An array dtype has no tensor equivalent.
    """
    tensors = {}
    for name, array in arrays.items():
        tensors[name] = as_tensor(array)
    return tensors
