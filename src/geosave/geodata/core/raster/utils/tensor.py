"""Torch tensor conversion shared by the `gs` raster accessors."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def as_tensor(
    to_numpy: Callable[..., np.ndarray], dtype: str | torch.dtype | None
) -> torch.Tensor:
    """Read an accessor's model-input array as a writable, contiguous tensor.

    Args:
        to_numpy: The accessor's own `to_numpy`, which fixes the axis order.
        dtype: Torch dtype or its string name. None keeps the pixel dtype. A
            dtype NumPy cannot hold, such as bfloat16, reads as float32 first.

    Returns:
        Tensor laid out as `to_numpy` lays out its array.

    Raises:
        ValueError: A dtype name is unknown.
        TypeError: `dtype` is neither a torch dtype nor a name, or the pixel
            dtype has no tensor equivalent.
    """
    # Imported here so that importing geodata never pays for torch.
    import torch

    if isinstance(dtype, str):
        target = getattr(torch, dtype, None)
        if not isinstance(target, torch.dtype):
            raise ValueError(f"Unknown torch dtype {dtype!r}")
    elif dtype is None or isinstance(dtype, torch.dtype):
        target = dtype
    else:
        raise TypeError(f"{dtype!r} is not a torch dtype or the name of one")

    # Read-only pixel buffers are copied, so tensor operations can safely write.
    if target is None:
        values = np.require(to_numpy(), requirements=["C", "W"])
        try:
            return torch.as_tensor(values)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Cannot convert numpy dtype {values.dtype} to a torch tensor"
            ) from error

    try:
        reading = torch.empty(0, dtype=target).numpy().dtype
    except TypeError:
        reading = np.dtype("float32")
    return torch.as_tensor(
        np.require(to_numpy(dtype=reading), requirements=["C", "W"]), dtype=target
    )
