"""Helpers for APIs that accept one raster or an iterable of rasters."""

from collections.abc import Iterable
from typing import cast

import xarray as xr

Raster = xr.DataArray | xr.Dataset | xr.DataTree


def as_rasters[T: Raster](value: T | Iterable[T]) -> Iterable[T]:
    """Wrap a single raster as a tuple; pass an iterable of rasters through."""
    # All three xarray objects are iterable in their own right.
    if isinstance(value, Raster):
        return (cast(T, value),)
    return value
