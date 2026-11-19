"""Compatibility rules among variables in one native Dataset."""

from __future__ import annotations

import xarray as xr

from .dataarray import same_encoding_as, same_pixel_meaning_as


def same_pixel_meaning(raster: xr.Dataset) -> None:
    """Require every variable to state the same CF packing, flags and units.

    Raises:
        RasterValidationError: Two variables state different pixel meaning.
    """
    names = list(raster.data_vars)
    if not names:
        return

    rule = same_pixel_meaning_as(raster[names[0]])
    for name in names[1:]:
        rule(raster[name])


def same_encoding(raster: xr.Dataset) -> None:
    """Require every variable to be stored with the same dtype and CF packing.

    Raises:
        RasterValidationError: Two variables are stored differently.
    """
    names = list(raster.data_vars)
    if not names:
        return

    rule = same_encoding_as(raster[names[0]])
    for name in names[1:]:
        rule(raster[name])
