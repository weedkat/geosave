"""Compatibility rules among variables in one native Dataset."""

from __future__ import annotations

import xarray as xr

from geosave.geodata.core.conventions import BAND_DIMENSION
from geosave.geodata.errors import RasterValidationError
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


def stackable(raster: xr.Dataset) -> None:
    """Require variables that stack along a new band axis.

    Stacking needs at least one variable, no existing band dimension, and
    variables of one dtype over the same dimensions.

    Raises:
        RasterValidationError: A requirement is broken.
    """
    names = list(raster.data_vars)
    if not names:
        raise RasterValidationError("Stacking needs at least one raster variable")

    if BAND_DIMENSION in raster.dims:
        raise RasterValidationError(
            "Rename the existing band dimension before stacking variables"
        )

    first = raster[names[0]]
    for name in names[1:]:
        variable = raster[name]
        if variable.dims != first.dims:
            raise RasterValidationError(
                f"Its dimensions {variable.dims} differ from {first.dims} of "
                f"{first.name!r}; broadcast explicitly before stacking.",
                variable=name,
            )
        if variable.dtype != first.dtype:
            raise RasterValidationError(
                f"Its dtype {variable.dtype} differs from {first.dtype} of "
                f"{first.name!r}; provide dtype to cast them onto one.",
                variable=name,
            )
