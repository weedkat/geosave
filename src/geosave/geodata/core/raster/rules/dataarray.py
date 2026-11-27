"""Value-interpretation rules for one native raster array."""

from __future__ import annotations

import xarray as xr

from geosave.geodata import attrs
from geosave.geodata.errors import RasterValidationError
from . import Rule


# Methods that pick an existing value rather than compute a new one.
_VALUE_PRESERVING = frozenset({"nearest", "mode", "first", "last"})


def decoded(band: xr.DataArray) -> None:
    """Require no active CF packing, so a reduction sees physical values.

    Raises:
        RasterValidationError: The band still states CF packing.
    """
    if attrs.CFPacking.is_stated_in(band.attrs):
        raise RasterValidationError(
            "The band still holds packed values. Decode them with "
            "mask_and_scale() first.",
            variable=band.name,
        )


def compatible_resampling(method: str) -> Rule[xr.DataArray]:
    """Build the rule that the band's values allow resampling with `method`.

    Class IDs allow only methods that pick an existing value: nearest or
    mode in space, first or last in time. Other values allow any method.

    Args:
        method: Resampling kernel or temporal reducer, such as ``"bilinear"``
            or ``"mean"``.

    Returns:
        Rule for `validate`.

    Examples:
        >>> landcover.gs.validate(compatible_resampling("bilinear"))
        RasterValidationError: Raster 'landcover': Its class IDs need ...
    """

    def check(band: xr.DataArray) -> None:
        if band.gs.is_categorical and method not in _VALUE_PRESERVING:
            raise RasterValidationError(
                f"Its class IDs need nearest, mode, first or last, not "
                f"{method!r}, which would compute values that name no class.",
                variable=band.name,
            )

    return check


def same_pixel_meaning_as(first: xr.DataArray) -> Rule[xr.DataArray]:
    """Build the rule that a band states the CF packing, flags and units `first` states.

    Args:
        first: Band whose attrs give the pixel meaning to match.

    Returns:
        Rule for `validate`.

    Examples:
        >>> ds["B08"].gs.validate(same_pixel_meaning_as(ds["B04"]))
    """

    def check(band: xr.DataArray) -> None:
        if (
            band.gs.packing != first.gs.packing
            or band.gs.flags != first.gs.flags
            or band.attrs.get("units") != first.attrs.get("units")
        ):
            raise RasterValidationError(
                f"It states different CF packing, flags or units than "
                f"{first.name!r}, so their pixels would mean different things "
                "together. Decode them with mask_and_scale() first.",
                variable=band.name,
            )

    return check


def same_encoding_as(first: xr.DataArray) -> Rule[xr.DataArray]:
    """Build the rule that a band is stored with the dtype and CF packing `first` is.

    Decoding moves the packing from attrs to encoding, where writers read it
    to pack the pixels again.

    Args:
        first: Band whose encoding gives the storage to match.

    Returns:
        Rule for `validate`.

    Examples:
        >>> ds["B08"].gs.validate(same_encoding_as(ds["B04"]))
    """

    first_packing = attrs.CFPacking.from_attrs(first.encoding)

    def check(band: xr.DataArray) -> None:
        packing = attrs.CFPacking.from_attrs(band.encoding)
        dtype = band.encoding.get("dtype")
        if dtype != first.encoding.get("dtype") or packing != first_packing:
            raise RasterValidationError(
                f"Its encoding stores another dtype or CF packing than {first.name!r}, "
                "so one written file cannot hold both. Prepare or split them first.",
                variable=band.name,
            )

    return check
