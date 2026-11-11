"""Convert between stored and decoded values using xarray's CF codecs."""

from __future__ import annotations

import numpy as np
import xarray as xr
from xarray.coding.variables import CFMaskCoder
from xarray.conventions import encode_cf_variable

from geosave.geodata import attrs


def mask_and_scale(variable: xr.DataArray) -> xr.DataArray:
    """Decode one variable's CF attrs, retaining native storage encoding.

    Missing pixels are masked before scaling. Without CF packing attrs the
    variable passes through. Names, coordinates and lazy arrays are preserved.

    Args:
        variable: Stored values with per-variable CF packing attrs.

    Returns:
        Decoded values with consumed attrs retained in storage encoding.

    Raises:
        ValueError: Missing codes do not fit the dtype, declarations disagree,
            or nonidentity scaling is attached to categorical codes.
    """
    packing = attrs.Packing.from_attrs(variable.attrs)
    if packing is None:
        return variable
    packing.check_dtype(variable.dtype)
    if variable.gs.is_categorical and (
        packing.scale_factor != 1 or packing.add_offset != 0
    ):
        raise ValueError(
            f"{str(variable.name)!r} holds class codes and declares packing; "
            "remove its scale/offset or its legend before decoding"
        )

    canonical = _canonicalize_missing(variable)
    # Xarray infers the decoded dtype from scale/offset metadata. Integer
    # scales can cast floating pixels to integers before scaling, losing NaNs.
    # Keep existing floating precision; normalize integer and GDAL text attrs.
    for key, value in (
        ("scale_factor", packing.scale_factor),
        ("add_offset", packing.add_offset),
    ):
        if key in canonical.attrs and not isinstance(
            canonical.attrs[key], (float, np.floating)
        ):
            canonical.attrs[key] = value

    # Only the data variable is decoded; coordinates may already be CF-decoded.
    name = "value"
    while name in variable.dims:
        name = "_" + name
    dataset = xr.Dataset({name: canonical.variable})
    decoded = xr.decode_cf(
        dataset,
        concat_characters=False,
        decode_times=False,
        decode_timedelta=False,
        decode_coords=False,
    )[name]
    result = xr.DataArray(decoded.variable, coords=variable.coords, name=variable.name)
    result.encoding = dict(decoded.encoding)
    return result


def scale_and_fill(variable: xr.DataArray) -> xr.DataArray:
    """Encode one variable using its CF encoding, without writing a file.

    Conversion stays lazy. Integer overflow, missing fill codes and collisions
    between valid values and fill codes raise when their pixels are evaluated.

    Args:
        variable: Decoded values with per-variable CF storage encoding.

    Returns:
        Stored values with consumed encoding moved to CF attrs.

    Raises:
        ValueError: Attrs and encoding conflict, or integer conversion would
            overflow, lose missing values, or turn valid values into nodata.
    """
    encoding = variable.encoding
    dtype = np.dtype(encoding.get("dtype", variable.dtype))
    if not attrs.Packing.is_stated_in(encoding) and dtype == variable.dtype:
        return variable
    if any(key in variable.attrs and key in encoding for key in attrs.Packing.KEYS):
        raise ValueError(
            f"{str(variable.name)!r} declares CF packing in both attrs and "
            "encoding; decode or clear the conflicting declaration first"
        )

    packing = attrs.Packing.model_validate(encoding)
    packing.check_dtype(dtype)
    checked = variable
    if dtype.kind in "iu":
        checked = xr.apply_ufunc(
            _check_integer,
            variable,
            kwargs={"packing": packing, "dtype": dtype},
            dask="parallelized",
            output_dtypes=[variable.dtype],
            keep_attrs=True,
        )
        checked.encoding = dict(encoding)

    encoded = encode_cf_variable(checked.variable, name=variable.name)
    result = xr.DataArray(encoded, coords=variable.coords, name=variable.name)
    result.encoding = dict(encoded.encoding)
    return result


def _check_integer(
    pixels: np.ndarray, *, packing: attrs.Packing, dtype: np.dtype
) -> np.ndarray:
    """Check one block before the CF encoder rounds and casts it."""
    missing = np.isnan(pixels)
    fills = []
    if packing.fill_value is not None:
        fills.append(packing.fill_value)
    if packing.missing_value is not None:
        fills.extend(np.atleast_1d(packing.missing_value).tolist())
    if missing.any() and not fills:
        raise ValueError("integer encoding needs a fill value for missing pixels")

    values = pixels[~missing]
    if (
        values.dtype.kind in "iu"
        and packing.scale_factor == 1
        and packing.add_offset == 0
    ):
        stored = values
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            stored = np.rint((values - packing.add_offset) / packing.scale_factor)
    limits = np.iinfo(dtype)
    outside = ~np.isfinite(stored)
    outside |= stored < int(limits.min)
    outside |= stored >= int(limits.max) + 1
    if outside.any():
        raise ValueError(
            f"encoded values overflow {dtype}; choose a wider dtype or scale"
        )
    if np.isin(stored, fills).any():
        raise ValueError("encoded valid values collide with a missing-value code")
    return pixels


def mask_missing(variable: xr.DataArray) -> xr.DataArray:
    """Mask stored missing pixels while retaining digital-number units."""
    packing = attrs.Packing.from_attrs(variable.attrs)
    if packing is None or (
        packing.fill_value is None and packing.missing_value is None
    ):
        return variable
    packing.check_dtype(variable.dtype)

    canonical = _canonicalize_missing(variable)
    coded = canonical.variable.copy(deep=False)
    coded.encoding.setdefault("dtype", variable.dtype)
    masked = CFMaskCoder().decode(coded, name=variable.name)
    result = xr.DataArray(masked, coords=variable.coords, name=variable.name)
    result.encoding = dict(masked.encoding)
    return result


def _canonicalize_missing(variable: xr.DataArray) -> xr.DataArray:
    """Spell missing codes as CF attrs at the precision of the stored pixels."""
    canonical = variable.copy(deep=False)
    if "nodata" in canonical.attrs:
        canonical.attrs["_FillValue"] = canonical.attrs.pop("nodata")

    # Float32 header tags may have been parsed as Python floats. Compare codes
    # at pixel precision in both full decoding and masking-only consumers.
    for key in ("_FillValue", "missing_value"):
        if key in canonical.attrs and canonical.attrs[key] is not None:
            value = np.asarray(canonical.attrs[key]).astype(variable.dtype)
            canonical.attrs[key] = value.tolist()
    return canonical
