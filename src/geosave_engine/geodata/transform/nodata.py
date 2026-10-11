"""Marking pixels nodata and decoding stored values.

A pixel's nodata is its `Nodata`, never an implicit NaN. Packing is read out
only after nodata is blanked, so a stored fill value is never scaled into a
bogus physical reading.

Examples:
    A scene classification band says which pixels are cloud::

        clear = mask(scene, scene.scl.isin(CLEAR_CLASSES))
        reflectance = unpack(to_nan(clear))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr
from xarray.coding.variables import CFMaskCoder, CFScaleOffsetCoder

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable


def mask[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T,
    valid: xr.DataArray | np.ndarray,
    *,
    fill: float | int | None = None,
) -> T:
    """Make nodata every pixel a boolean mask does not keep.

    Blanked pixels take the fill value their own variable carries, so no dtype
    is promoted and no variable gains a NaN it never carried.

    Args:
        data: DataArray, Dataset, or DataTree to mask.
        valid: Boolean array, True where a pixel is real data. A DataArray
            names its own axes and may span fewer than `data`, broadcasting
            over the rest; a bare numpy array names none, so it is read as the
            grid alone and must match its shape.
        fill: Value the blanked pixels take, also written onto the result for
            variables carrying no fill value yet. None reads what each
            variable already carries.

    Returns:
        New object of the same kind, its unkept pixels holding the fill value
        their variable carries.

    Raises:
        ValueError: `valid` is not boolean, does not span the grid, spans an
            axis `data` does not, `fill` does not fit a variable's dtype, or a
            variable carries no fill value and none is given.

    Examples:
        Sentinel-2 ships a scene classification band naming what each pixel is;
        keep only the classes that are actually ground:

        >>> clear = mask(scene, scene.scl.isin([4, 5, 6, 7]))
        >>> int((clear.red == 0).sum()) > 0
        True

        A variable carrying no fill value is refused rather than promoted, so
        say what nodata looks like and it travels onto the result:

        >>> mask(unpacked, valid, fill=255).red.attrs["_FillValue"]
        255
    """
    if valid.dtype != bool:
        raise ValueError(
            f"valid is {valid.dtype}, not boolean; pass a comparison such as "
            f"scl.isin([...]) rather than the values themselves"
        )

    if isinstance(data, xr.DataTree):

        def mask_group(raster: xr.Dataset) -> xr.Dataset:
            # The root of a stack holds no variables, so it has nothing to mask.
            return _mask_raster(raster, valid, fill) if raster.data_vars else raster

        result = data.map_over_datasets(mask_group)
    else:
        result = _mask_raster(data, valid, fill)
    return cast("T", result)


def _mask_raster[R: xr.DataArray | xr.Dataset](
    raster: R,
    valid: xr.DataArray | np.ndarray,
    fill: float | int | None,
) -> R:
    """Make nodata every pixel of one raster or band a mask does not keep."""
    valid = _grid_mask(raster, valid)

    # Writing the fill first means every variable reads one spelling of nodata.
    if fill is not None:
        raster = raster.gs.write_nodata(fill)

    def blank(band: xr.DataArray) -> xr.DataArray:
        band_fill = band.gs.nodata
        if band_fill is None:
            raise ValueError(
                f"{str(band.name)!r} carries no fill value; write one with "
                f"raster.gs.write_nodata(...) first"
            )
        return band.where(valid, band_fill)

    return _map_bands(raster, blank)


def _grid_mask(
    raster: xr.DataArray | xr.Dataset, valid: xr.DataArray | np.ndarray
) -> xr.DataArray:
    """Read a mask as a DataArray spanning the raster's grid and no foreign axis."""
    grid_dims = list(SPATIAL_DIMENSIONS)

    # A bare array names no axes, so it can only be the grid itself.
    if isinstance(valid, np.ndarray):
        grid_shape = tuple(raster.sizes[dim] for dim in SPATIAL_DIMENSIONS)
        if valid.shape != grid_shape:
            raise ValueError(
                f"valid is shaped {valid.shape} but the grid {grid_dims} is "
                f"{grid_shape}; a bare array names no axes, so pass a DataArray "
                f"to mask along anything else"
            )
        valid = xr.DataArray(valid, dims=SPATIAL_DIMENSIONS)

    if not set(SPATIAL_DIMENSIONS) <= set(valid.dims):
        raise ValueError(
            f"valid spans {list(valid.dims)}, which does not cover the grid "
            f"{grid_dims}; a mask names which pixels survive, so it spans at "
            f"least the two spatial axes"
        )

    foreign = sorted(str(dim) for dim in valid.dims if dim not in raster.dims)
    if foreign:
        raise ValueError(
            f"valid spans {foreign}, which {type(raster).__name__} does not "
            f"carry, so those axes name no pixels to mask; select them off "
            f"valid first"
        )
    return valid


def to_nan[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T:
    """Replace each variable's fill value with NaN.

    A reduction reads a stored fill value as data unless it finds NaN there
    instead. A variable carrying no fill value passes through unchanged. A
    stack is blanked group by group.

    Args:
        data: DataArray, Dataset, or DataTree holding stored values.

    Returns:
        New object of the same kind holding NaN where the pixels were nodata.
        The fill value leaves attrs with them, since no pixel holds it now.

    Examples:
        >>> to_nan(scene).red.dtype
        dtype('float32')
        >>> to_nan(scene).red.attrs
        {}
    """
    return _map_bands(data, _band_to_nan)


def unpack[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T:
    """Read physical values out of each variable's stored digital numbers.

    A variable carrying neither `scale_factor` nor `add_offset` passes through
    unchanged. A stack is unpacked group by group.

    Args:
        data: DataArray, Dataset, or DataTree holding stored values.

    Returns:
        New object of the same kind holding physical values, float32 where
        the stored numbers are float32 or integers of 16 bits or fewer, and
        float64 otherwise. Each unpacked variable's packing is dropped from
        attrs, since its values are no longer the stored numbers packing
        described.

    Raises:
        ValueError: A packed variable holds class codes, or still marks its
            nodata pixels with a fill value, which scaling would read as an
            ordinary value.

    Examples:
        >>> unpack(to_nan(scene)).red.max().item()
        0.09
    """
    return _map_bands(data, _band_unpack)


def _map_bands[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T, change: Callable[[xr.DataArray], xr.DataArray]
) -> T:
    """Apply one per-variable change to a band, raster, or stack.

    Args:
        data: DataArray, Dataset, or DataTree to change.
        change: Returns the changed variable for one variable.

    Returns:
        New object of the same kind, every data variable changed. A Dataset's
        encoding no longer names the file it was read from.
    """
    if isinstance(data, xr.DataTree):
        changed = data.map_over_datasets(lambda raster: _map_bands(raster, change))
    elif isinstance(data, xr.DataArray):
        changed = change(data)
    else:
        bands = {name: change(band) for name, band in data.data_vars.items()}
        changed = data.assign(bands)
        # Its pixels are no longer the file's.
        changed.encoding.pop("source", None)
    return cast("T", changed)


def _band_to_nan(band: xr.DataArray) -> xr.DataArray:
    """Replace one variable's fill value with NaN."""
    blanked = band
    nodata = attrs.parse(band, attrs.Nodata)
    if nodata is not None:
        # Normalize supported spellings before the native CF codec reads the fill.
        variable = attrs.stamp(band, attrs.Nodata, nodata).variable

        # Packing read off disk would widen 16-bit bands to float64; mask without it.
        packing = {}
        for key in attrs.Packing.KEYS:
            if key in variable.attrs:
                packing[key] = variable.attrs.pop(key)

        decoded = CFMaskCoder().decode(variable, name=band.name)
        blanked = xr.DataArray(
            decoded, coords=band.coords, name=band.name, attrs=decoded.attrs | packing
        )
    return blanked


def _band_unpack(band: xr.DataArray) -> xr.DataArray:
    """Read physical values out of one variable's stored digital numbers."""
    packed = band.gs.packing is not None

    # A class code names a class, so scaling it yields a number naming none.
    if packed and attrs.is_flag(band):
        raise ValueError(
            f"{str(band.name)!r} holds class codes and declares packing, which "
            f"cannot both be true; drop its Legend or its Packing, whichever "
            f"does not describe the pixels"
        )
    fill = band.gs.nodata
    if packed and fill is not None:
        raise ValueError(
            f"{str(band.name)!r} marks its nodata pixels with {fill!r}, which "
            f"scaling turns into an ordinary reading nothing can tell from data; "
            f"blank them with nodata.to_nan first, then unpack"
        )

    physical = band
    if packed:
        # Copy so decoding never writes into the caller's attrs.
        stored = band.variable.copy(deep=False)

        # CF reads these out as float32 only where the packing is float32 too.
        small_int = band.dtype.kind in "iu" and band.dtype.itemsize <= 2
        if small_int or band.dtype == np.float32:
            for key in attrs.Packing.KEYS:
                if key in stored.attrs:
                    stored.attrs[key] = np.float32(stored.attrs[key])

        decoded = CFScaleOffsetCoder().decode(stored, name=band.name)
        physical = xr.DataArray(
            decoded, coords=band.coords, name=band.name, attrs=decoded.attrs
        )
    return physical
