"""Marking pixels nodata and decoding stored values.

A pixel's nodata is its `Nodata`, never an implicit NaN. Packing is read out
only after nodata is masked, so a stored fill value is never scaled into a
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

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import SPATIAL_DIMENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable


def mask[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T,
    valid: xr.DataArray | np.ndarray,
    *,
    fill: float | int | None = None,
) -> T:
    """Make nodata every pixel a boolean mask does not keep.

    Masked pixels take the fill value their own variable carries, so no dtype
    is promoted and no variable gains a NaN it never carried.

    Args:
        data: DataArray, Dataset, or DataTree to mask.
        valid: Boolean array, True where a pixel is real data. A DataArray
            names its own axes and may span fewer than `data`, broadcasting
            over the rest; a bare numpy array names none, so it is read as the
            grid alone and must match its shape.
        fill: Value the masked pixels take, replacing the result's nodata
            metadata for every affected variable. None reads what each
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
        # No pixel moves, so the tree keeps its root; the root holds no variables.
        masked_tree = data.map_over_datasets(
            lambda raster: (
                mask(raster, valid, fill=fill) if raster.data_vars else raster
            )
        )
        return cast("T", masked_tree)

    valid = _validate_mask(data, valid)
    # Writing the fill first gives every variable one stated fill value.
    if fill is not None:
        data = data.gs.write_nodata(fill)

    def mask_variable(variable: xr.DataArray) -> xr.DataArray:
        variable_fill = variable.gs.nodata
        if variable_fill is None:
            raise ValueError(
                f"{str(variable.name)!r} carries no fill value; write one with "
                f"raster.gs.write_nodata(...) first"
            )
        return variable.where(valid, variable_fill)

    return _map_variables(data, mask_variable)


def _validate_mask(
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

    foreign_dims = sorted(str(dim) for dim in valid.dims if dim not in raster.dims)
    if foreign_dims:
        raise ValueError(
            f"valid spans {foreign_dims}, which {type(raster).__name__} does not "
            f"carry, so those axes name no pixels to mask; select them off "
            f"valid first"
        )
    return valid


def to_nan[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T:
    """Replace each variable's fill value with NaN.

    A reduction reads a stored fill value as data unless it finds NaN there
    instead. A variable carrying no fill value passes through unchanged. A
    tree is decoded group by group.

    Args:
        data: DataArray, Dataset, or DataTree holding stored values.

    Returns:
        New object of the same kind holding NaN where the pixels were nodata.
        The fill value leaves attrs with them, since no pixel holds it now.

    Examples:
        >>> to_nan(scene).red.dtype
        dtype('float32')
        >>> to_nan(scene).red.attrs  # packing stays; the fill value is gone
        {'scale_factor': 0.0001, 'add_offset': -0.1}
    """

    def blank_fill(variable: xr.DataArray) -> xr.DataArray:
        value = variable.gs.nodata
        if value is None:
            return variable
        # The codec compares in the variable's dtype, so the fill is cast to it first.
        fill = attrs.Nodata(value=np.asarray(value).astype(variable.dtype).item())
        # Packing read off disk would widen 16-bit variables to float64; mask without it.
        packing = variable.gs.packing
        stamped = attrs.stamp_attrs(variable, fill)
        coded = attrs.clear_attrs(stamped, attrs.Packing).variable
        blanked = xr.DataArray(
            CFMaskCoder().decode(coded, name=variable.name),
            coords=variable.coords,
            name=variable.name,
        )
        return blanked if packing is None else attrs.stamp_attrs(blanked, packing)

    return _map_variables(data, blank_fill)


def unpack[T: xr.DataArray | xr.Dataset | xr.DataTree](data: T) -> T:
    """Read physical values out of each variable's stored digital numbers.

    A variable carrying neither `scale_factor` nor `add_offset` passes through
    unchanged. A tree is unpacked group by group.

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

    def scale(variable: xr.DataArray) -> xr.DataArray:
        if variable.gs.packing is None:
            return variable
        # A class code names a class, so scaling it yields a number naming none.
        if variable.gs.is_categorical:
            raise ValueError(
                f"{str(variable.name)!r} holds class codes and declares packing, "
                f"which cannot both be true; drop its Legend or its Packing, "
                f"whichever does not describe the pixels"
            )
        fill = variable.gs.nodata
        if fill is not None:
            raise ValueError(
                f"{str(variable.name)!r} marks its nodata pixels with {fill!r}, "
                f"which scaling turns into an ordinary reading nothing can tell "
                f"from data; blank them with nodata.to_nan first, then unpack"
            )
        # Copy so decoding never writes into the caller's attrs.
        coded = variable.variable.copy(deep=False)
        # CF reads these out as float32 only where the packing is float32 too.
        is_small_int = variable.dtype.kind in "iu" and variable.dtype.itemsize <= 2
        if is_small_int or variable.dtype == np.float32:
            for key in attrs.Packing.KEYS:
                if key in coded.attrs:
                    coded.attrs[key] = np.float32(coded.attrs[key])
        physical = CFScaleOffsetCoder().decode(coded, name=variable.name)
        return xr.DataArray(
            physical, coords=variable.coords, name=variable.name, attrs=physical.attrs
        )

    return _map_variables(data, scale)


def _map_variables[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T, change: Callable[[xr.DataArray], xr.DataArray]
) -> T:
    """Apply one per-variable change to a variable, raster, or tree.

    Args:
        data: DataArray, Dataset, or DataTree to change.
        change: Returns the changed variable for one variable.

    Returns:
        New object of the same kind, every data variable changed. A Dataset's
        encoding no longer names the file it was read from.
    """
    # No pixel moves, so a tree keeps its root and its coordinates.
    if isinstance(data, xr.DataTree):
        tree = data.map_over_datasets(lambda raster: _map_variables(raster, change))
        return cast("T", tree)
    if isinstance(data, xr.DataArray):
        return cast("T", change(data))

    variables = {}
    for name, variable in data.data_vars.items():
        variables[name] = change(variable)
    raster = data.assign(variables)
    # Its pixels are no longer the file's.
    raster.encoding.pop("source", None)
    return cast("T", raster)
