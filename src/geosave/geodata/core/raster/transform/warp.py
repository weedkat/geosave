"""Destination grids and resampling inputs for raster accessors."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr
from ..utils.grid import resolve_grid, clear_mapping_attr
from odc.geo.geobox import GeoBox

from geosave import __path__ as _package_paths
from geosave.geodata.warnings import AssumedFillWarning

if TYPE_CHECKING:
    from odc.geo import SomeCRS, SomeResolution

# Kernel names odc forwards to rasterio.warp.Resampling.
type Resampling = Literal[
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
    "rms",
]


def read_kernel(
    resampling: Resampling | Mapping[str, Resampling], name: str
) -> Resampling:
    """Read the kernel one variable warps with, `"*"` answering for unnamed variables.

    Raises:
        KeyError: A mapping names neither `name` nor `"*"`.
    """
    if not isinstance(resampling, Mapping):
        return resampling
    if name in resampling:
        return resampling[name]
    if "*" in resampling:
        return resampling["*"]
    raise KeyError(
        f"{name!r} names no resampling kernel and the mapping carries "
        "no '*'; name it, or provide a default kernel"
    )


def _write_default_nodata(variable: xr.DataArray) -> xr.DataArray:
    """Give a variable carrying no fill value one, warning which.

    Returns:
        The variable, carrying NaN where float and its dtype's far end where
        integer: the maximum where unsigned, the minimum where signed.
    """
    if variable.gs.nodata is not None or variable.dtype.kind not in "iuf":
        return variable
    if variable.dtype.kind == "f":
        fill = float("nan")
    else:
        limits = np.iinfo(variable.dtype)
        fill = int(limits.max if variable.dtype.kind == "u" else limits.min)
    warnings.warn(
        f"{str(variable.name)!r} carries no fill value, so the pixels its warp "
        f"does not cover take {fill}; choose one with gs.write_nodata(...) "
        f"before warping",
        AssumedFillWarning,
        skip_file_prefixes=tuple(_package_paths),
    )
    return variable.gs.write_nodata(fill)


def reproject(
    variable: xr.DataArray,
    how: GeoBox | SomeCRS,
    *,
    resolution: SomeResolution | None = None,
    resampling: Resampling | Mapping[str, Resampling] = "nearest",
) -> xr.DataArray:
    """Reproject one array, retaining axis order, identity and nodata."""
    from geosave.geodata.conventions import SPATIAL_DIMENSIONS, to_yx

    variable.gs.validate(georeferenced=True, affine=True)
    kernel = read_kernel(resampling, str(variable.name))
    dims = cast("tuple[str, str]", variable.gs.spatial_dims)
    if variable.gs.is_categorical and kernel not in {"nearest", "mode"}:
        raise ValueError(
            "class codes require a value-preserving kernel: nearest or mode"
        )
    source = cast("GeoBox", variable.gs.geobox)
    geobox = resolve_grid(source, how, resolution)
    if geobox == source:
        return variable

    variable = _write_default_nodata(variable)
    fill = variable.gs.nodata
    # ODC needs spatial axes last; restore the caller's dimension order afterwards.
    yx_last = to_yx(variable).transpose(..., *SPATIAL_DIMENSIONS)
    pixels = to_yx(yx_last.odc.reproject(geobox, resampling=kernel))
    if dims != SPATIAL_DIMENSIONS:
        pixels = pixels.rename(dict(zip(SPATIAL_DIMENSIONS, dims, strict=True)))
    pixels = pixels.transpose(*variable.dims).rename(variable.name)
    # ODC omits NaN fill metadata and narrows some floating fill values.
    if fill is not None:
        pixels = pixels.gs.write_nodata(fill)
    return clear_mapping_attr(pixels)
