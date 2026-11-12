"""Destination grids and resampling inputs for raster accessors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import xarray as xr
from odc.geo.geobox import GeoBox


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


def reproject(
    variable: xr.DataArray,
    how: GeoBox | SomeCRS,
    *,
    resolution: SomeResolution | None = None,
    resampling: Resampling = "nearest",
) -> xr.DataArray:
    """Reproject one array through ODC, returning spatial axes last."""
    variable.gs.validate()
    if variable.gs.is_categorical and resampling not in {"nearest", "mode"}:
        raise ValueError(
            "class codes require a value-preserving kernel: nearest or mode"
        )
    source = cast("GeoBox", variable.gs.geobox)
    if isinstance(how, GeoBox):
        geobox = how
    else:
        size = "auto" if resolution is None else resolution
        geobox = variable.gs.output_geobox(how, resolution=size)

    fill = variable.gs.nodata
    spatial_last = variable.transpose(
        ..., *cast("tuple[str, str]", variable.gs.spatial_dims)
    )
    if geobox == source:
        return spatial_last

    pixels = spatial_last.odc.reproject(geobox, resampling=resampling)
    pixels = pixels.rename(variable.name)
    # ODC omits NaN fill metadata and narrows some floating fill values.
    if fill is not None:
        pixels = pixels.gs.write_nodata(fill)
    return pixels
