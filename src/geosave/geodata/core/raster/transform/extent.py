"""Crop native rasters and pad them onto exact aligned grids."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Literal, cast
import geopandas as gpd
import numpy as np
import xarray as xr
from odc.geo import SomeCRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox, Geometry
from ..utils.grid import read_window

type Padding = Literal["constant", "edge", "reflect", "wrap"]
type Region = Geometry | BoundingBox | gpd.GeoSeries | gpd.GeoDataFrame


def _read_geometry(region: Region) -> Geometry:
    """Read a region without guessing its CRS."""
    if isinstance(region, gpd.GeoDataFrame):
        geometry = region.gs.footprint
    elif isinstance(region, gpd.GeoSeries):
        geometry = gpd.GeoDataFrame(geometry=region).gs.footprint
    elif isinstance(region, BoundingBox):
        geometry = region.polygon
    else:
        geometry = region

    if not isinstance(geometry, Geometry) or geometry.crs is None:
        raise TypeError(
            f"{type(region).__name__} carries no CRS; pass an odc Geometry, "
            "GeoSeries, or GeoDataFrame"
        )
    return geometry


def _pad_pixels[T: (xr.DataArray, xr.Dataset)](
    data: T,
    widths: Mapping[str, tuple[int, int]],
    mode: Padding,
    constant_value: int | float,
) -> T:
    """Pad pixels; the accessor writes the requested grid afterward."""
    if not any(before or after for before, after in widths.values()):
        return data
    if mode == "constant":
        return data.pad(dict(widths), mode=mode, constant_values=constant_value)

    # Dask truncates repeated reflection/wrapping; select padded indexes instead.
    positions = {}
    for dim, width in widths.items():
        positions[dim] = np.pad(np.arange(data.sizes[dim]), width, mode=mode)
    return data.isel(positions)


def crop[T: (xr.DataArray, xr.Dataset)](
    raster: T, region: Region, *, mask: bool = True
) -> T:
    """Crop one raster to a CRS-bearing region, preserving declared extent."""
    from odc.geo.xr import rasterize

    raster.gs.validate(georeferenced=True)
    footprint = _read_geometry(region).to_crs(cast("SomeCRS", raster.gs.crs))
    cut = raster.odc.crop(footprint, apply_mask=False)
    cut.encoding.pop("source", None)
    if mask:
        cut.gs.validate(affine=True)
        # Rasterization follows the grid by position, avoiding coordinate drift.
        inside = rasterize(footprint, cast("GeoBox", cut.gs.geobox)).values
        cut = cut.gs.mask(inside)
    return cut


def crop_to_grid[T: (xr.DataArray, xr.Dataset)](
    raster: T,
    geobox: GeoBox,
    *,
    mode: Padding = "constant",
    constant_value: int | float = 0,
) -> T:
    """Return pixels on an exact aligned grid, clipping declared extent."""

    raster.gs.validate(affine=True)
    source = cast("GeoBox", raster.gs.geobox)
    slices, widths = read_window(source, geobox)
    dims = cast("tuple[str, str]", raster.gs.spatial_dims)
    window = raster.isel(dict(zip(dims, slices, strict=True)))
    axis_attrs = {}
    for dim in dims:
        axis_attrs[dim] = window[dim].attrs
    window = window.drop_vars(list(axis_attrs))
    selected = _pad_pixels(
        window, dict(zip(dims, widths, strict=True)), mode, constant_value
    )
    selected = selected.gs.write_grid(geobox)
    for dim, metadata in axis_attrs.items():
        selected[dim].attrs = {**metadata, **selected[dim].attrs}
    selected.encoding.pop("source", None)
    return selected.gs.clip_extent(geobox.extent)
