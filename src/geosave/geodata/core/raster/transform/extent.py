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
from odc.geo.roi import roi_is_empty
from odc.geo.xr import xr_coords

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

    raster.gs.validate()
    footprint = _read_geometry(region).to_crs(cast("SomeCRS", raster.gs.crs))
    cut = raster.odc.crop(footprint, apply_mask=False)
    cut.encoding.pop("source", None)
    if mask:
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
    raster.gs.validate()
    source = cast("GeoBox", raster.gs.geobox)
    try:
        rows, cols = source.overlap_roi(geobox)
        inner_rows, inner_cols = geobox.overlap_roi(source)
    except ValueError as error:
        raise ValueError(
            "geobox is not pixel-aligned with the raster: it needs the same CRS "
            "and pixel size and a whole-pixel offset; reproject otherwise"
        ) from error
    if roi_is_empty((rows, cols)):
        raise ValueError("geobox does not overlap the raster")

    y, x = cast("tuple[str, str]", raster.gs.spatial_dims)
    height, width = geobox.shape
    window = raster.isel({y: rows, x: cols})
    widths = {
        y: (inner_rows.start, height - inner_rows.stop),
        x: (inner_cols.start, width - inner_cols.stop),
    }
    window = window.drop_vars([y, x])
    padded = _pad_pixels(window, widths, mode, constant_value)
    selected = padded.assign_coords(xr_coords(geobox, dims=(y, x)))
    selected.encoding.pop("source", None)
    return selected.gs.clip_extent(geobox.extent)
