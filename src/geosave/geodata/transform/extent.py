"""Changing how much ground a raster covers.

`crop` cuts a raster down to ground named by geometry; `pad` grows it past
its edges without computing the new pixels.

Examples:
    A field's pixels, and a scene grown by a reflected halo::

        field = crop(scene, boundaries)
        chip = pad(scene, [(16, 16), (16, 16)], mode="reflect")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox, Geometry
from odc.geo.xr import rasterize, xr_coords

from geosave.geodata.conventions import SPATIAL_DIMENSIONS
from geosave.geodata.core.raster.datatree import map_groups
from geosave.geodata.transform import nodata

# Numpy's pad modes, less "wrap": a chip never reads the far edge a wrap needs.
type Padding = Literal["constant", "edge", "reflect"]

# A region to cut against, carrying its own CRS.
type Region = Geometry | BoundingBox | gpd.GeoSeries | gpd.GeoDataFrame


def crop[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T, region: Region, *, mask: bool = True
) -> T:
    """Cut a raster, variable, or tree down to the extent of a region.

    Args:
        data: Dataset or DataArray on a regular grid, or a tree of them,
            which is cut group by group.
        region: Geometry, geometries, or bounding box to cut against,
            reprojected onto `data`'s CRS where they sit in another.
            Disconnected geometries share one cut, and `mask` masks the
            pixels between them.
        mask: Also make nodata the pixels outside the geometries, which then
            take their own variable's fill value.

    Returns:
        `data` covering the region's extent, its dtype unchanged.

    Raises:
        TypeError: `region` carries no CRS, as a bare shapely geometry or
            a BoundingBox built without one.
        ValueError: `data` carries no CRS, `region` is empty or does not
            overlap `data`, the crop leaves no regular grid to mask on, or
            `mask` is set while a variable carries no fill value.

    Examples:
        >>> crop(scene, field_boundaries).gs.geobox.shape
        (64, 48)
        >>> crop(scene, box(10, 10, 30, 30, "EPSG:32633"))
        >>> crop(scene, other.gs.bounds)
    """
    if isinstance(data, xr.DataTree):
        return cast(
            "T", map_groups(data, lambda raster: crop(raster, region, mask=mask))
        )

    crs = data.gs.crs
    if crs is None:
        raise ValueError(
            f"{type(data).__name__} carries no CRS; write one with "
            f"gs.write_crs before cropping"
        )
    if isinstance(region, gpd.GeoDataFrame):
        geometry = region.gs.footprint
    elif isinstance(region, gpd.GeoSeries):
        geometry = gpd.GeoDataFrame(geometry=region).gs.footprint
    elif isinstance(region, BoundingBox):
        geometry = region.polygon
    else:
        geometry = region
    # Assuming a CRS for a bare geometry would cut the wrong ground.
    if not isinstance(geometry, Geometry) or geometry.crs is None:
        raise TypeError(
            f"{type(region).__name__} carries no CRS; pass an odc Geometry, "
            f"GeoSeries, or GeoDataFrame"
        )

    # The region moves onto the raster's CRS; the raster's pixels never move.
    footprint = geometry.to_crs(crs)
    # odc's own apply_mask writes NaN, which would promote integer variables.
    cut = data.odc.crop(footprint, apply_mask=False)
    if isinstance(cut, xr.Dataset):
        # The cut pixels are no longer the file's.
        cut.encoding.pop("source", None)
    if not mask:
        return cast("T", cut)

    grid = cut.gs.geobox
    if not isinstance(grid, GeoBox):
        raise ValueError(
            f"crop left {type(cut).__name__} with no regular grid, so mask "
            f"has nothing to rasterize the region onto"
        )
    # Sliced coordinates drift by float error, so the mask goes by position.
    inside = rasterize(footprint, grid).values
    return cast("T", nodata.mask(cut, inside))


def pad[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T, widths: Sequence[tuple[int, int]], *, mode: Padding = "constant"
) -> T:
    """Grow a raster past its edges, extending its grid over the new pixels.

    Args:
        data: Raster, variable, or tree; leading dimensions remain intact.
        widths: Pixels added before and after `y`, then before and after `x`.
        mode: How the new pixels are filled. `"constant"` takes xarray's
            missing value, which promotes integers to float.

    Returns:
        Lazy object of the same kind. A georeferenced raster carries the grown
        grid, its coordinates continuing the parent's.

    Examples:
        >>> pad(scene, [(16, 16), (16, 16)], mode="reflect").gs.geobox.shape
        Shape2d(x=288, y=288)
    """
    if isinstance(data, xr.DataTree):
        return cast(
            "T", map_groups(data, lambda raster: pad(raster, widths, mode=mode))
        )

    (top, bottom), (left, right) = widths
    # Padding nothing would still promote integers to hold xarray's NaN fill.
    if not any((top, bottom, left, right)):
        return data

    y, x = SPATIAL_DIMENSIONS
    pad_widths = {y: (top, bottom), x: (left, right)}
    if mode == "reflect":
        # Dask truncates repeated reflection, so pad index arrays and select by them.
        positions = {}
        for dim, width in pad_widths.items():
            positions[dim] = np.pad(np.arange(data.sizes[dim]), width, mode=mode)
        grown = data.isel(positions)
    else:
        grown = data.pad(pad_widths, mode=mode)

    # Reflected coordinate labels repeat, so the pixels take the grown grid's.
    grid = data.gs.geobox
    if isinstance(grid, GeoBox):
        shape = (grown.sizes[y], grown.sizes[x])
        grown_grid = grid.translate_pix(-left, -top).crop(shape)
        grown = grown.assign_coords(xr_coords(grown_grid, dims=SPATIAL_DIMENSIONS))
    grown.encoding.pop("source", None)
    return cast("T", grown)
