"""Cutting rasters down to a region or a pixel window.

`crop` follows ground named by geometry; `window` follows pixel offsets, as
a native Tiler cuts them, without computing pixels.

Examples:
    A field's pixels, and a padded chip at the scene's corner::

        field = crop(scene, boundaries)
        chip = window(scene, (0, 0), (256, 256), padding=((16, 16), (16, 16)))
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.xr import xr_coords

from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS
from geosave_engine.geodata.core.stack import map_groups
from geosave_engine.geodata.transform import nodata
from geosave_engine.geodata.transform.vector import rasterize

# How halo and fringe pixels are filled, as numpy names its pad modes.
type PaddingMode = Literal["constant", "edge", "reflect", "wrap"]

# A region to cut against, carrying its own CRS.
type Region = Geometry | gpd.GeoSeries | gpd.GeoDataFrame


def crop[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T, region: Region, *, mask: bool = True
) -> T:
    """Cut a raster, band, or stack down to the extent of a region.

    Args:
        data: Dataset or DataArray on a regular grid, or a stack of them,
            which is cut group by group.
        region: Geometry, or geometries, to cut against, reprojected onto
            `data`'s CRS where they sit in another. Disconnected geometries
            share one cut, and `mask` blanks the pixels between them.
        mask: Also make nodata the pixels outside the geometries, which then
            take their own variable's fill value.

    Returns:
        `data` covering the region's extent, its dtype unchanged.

    Raises:
        TypeError: `region` carries no CRS, as a bare shapely geometry.
        ValueError: `data` carries no CRS, `region` is empty or does not
            overlap `data`, the crop leaves no regular grid to mask on, or
            `mask` is set while a variable carries no fill value.

    Examples:
        >>> crop(scene, field_boundaries).gs.geobox.shape
        (64, 48)
        >>> crop(scene, box(10, 10, 30, 30, "EPSG:32633"))
    """
    if isinstance(data, xr.DataTree):
        cut = map_groups(data, lambda raster: _crop_raster(raster, region, mask))
    else:
        cut = _crop_raster(data, region, mask)
    return cast("T", cut)


def _crop_raster[R: xr.DataArray | xr.Dataset](
    raster: R, region: Region, mask: bool
) -> R:
    """Cut one raster or band down to the extent of a region."""
    crs = raster.odc.crs
    if crs is None:
        raise ValueError(
            f"{type(raster).__name__} carries no CRS; write one with "
            f"gs.write_crs before cropping"
        )

    # The region moves onto the raster's CRS; the raster's pixels never move.
    vector = _to_frame(region).to_crs(crs)

    # odc's own apply_mask writes NaN, which would promote integer variables.
    cut = raster.odc.crop(vector.gs.footprint, apply_mask=False)
    if isinstance(cut, xr.Dataset):
        # The cut pixels are no longer the file's.
        cut.encoding.pop("source", None)

    if mask:
        if not isinstance(cut.odc.geobox, GeoBox):
            raise ValueError(
                f"crop left {type(cut).__name__} with no regular grid, so mask "
                f"has nothing to rasterize the vector onto"
            )
        # Sliced coordinates drift by float error, so the mask goes by position.
        inside = rasterize(vector, cut).values
        cut = nodata.mask(cut, inside)
    return cast("R", cut)


def _to_frame(region: Region) -> gpd.GeoDataFrame:
    """Read a region as a GeoDataFrame carrying its CRS."""
    if isinstance(region, gpd.GeoDataFrame):
        frame = region
    elif isinstance(region, gpd.GeoSeries):
        frame = gpd.GeoDataFrame(geometry=region)
    elif isinstance(region, Geometry) and region.crs is not None:
        frame = gpd.GeoDataFrame(geometry=[region.geom], crs=region.crs.proj)
    else:
        # Shapely carries no CRS, and assuming one would cut the wrong ground.
        raise TypeError(
            f"{type(region).__name__} carries no CRS; pass an odc Geometry, "
            f"GeoSeries, or GeoDataFrame"
        )
    return frame


def window[T: xr.DataArray | xr.Dataset | xr.DataTree](
    data: T,
    offset: tuple[int, int],
    shape: tuple[int, int],
    *,
    padding: Sequence[tuple[int, int]] = ((0, 0), (0, 0)),
    mode: PaddingMode = "constant",
    constant_value: int | float | None = None,
) -> T:
    """Read a pixel window with the same padding its native Tiler uses.

    Args:
        data: Raster, band, or stack; leading dimensions remain intact.
        offset: Row and column of the window's corner in the original raster,
            before padding. Negative values reach outside it.
        shape: Window height and width.
        padding: Halo widths before and after each spatial axis.
        mode: Padding mode for halo and fringe pixels.
        constant_value: Constant fill. None uses xarray's missing-value fill.

    Returns:
        Lazy window of the same native type. Georeferenced rasters retain
        the requested grid, including padded pixels outside the parent.

    Examples:
        >>> window(scene, (0, 0), (256, 256)).sizes
        Frozen({'y': 256, 'x': 256})
    """

    def cut(raster: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        return _window_raster(raster, offset, shape, padding, mode, constant_value)

    if isinstance(data, xr.DataTree):
        result = map_groups(data, lambda raster: cast("xr.Dataset", cut(raster)))
    else:
        result = cut(data)
    return cast("T", result)


def _window_raster(
    raster: xr.DataArray | xr.Dataset,
    offset: tuple[int, int],
    shape: tuple[int, int],
    padding: Sequence[tuple[int, int]],
    mode: PaddingMode,
    constant_value: int | float | None,
) -> xr.DataArray | xr.Dataset:
    """Read one pixel window from a raster or band."""
    row, col = offset
    height, width = shape
    (top, bottom), (left, right) = padding
    y, x = SPATIAL_DIMENSIONS

    # A negative offset names pixels outside the parent, so the halo grows to hold them.
    top = max(top, -row)
    left = max(left, -col)
    padded = _pad(raster, {y: (top, bottom), x: (left, right)}, mode, constant_value)

    # The halo shifted the parent's corner by `top` rows and `left` columns.
    first_row = row + top
    first_col = col + left
    tile = padded.isel(
        {
            y: slice(first_row, first_row + height),
            x: slice(first_col, first_col + width),
        }
    )

    # A tile cut short at the parent's far edge is padded out to full size.
    fringe = {y: (0, height - tile.sizes[y]), x: (0, width - tile.sizes[x])}
    tile = _pad(tile, fringe, mode, constant_value)

    # Reflected coordinate labels repeat, so the tile takes its own grid's.
    grid = raster.gs.geobox
    if isinstance(grid, GeoBox):
        tile_grid = grid.translate_pix(col, row).crop(shape)
        tile = tile.assign_coords(xr_coords(tile_grid, dims=SPATIAL_DIMENSIONS))
    return tile


def _pad(
    data: xr.DataArray | xr.Dataset,
    widths: Mapping[str, tuple[int, int]],
    mode: PaddingMode,
    constant_value: int | float | None,
) -> xr.DataArray | xr.Dataset:
    """Pad pixels lazily, including reflection wider than the source."""
    if not any(before or after for before, after in widths.values()):
        # Padding nothing would still promote integers to hold xarray's NaN fill.
        padded = data
    elif mode in ("reflect", "wrap"):
        # Dask truncates repeated reflection, so pad index arrays and select by them.
        indices = {
            dim: np.pad(np.arange(data.sizes[dim]), width, mode=mode)
            for dim, width in widths.items()
        }
        padded = data.isel(indices)
    elif mode == "constant":
        padded = data.pad(widths, mode=mode, constant_values=constant_value)
    else:
        padded = data.pad(widths, mode=mode)
    return padded
