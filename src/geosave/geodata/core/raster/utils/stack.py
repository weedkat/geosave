"""Stack rasters of one grid onto a GeoBox, as odc-stac builds one dask array per band.

Pixels stack as dask arrays and the xarray structure is built once from the
GeoBox, so many rasters cost no per-raster alignment.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import dask.array as da
import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave.geodata.core.conventions import GRID_MAPPING


def stack_rasters(
    rasters: Sequence[xr.Dataset], geobox: GeoBox, dim: str
) -> xr.Dataset:
    """Stack rasters of one grid along `dim` onto `geobox`, NaN where one states no pixels.

    Args:
        rasters: Rasters on the grid of `geobox`, each within its extent, with
            the same scalar coordinates, such as `time`.
        geobox: GeoBox every raster fits in, such as the union of theirs.
        dim: New dimension the rasters stack along.

    Returns:
        Lazy Dataset over `geobox`; each variable keeps the attrs of the first
        raster holding it, and the Dataset the first raster's attrs.

    Examples:
        >>> stack_rasters([early, late], early.odc.geobox | late.odc.geobox, "time")
        <xarray.Dataset> Dimensions: (time: 2, y: 64, x: 64) ...
    """
    first = rasters[0]

    names: list[Hashable] = []
    for raster in rasters:
        for name in raster.data_vars:
            if name not in names:
                names.append(name)

    variables = {}
    for name in names:
        holder = next(raster for raster in rasters if name in raster)
        pixels = stack_pixels(rasters, name, geobox)
        variables[name] = ((dim, *geobox.dimensions), pixels, holder[name].attrs)

    # Pixel labels and the CRS coordinate describe the GeoBox; the labels keep
    # the rasters' attrs, which name the CRS as the rasters do.
    coords = xr_coords(geobox)
    for name in geobox.dimensions:
        coords[name].attrs = dict(first[name].attrs)

    # Each raster's scalar coordinates become one coordinate along `dim`.
    for name, coord in first.coords.items():
        if coord.ndim == 0 and name != GRID_MAPPING:
            values = np.stack([raster[name].values for raster in rasters])
            coords[name] = xr.DataArray(values, dims=dim, attrs=coord.attrs)

    return xr.Dataset(variables, coords=coords, attrs=first.attrs)


def stack_pixels(
    rasters: Sequence[xr.Dataset], name: Hashable, geobox: GeoBox
) -> da.Array:
    """Stack one variable's pixels from every raster onto `geobox` as one dask array.

    Returns:
        Lazy array shaped `(raster, y, x)`; float where a raster lacks the
        variable or covers less than `geobox`, so those pixels can be NaN.
    """
    holder = next(raster for raster in rasters if name in raster)
    dtype = holder[name].dtype
    chunks = da.asarray(holder[name].data).chunksize

    # On one grid, a raster covers the GeoBox exactly when its shape matches.
    for raster in rasters:
        if name not in raster or raster[name].shape != geobox.shape.yx:
            dtype = np.promote_types(dtype, np.float32)

    layers = []
    for raster in rasters:
        if name in raster:
            layers.append(extend_pixels(raster[name], geobox, dtype))
        else:
            layers.append(da.full(geobox.shape.yx, np.nan, dtype=dtype, chunks=chunks))
    return da.stack(layers)


def extend_pixels(band: xr.DataArray, geobox: GeoBox, dtype: np.dtype) -> da.Array:
    """Extend one band's pixels onto a larger GeoBox of its grid, NaN outside.

    Returns:
        Lazy array shaped as `geobox`.
    """
    pixels = da.asarray(band.data).astype(dtype)
    if band.shape == geobox.shape.yx:
        return pixels

    rows, columns = geobox.overlap_roi(band.odc.geobox)
    padding = (
        (rows.start, geobox.shape.y - rows.stop),
        (columns.start, geobox.shape.x - columns.stop),
    )
    return da.pad(pixels, padding, constant_values=np.nan)
