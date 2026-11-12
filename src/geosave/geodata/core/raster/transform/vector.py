"""Numerical kernels for polygonizing one raster plane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

_POLYGONIZABLE = frozenset(
    np.dtype(name) for name in ("uint8", "uint16", "int16", "int32", "float32")
)


def _narrow_dtype(values: np.ndarray) -> np.ndarray:
    """Narrow values onto a dtype rasterio polygonizes."""
    if values.dtype == np.bool_:
        return values.astype("uint8")
    if values.dtype.kind in "iu" and values.dtype not in _POLYGONIZABLE:
        # An argmax emits int64, which narrows to int32 when its values fit.
        bounds = np.iinfo("int32")
        if values.size and (values.min() < bounds.min or values.max() > bounds.max):
            raise ValueError(
                f"flags are {values.dtype} and hold values beyond int32, the "
                f"widest integer raster polygonization supports"
            )
        return values.astype("int32")
    if values.dtype not in _POLYGONIZABLE:
        raise ValueError(f"raster polygonization does not support dtype {values.dtype}")
    return values


def _read_valid_pixels(
    values: np.ndarray,
    fill: float | int | None,
    mask: xr.DataArray | np.ndarray | None,
    geobox: GeoBox,
) -> np.ndarray:
    """Read which pixels hold a finite value other than nodata, narrowed by `mask`."""
    valid = np.isfinite(values)
    if fill is not None:
        valid &= values != fill
    if mask is None:
        return valid

    if isinstance(mask, xr.DataArray):
        if mask.gs.geobox != geobox:
            raise ValueError("mask must be on the exact flags grid")
        mask_values = np.asarray(mask.compute().values)
    else:
        mask_values = np.asarray(mask)
    if mask_values.shape != values.shape:
        raise ValueError(
            f"mask shape {mask_values.shape} does not match flags {values.shape}"
        )
    return valid & mask_values.astype(bool)


def _polygonize(
    values: np.ndarray,
    valid: np.ndarray,
    geobox: GeoBox,
    value_name: str,
    connectivity: Literal[4, 8],
) -> GeoDataFrame:
    """Trace one polygon per contiguous region of equal value."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    features = list(
        shapes(
            values, mask=valid, transform=geobox.transform, connectivity=connectivity
        )
    )
    geometries = [shape(geometry) for geometry, _ in features]

    # Rasterio's values are numpy scalars, read out as plain Python numbers.
    if features:
        scalar = values.dtype.type
        region_values = [scalar(value).item() for _, value in features]
    else:
        # An empty column still says what type its values would hold.
        region_values = pd.Series(dtype=values.dtype)

    # Regions touching at a corner trace one self-touching ring.
    regions = gpd.GeoDataFrame(
        {value_name: region_values},
        geometry=gpd.GeoSeries(geometries, crs=geobox.crs).make_valid(),
    )
    return cast("GeoDataFrame", regions)


def vectorize(
    flags: xr.DataArray,
    *,
    value_name: str = "value",
    mask: xr.DataArray | np.ndarray | None = None,
    connectivity: Literal[4, 8] = 4,
) -> GeoDataFrame:
    """Polygonize one raster plane with explicit validity and value identity."""
    flags.gs.validate()
    geobox = cast("GeoBox", flags.gs.geobox)
    dims = flags.gs.spatial_dims
    if flags.ndim != 2 or tuple(flags.dims) != dims:
        raise ValueError(
            "vectorize needs one two-dimensional spatial plane on "
            f"{dims}; select extra dimensions from {flags.dims} first"
        )
    if value_name == "geometry":
        raise ValueError("value_name must not replace the geometry column")

    values = _narrow_dtype(np.asarray(flags.compute().values))
    valid = _read_valid_pixels(values, flags.gs.nodata, mask, geobox)
    return _polygonize(values, valid, geobox, value_name, connectivity)
