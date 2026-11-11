"""Numerical kernels for burning vector properties onto a grid."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from odc.geo.geobox import GeoBox

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def burn_presence(
    frame: gpd.GeoDataFrame, geobox: GeoBox, fill: int | float | bool, all_touched: bool
) -> np.ndarray:
    """Burn True inside the geometries and False outside."""
    # A presence mask is True inside, so a truthy fill would mark every pixel.
    if fill:
        raise ValueError(
            f"a presence mask is True inside the geometries, so fill={fill!r} "
            f"would mark every pixel; pass column to burn values, or drop fill"
        )
    ones = np.ones(len(frame), dtype="uint8")
    return _burn(frame, ones, geobox, 0, all_touched).view(bool)


def burn_column(
    frame: gpd.GeoDataFrame,
    geobox: GeoBox,
    column: str,
    fill: int | float | bool,
    dtype: DTypeLike | None,
    all_touched: bool,
) -> np.ndarray:
    """Burn one numeric property, refusing values its dtype stores as other numbers."""
    if column not in frame:
        raise KeyError(f"the vector has no {column!r} column")
    series = cast("pd.Series", frame[column])
    if series.isna().any():
        raise ValueError("rasterize values must not be null")

    column_values = np.asarray(series.tolist())
    output_dtype = _resolve_dtype(column_values, fill, dtype)
    values = _cast_losslessly(column_values, output_dtype, column)
    fill_value = _cast_losslessly(np.asarray([fill]), output_dtype, column)[0].item()
    return _burn(frame, values, geobox, fill_value, all_touched)


def _resolve_dtype(
    values: np.ndarray, fill: int | float | bool, dtype: DTypeLike | None
) -> np.dtype:
    """Read the numeric dtype burned values are stored in."""
    try:
        if dtype is None:
            output_dtype = np.dtype(np.result_type(values, type(fill)))
        else:
            output_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("rasterize needs a numeric raster dtype") from error
    if output_dtype.kind not in "iuf":
        raise ValueError(f"rasterize needs a numeric raster dtype, got {output_dtype}")
    return output_dtype


def _cast_losslessly(values: np.ndarray, dtype: np.dtype, column: str) -> np.ndarray:
    """Cast values to `dtype`, refusing any it stores as another number."""
    if values.dtype.kind not in "biuf":
        raise ValueError("rasterize needs a numeric raster dtype")
    # astype wraps and rounds, so an unheld -1 would burn as 65535 on uint16.
    with np.errstate(invalid="ignore", over="ignore"):
        cast_values = values.astype(dtype)
    if not np.array_equal(cast_values, values, equal_nan=True):
        raise ValueError(
            f"{column!r} or the fill holds values {dtype} stores as "
            f"other numbers; pass a dtype that holds them"
        )
    return cast_values


def _burn(
    frame: gpd.GeoDataFrame,
    values: np.ndarray,
    geobox: GeoBox,
    fill: int | float,
    all_touched: bool,
) -> np.ndarray:
    """Burn each geometry's value onto the grid, later rows winning overlaps."""
    from rasterio.features import rasterize as burn

    if frame.empty:
        return np.full(geobox.shape.yx, fill, dtype=values.dtype)
    pixels = burn(
        zip(frame.geometry, values, strict=True),
        out_shape=geobox.shape.yx,
        transform=geobox.transform,
        fill=fill,  # type: ignore
        dtype=values.dtype,
        all_touched=all_touched,
    )
    if pixels is None:
        raise RuntimeError("Rasterio returned no rasterized array")
    return pixels
