"""Bake the colours a band's legend names into display channels."""

from __future__ import annotations

import numpy as np
import odc.geo.xr  # noqa: F401  — registers the .odc accessor
import xarray as xr

import geosave_engine.geodata.attrs as attrs
from geosave_engine.geodata.conventions import BAND_DIMENSION, SPATIAL_DIMENSIONS
from geosave_engine.geodata.utils.color import parse_color


def colorize(band: xr.DataArray) -> xr.DataArray:
    """Colour each pixel by the class its code names.

    Args:
        band: Class codes carrying a `Legend`.

    Returns:
        Lazy georeferenced array shaped `(*axes, band, y, x)` valued in `[0, 1]`,
        whose `band` coordinate is `("red", "green", "blue")`. A pixel no
        class names is absent on every channel.

    Raises:
        ValueError: The band lists no classes, or names a class carrying no
            colour.

    Examples:
        >>> colorize(ds["landcover"]).sizes["band"]
        3
    """
    legend = attrs.parse(band, attrs.Legend)
    class_map = None if legend is None else legend.classes
    if legend is None or class_map is None:
        raise ValueError(
            "band lists no classes, so its values name none to colour; write "
            "a Legend, or compose channels with GeoRaster.to_array"
        )

    colors = legend.color_map or {}
    codes = sorted(class_map)
    missing_colors = [code for code in codes if code not in colors]
    if missing_colors:
        raise ValueError(
            f"classes {missing_colors} carry no colour; give Legend.color_map an "
            f"entry for every class the band lists"
        )

    palette = np.array([parse_color(colors[code]) for code in codes], dtype="float32")
    palette /= 255.0  # (class, 3)

    def lookup(pixels: np.ndarray) -> np.ndarray:
        valid = np.isin(pixels, codes)
        indices = np.where(valid, np.searchsorted(codes, pixels), 0)
        return np.where(valid[..., None], palette[indices], np.float32(np.nan))

    colored = xr.apply_ufunc(
        lookup,
        band,
        output_core_dims=[[BAND_DIMENSION]],
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {BAND_DIMENSION: 3}},
        keep_attrs=False,
    )
    return (
        colored.assign_coords(band.coords)
        .assign_coords({BAND_DIMENSION: ["red", "green", "blue"]})
        .transpose(*band.gs.axes, BAND_DIMENSION, *SPATIAL_DIMENSIONS)
        .rename(None)
    )
