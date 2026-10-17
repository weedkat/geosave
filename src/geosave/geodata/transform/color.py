"""Bake the colours a variable's legend names into display channels."""

from __future__ import annotations

import numpy as np
import xarray as xr

from geosave.geodata.conventions import BAND_DIMENSION, SPATIAL_DIMENSIONS
from geosave.geodata.utils.color import parse_color


def colorize(variable: xr.DataArray) -> xr.DataArray:
    """Colour each pixel by the class its code names.

    Args:
        variable: Class codes carrying a `Legend`.

    Returns:
        Lazy georeferenced array shaped `(*axes, band, y, x)` valued in `[0, 1]`,
        whose `band` coordinate is `("red", "green", "blue")`. A pixel no
        class names is absent on every channel.

    Raises:
        ValueError: The variable lists no classes, or a class carries no colour.

    Examples:
        >>> colorize(ds["landcover"]).sizes["band"]
        3
    """
    legend = variable.gs.legend
    if legend is None:
        raise ValueError(
            "the variable lists no classes, so its values name none to colour; "
            "write a Legend, or compose channels with GeoDataset.to_array"
        )
    colors = legend.colors
    codes = np.array(sorted(colors))
    # One row per class, its red, green and blue in [0, 1].
    palette = np.array([parse_color(colors[code]) for code in codes], dtype="float32")
    palette /= 255.0

    def look_up(pixels: np.ndarray) -> np.ndarray:
        named = np.isin(pixels, codes)
        rows = np.where(named, np.searchsorted(codes, pixels), 0)
        return np.where(named[..., None], palette[rows], np.float32(np.nan))

    channels = xr.apply_ufunc(
        look_up,
        variable,
        output_core_dims=[[BAND_DIMENSION]],
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {BAND_DIMENSION: 3}},
        keep_attrs=False,
    )
    return (
        channels.assign_coords(variable.coords)
        .assign_coords({BAND_DIMENSION: ["red", "green", "blue"]})
        .transpose(*variable.gs.axes, BAND_DIMENSION, *SPATIAL_DIMENSIONS)
        .rename(None)
    )
