"""Colour choices figures and the explorer share, where HoloViz leaves them to us.

The default colormap and the colormap picker's choices, and a contrast stretch
for composites, which HoloViews applies per element but not per channel.
Classes and categories take `Display.colors` and `CATEGORY_PALETTE`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

DEFAULT_CMAP = "viridis"

# Offered in the colormap picker; any matplotlib, colorcet or bokeh name is accepted.
CMAPS = ["viridis", "cividis", "magma", "RdYlGn", "YlGn", "Spectral", "gray"]

# Default percentiles a stretch fits, so dark reflectance fills the display.
STRETCH_PERCENTILES = (2.0, 98.0)


def find_percentile_range(
    values: np.ndarray, percentiles: tuple[float, float]
) -> tuple[float, float] | None:
    """Return the values at `percentiles` of the finite values; None when none is finite."""
    finite_values = values[np.isfinite(values)]
    if not finite_values.size:
        return None
    low, high = np.percentile(finite_values, percentiles)
    return float(low), float(high)


def find_stretch_limits(
    values: np.ndarray, percentiles: tuple[float, float] = STRETCH_PERCENTILES
) -> tuple[float, float]:
    """Return the values a stretch draws dark and full: `uint8` as stored, else percentiles.

    Args:
        values: Values of one or more channels.
        percentiles: Percentiles drawn dark and full, for values not `uint8`.

    Raises:
        ValueError: Every value is nan, or the values are flat.
    """
    if values.dtype == np.uint8:
        return 0.0, 255.0
    limits = find_percentile_range(values, percentiles)
    if limits is None:
        raise ValueError("every pixel is nan or nodata; nothing to draw")
    low, high = limits
    if not high > low:
        raise ValueError(f"channels are flat near {low}; pass clim=(low, high)")
    return low, high


def stretch_channels[T: (np.ndarray, xr.DataArray)](
    channels: T,
    clim: tuple[float, float] | None = None,
    percentiles: tuple[float, float] = STRETCH_PERCENTILES,
) -> T:
    """Map channels onto `[0, 1]` between `clim`, clipping outside; nan stays nan.

    Args:
        channels: Values of one or more channels.
        clim: Value drawn dark and value drawn full. None reads them with
            `find_stretch_limits`.
        percentiles: Percentiles `find_stretch_limits` fits when `clim` is None.

    Raises:
        ValueError: `clim` is empty or inverted, or no limits can be read.

    Examples:
        >>> stretch_channels(ds[["B08", "B04", "B03"]].gs.to_array(), (0.0, 0.3))
    """
    if clim is None:
        low, high = find_stretch_limits(np.asarray(channels), percentiles)
    elif clim[1] > clim[0]:
        low, high = clim
    else:
        raise ValueError(f"stretch bounds {clim} are empty or inverted")
    return ((channels.astype("float32") - low) / (high - low)).clip(0.0, 1.0)
