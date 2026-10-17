"""Colors and stretches shared by static figures and interactive legends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import colorcet as cc
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

type Theme = Literal["light", "dark"]

# Panel's classic widgets otherwise prefer the notebook editor's colors over
# the enclosing Material UI Page. These are Panel's supported design overrides.
MATERIAL_STYLESHEET = """
:host {
  --design-background-color: var(--mui-palette-background-default);
  --design-surface-color: var(--mui-palette-background-paper);
  --design-background-text-color: var(--mui-palette-text-primary);
  --design-surface-text-color: var(--mui-palette-text-primary);
  --design-primary-color: var(--mui-palette-primary-main);
  --design-primary-text-color: var(--mui-palette-primary-contrastText);
}
"""

EXPLORER_THEME = {
    "typography": {
        "fontFamily": "Inter, system-ui, sans-serif",
        "fontSize": 12,
        "button": {"textTransform": "none"},
    },
    "shape": {"borderRadius": 5},
    "components": {
        "MuiToolbar": {
            "styleOverrides": {
                "root": {"minHeight": 44, "@media (min-width:600px)": {"minHeight": 44}}
            }
        },
        "MuiTab": {
            "styleOverrides": {
                "root": {"minHeight": 34, "minWidth": 0, "padding": "6px 14px"}
            }
        },
        "MuiTabs": {"styleOverrides": {"root": {"minHeight": 34}}},
        "MuiSlider": {
            "styleOverrides": {
                "root": {"padding": "10px 0"},
                "thumb": {"width": 12, "height": 12},
            }
        },
        "MuiCardHeader": {
            "styleOverrides": {
                "root": {"padding": "8px 12px"},
                "title": {"fontSize": 13, "fontWeight": 600},
            }
        },
        "MuiCardContent": {
            "styleOverrides": {
                "root": {"padding": 12, "&:last-child": {"paddingBottom": 12}}
            }
        },
    },
}


DEFAULT_CMAP = "viridis"

# Colour of features no column colours, and of selected features.
FEATURE_COLOR = "#1f77b4"
SELECTION_COLOR = "#ffd400"

# Offered in the colormap picker; any matplotlib, colorcet or bokeh name is accepted.
CMAPS = ["viridis", "cividis", "magma", "RdYlGn", "YlGn", "Spectral", "gray"]


TABLE_THEMES = {"light": "materialize", "dark": "midnight"}

# Category colours that stay visible on each theme's basemap: dark hues on light, light on dark.
CATEGORY_PALETTE: dict[Theme, list[str]] = {
    "light": cc.glasbey_dark,
    "dark": cc.glasbey_light,
}


@dataclass(frozen=True)
class Stretch:
    """The limits a composite's channels stretch between onto `[0, 1]`.

    Attributes:
        clim: Value drawn dark and value drawn full. None reads them from the
            channels.

    Examples:
        >>> Stretch((0.0, 0.3)).apply(ds[["B08", "B04", "B03"]].gs.to_array())
    """

    clim: tuple[float, float] | None = None

    def get_limits(self, values: np.ndarray) -> tuple[float, float]:
        """Return the limits set, else `uint8` as stored, else the 2-98 percentile.

        The percentile lets dark reflectance fill the display.

        Raises:
            ValueError: The limits set are empty or inverted, every value is
                nan, or the channels are flat.
        """
        if self.clim is not None:
            if not self.clim[1] > self.clim[0]:
                raise ValueError(f"stretch bounds {self.clim} are empty or inverted")
            return self.clim
        if values.dtype == np.uint8:
            return 0.0, 255.0
        finite_values = values[np.isfinite(values)]
        if not finite_values.size:
            raise ValueError("every pixel is nan or nodata; nothing to draw")
        low, high = (float(value) for value in np.percentile(finite_values, (2, 98)))
        if not high > low:
            raise ValueError(f"channels are flat near {low}; pass clim=(low, high)")
        return low, high

    def apply[T: (np.ndarray, xr.DataArray)](self, channels: T) -> T:
        """Map channels onto `[0, 1]`, clipping outside the limits; nan stays nan."""
        low, high = self.get_limits(np.asarray(channels))
        return ((channels.astype("float32") - low) / (high - low)).clip(0.0, 1.0)


def get_category_colors(values: pd.Series, theme: Theme = "light") -> dict[object, str]:
    """Colour each category of a text column from the theme's glasbey palette.

    Categories take colours in sorted order, so one keeps its colour across
    geometry types and redraws; missing values take none.

    Returns:
        {category: "#rrggbb"}

    Examples:
        >>> list(get_category_colors(fields["crop"]))
        ['palm', 'rice']
    """
    palette = CATEGORY_PALETTE[theme]
    categories = sorted(values.dropna().unique(), key=str)
    return {
        category: palette[index % len(palette)]
        for index, category in enumerate(categories)
    }


def get_ramp(cmap: str) -> np.ndarray:
    """Return a matplotlib colormap as 256 RGB rows, low to high."""
    from matplotlib import colormaps

    samples = colormaps[cmap](np.linspace(0.0, 1.0, 256))
    return (samples[:, :3] * 255).round().astype("uint8")


def to_rgb255(color: str) -> tuple[int, int, int]:
    """Return a colour name or hex as 0-255 red, green and blue."""
    from matplotlib.colors import to_rgb

    red, green, blue = (round(channel * 255) for channel in to_rgb(color))
    return red, green, blue
