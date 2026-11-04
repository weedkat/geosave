"""What every map layer is: one raster or vector with its live display settings."""

from __future__ import annotations

import copy
from typing import Any, Self

import holoviews as hv
import param
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox


class Layer(param.Parameterized):
    """One raster or vector on the map, with its live display settings.

    A layer is the explorer's counterpart of a `.gs.plot` call: the same data
    and choices, kept as params. The explorer redraws the map when one
    changes. It holds no widgets; the page builds those from its params.
    """

    # Parameters consumed by map rendering, legends, and layer-list summaries.
    draw_params = ("visible", "alpha")
    legend_params = ()
    summary_params = ("name", "visible")

    visible = param.Boolean(True)
    alpha = param.Number(0.8, bounds=(0.0, 1.0), label="Opacity")

    # Layer list heading and Material icon of this kind of layer.
    heading: str
    icon: str

    # The map's extent of this layer, in Web Mercator metres.
    bbox: BoundingBox

    @property
    def summary(self) -> str:
        """Return the one line shown under this layer's name in the layer list."""
        raise NotImplementedError

    @property
    def caption(self) -> str | None:
        """Return the legend's text, when the legend is text alone."""
        return None

    @property
    def swatches(self) -> dict[str, str] | None:
        """Return the legend's colour per class or category, when it has classes."""
        return None

    @property
    def ramp(self) -> tuple[str, str, tuple[float, float] | None] | None:
        """Return the legend's title, colormap and value limits, when it is a ramp."""
        return None

    def draw(self, view: GeoBox | None) -> hv.Overlay:
        """Return this layer drawn in its current settings, for the map's view.

        `view` is the map's grid in Web Mercator; None draws the whole layer.
        """
        raise NotImplementedError

    def identify(self, x: float, y: float, tolerance: float) -> dict[str, str]:
        """Return this layer's values at a map click, in Web Mercator metres."""
        raise NotImplementedError

    def get_info(self) -> dict[str, Any]:
        """Return this layer's source metadata for the Info tab."""
        raise NotImplementedError

    def copy(self) -> Self:
        """Return a layer on the same data with independent display settings."""
        copied = self._create_copy()
        settings = {}
        for name, value in self.param.values().items():
            if name in copied.param and name != "name":
                settings[name] = copy.copy(value)
        copied.param.update(**settings)
        return copied

    def _create_copy(self) -> Self:
        """Return a layer on the same data, with default display settings."""
        raise NotImplementedError
