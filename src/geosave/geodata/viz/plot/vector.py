"""Static vector figures: every geometry type of a frame, overlaid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import spatialpandas as spd
from matplotlib.colors import to_hex, to_rgba

from geosave.geodata.viz import shapes

if TYPE_CHECKING:
    import geopandas as gpd


# Width of lines in a figure, in points.
LINE_WIDTH = 2

# Element drawing each geometry type.
ELEMENTS: dict[str, type[hv.Element]] = {
    "Polygon": hv.Polygons,
    "LineString": hv.Path,
    "Point": hv.Points,
}


def draw_vector(frame: gpd.GeoDataFrame, *, color: str, alpha: float) -> hv.Overlay:
    """Draw every geometry of a frame in its own coordinates.

    Polygons, lines and points of one frame overlay each other.

    Args:
        frame: Frame with an active geometry column.
        color: Colour of outlines, lines and points; polygons fill with it at
            `alpha`.
        alpha: Opacity of the polygon fill.

    Returns:
        Overlay of one element per geometry type the frame holds.

    Raises:
        ValueError: The frame holds no geometry, or a GeometryCollection.
    """
    color = to_hex(color)
    options_by_element: dict[type[hv.Element], dict[str, object]] = {
        hv.Polygons: {"facecolor": to_rgba(color, alpha), "edgecolor": color},
        hv.Path: {"color": color, "linewidth": LINE_WIDTH},
        hv.Points: {"color": color},
    }
    # One element per geometry type, overlaid
    elements = []
    for geometry_type, rows in shapes.split_by_geometry_type(frame).items():
        element = ELEMENTS[geometry_type]
        elements.append(
            element(spd.GeoDataFrame(rows)).opts(
                **options_by_element[element], backend="matplotlib"
            )
        )
    plot = hv.Overlay(elements)
    plot.opts(aspect="equal", backend="matplotlib")
    return plot
