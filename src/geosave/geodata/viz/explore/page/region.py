"""Box and polygon drawing tools on the map, and the shapes they draw."""

from __future__ import annotations

from typing import Any

import shapely
from bokeh.models import BoxEditTool, ColumnDataSource, PolyDrawTool
from bokeh.plotting import figure as Figure


# Colour of the regions drawn on the map.
REGION_COLOR = "#e8590c"


class RegionTools:
    """Keep the box and polygon sources of each figure the tools are added to.

    Bokeh's own drawing tools and sources sit outside the overlay HoloViews
    redraws, so what was drawn stays as layers come and go.
    """

    def __init__(self) -> None:
        """Start with no figure drawn on."""
        # Box and polygon sources of each figure, one pair per figure.
        self.sources_per_figure: list[tuple[ColumnDataSource, ColumnDataSource]] = []

    def has_tools(self, figure: Figure) -> bool:
        """Return whether the drawing tools were already added to `figure`."""
        for tool in figure.tools:
            if isinstance(tool, BoxEditTool):
                return True
        return False

    def add_tools(self, figure: Any) -> None:
        """Add a box and a polygon drawing tool to `figure`, with their sources.

        `figure` is a Bokeh figure; its glyph methods return renderers whose
        types Bokeh's own tool annotations do not accept.
        """
        boxes = ColumnDataSource({"left": [], "bottom": [], "right": [], "top": []})
        polygons = ColumnDataSource({"xs": [], "ys": []})
        style = {
            "fill_color": REGION_COLOR,
            "fill_alpha": 0.1,
            "line_color": REGION_COLOR,
            "line_width": 2,
            "level": "overlay",
        }
        box_glyphs = figure.quad(
            left="left",
            bottom="bottom",
            right="right",
            top="top",
            source=boxes,
            **style,
        )
        polygon_glyphs = figure.patches("xs", "ys", source=polygons, **style)
        vertices = figure.scatter([], [], size=7, color=REGION_COLOR, level="overlay")
        figure.add_tools(
            BoxEditTool(
                renderers=[box_glyphs],
                description="Draw a region box: shift-drag; backspace deletes",
            ),
            PolyDrawTool(
                renderers=[polygon_glyphs],
                vertex_renderer=vertices,
                description=(
                    "Draw a region polygon: hold to start, click to add corners, "
                    "hold to finish"
                ),
            ),
        )
        self.sources_per_figure.append((boxes, polygons))

    def collect_shapes(self) -> list[shapely.Geometry]:
        """Return every drawn box and polygon, in Web Mercator metres."""
        shapes = []
        for boxes, polygons in self.sources_per_figure:
            corners = zip(
                boxes.data["left"],
                boxes.data["bottom"],
                boxes.data["right"],
                boxes.data["top"],
                strict=True,
            )
            for left, bottom, right, top in corners:
                shapes.append(shapely.box(left, bottom, right, top))
            outlines = zip(polygons.data["xs"], polygons.data["ys"], strict=True)
            for xs, ys in outlines:
                # A polygon still being drawn can hold fewer than three corners.
                if len(xs) >= 3:
                    shapes.append(shapely.Polygon(list(zip(xs, ys, strict=True))))
        return shapes
