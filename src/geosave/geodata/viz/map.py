"""Native GeoViews basemaps and map extents."""

import re

import geoviews as gv
from bokeh.models import HoverTool, Image
from odc.geo.geom import BoundingBox

BASEMAPS = {
    "Light": gv.tile_sources.EsriWorldLightGrayBase,
    "Dark": gv.tile_sources.EsriWorldDarkGrayBase,
    "Imagery": gv.tile_sources.EsriImagery,
    "OpenStreetMap": gv.tile_sources.OSM,
    "None": None,
}


def scope_hover(plot, _):
    """Keep dynamically added glyphs out of unrelated attribute hover tools.

    HoloViews attaches new overlay renderers to its first hover tool. Only
    renderers carrying that tool's declared fields can answer those hovers.
    """
    for tool in plot.state.select({"type": HoverTool}):
        fields = set()
        for _, expression in tool.tooltips or []:
            for match in re.finditer(r"@(?:\{([^}]+)\}|(\w+))", expression):
                fields.add(match.group(1) or match.group(2))
        if not fields:
            continue
        tool.renderers = [
            renderer
            for renderer in plot.state.renderers
            if hasattr(renderer, "data_source")
            and fields <= set(renderer.data_source.data)
            and (fields != {"image"} or isinstance(renderer.glyph, Image))
        ]


def frame_bbox(bbox: BoundingBox) -> BoundingBox:
    """Frame a projected extent with a margin, including isolated points."""
    box = bbox.to_crs("EPSG:3857")
    margin = max(box.span_x, box.span_y) * 0.05 or 1000
    return box.buffered(margin, margin)
