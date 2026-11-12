"""The explorer's map: its layers over a basemap, the viewport and map clicks."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import geoviews as gv
import geoviews.plotting.bokeh  # noqa: F401 -- register the geographic Bokeh plots
import holoviews as hv
from holoviews.core.spaces import Callable
import panel as pn
import param
from bokeh.models import HoverTool, Image
from holoviews.streams import Params, PlotSize, RangeXY, SingleTap, Counter
from odc.geo.geom import BoundingBox, bbox_union
from panel.viewable import Viewer

from ..layers import VectorLayer
from ..mercator import (
    DEFAULT_VIEW_SHAPE,
    create_view,
    frame_bbox,
    is_known_range,
)
from .region import RegionTools

if TYPE_CHECKING:
    from ..explorer import Explorer

BASEMAPS = {
    "Light": gv.tile_sources.EsriWorldLightGrayBase,
    "Dark": gv.tile_sources.EsriWorldDarkGrayBase,
    "Imagery": gv.tile_sources.EsriImagery,
    "OpenStreetMap": gv.tile_sources.OSM,
    "None": None,
}


# Screen pixels a click may miss a feature by and still identify it.
CLICK_TOLERANCE = 5


def set_map_aspect(plot, _):
    """Keep Web Mercator metres square as the browser sizes the map."""
    plot.state.match_aspect = True


def scope_hover(plot, _):
    """Keep dynamically added glyphs out of unrelated attribute hover tools.

    HoloViews attaches new overlay renderers to its first hover tool. Only
    renderers carrying that tool's declared fields can answer those hovers.
    """
    for tool in plot.state.select({"type": HoverTool}):
        tool.point_policy = "follow_mouse"
        tool.line_policy = "none"
        tool.limit = 1
        tool.show_arrow = False
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


class Map(Viewer):
    """Draw the explorer's layers over its basemap, and keep the viewport they read.

    The view redraws when the viewport, the explorer's layers, their settings,
    the basemap or the comparison change. A click identifies the visible
    layers' values into `clicked_values`.
    """

    # Values at the last click, by layer name; None before any click.
    clicked_values = param.Dict(default=None, allow_None=True)

    def __init__(self, explorer: Explorer, height: int) -> None:
        """Frame every layer, using `height` before a browser size is available."""
        super().__init__()
        self._explorer = explorer
        self.height = height
        self._layer_watchers = {}
        self._hold = 0
        self._pending = False
        # Distinct events also invalidate HoloViews nested plot caches.
        self._refresh = Counter()
        self.region = RegionTools()
        # A framing requested by `zoom`, applied on the next redraw.
        self._extent = None
        # The last plot size in pixels, to keep the map's aspect on resize.
        self._shape = None
        bounds = [layer.bbox for layer in explorer.layers]
        for table in explorer.tables.tables.values():
            bounds.append(table.footprints.bbox)
        extent = frame_bbox(bbox_union(bounds))
        self._anchor = gv.Points(
            [(extent.left, extent.bottom), (extent.right, extent.top)],
            crs=gv.tile_sources.OSM.crs,
        ).opts(alpha=0, size=0, backend="bokeh")
        self.ranges = RangeXY(
            source=self._anchor,
            x_range=(extent.left, extent.right),
            y_range=(extent.bottom, extent.top),
        )
        self.tap = SingleTap(source=self._anchor)
        self.size = PlotSize(source=self._anchor)
        self.size.add_subscriber(self._resize)
        self.tap.add_subscriber(self._identify)
        # Transient, so each settings event triggers, even one equal to the last.
        self._settings = [
            Params(
                explorer,
                ["basemap"],
                transient=True,
            ),
            Params(
                explorer.comparison,
                ["enabled", "left", "right", "side"],
                transient=True,
            ),
        ]
        self.pane = pn.ui.HoloViews(backend="bokeh", sizing_mode="stretch_both")
        # Every event redraws: a repeated settings event, or a zoom to the
        # range already shown, must still run the hooks. Layers keep their own
        # caches, so a redraw reads no pixels it already has.
        self.pane.object = hv.DynamicMap(
            Callable(self._draw, memoize=False),
            streams=[self.ranges, self._refresh, *self._settings],
        )
        self._watcher = explorer.param.watch(self._follow_layers, "layers")
        self._follow_layers()

    def _follow_layers(self, *_):
        for layer in list(self._layer_watchers):
            if layer not in self._explorer.layers:
                layer.param.unwatch(self._layer_watchers.pop(layer))
        for layer in self._explorer.layers:
            if layer not in self._layer_watchers:
                self._layer_watchers[layer] = layer.param.watch(
                    self.redraw, list(layer.draw_params)
                )
        self.redraw()

    @contextmanager
    def defer_draw(self):
        """Draw once after a coordinated change to several layers."""
        self._hold += 1
        try:
            yield
        finally:
            self._hold -= 1
            if not self._hold and self._pending:
                self._pending = False
                self.redraw()

    def redraw(self, *_):
        """Request a map update without replacing its streams or pane."""
        if self._explorer.is_closed:
            return
        if self._hold:
            self._pending = True
        else:
            self._refresh.event()

    def __panel__(self):
        return self.pane

    def zoom(self, bbox: BoundingBox) -> None:
        """Frame `bbox`, in Web Mercator metres, keeping the map's aspect."""
        self._frame_extent(frame_bbox(bbox))

    def snapshot(self) -> hv.Overlay:
        """Freeze the current view without live streams or drawing tools."""
        plot = self._draw(**self.ranges.contents)
        # The invisible anchor carries the live viewport streams. Detach every
        # element so rendering the snapshot cannot subscribe to this map.
        snapshot = plot.map(lambda element: element.clone(link=False), hv.Element)
        left, right = self.ranges.x_range
        bottom, top = self.ranges.y_range
        # A standalone map has no Python resize callback to adjust its ranges.
        # Fix the frame's aspect to the captured extent instead.
        return snapshot.opts(
            responsive=False,
            width=None,
            height=None,
            frame_width=max(1, round(self.height * (right - left) / (top - bottom))),
            frame_height=self.height,
            hooks=[scope_hover, set_map_aspect],
            tools=["hover"],
            active_tools=["pan", "wheel_zoom"],
            backend="bokeh",
        )

    def _draw(self, x_range=None, y_range=None, **_: Any) -> hv.Overlay:
        # HoloViews attaches a stream to the plot at its source's position in
        # the overlay, so the anchor leads and keeps its position as layers
        # and basemaps come and go. HoloViews refuses a DynamicMap inside one,
        # so this is the map's only DynamicMap and each layer draws an element.
        explorer = self._explorer
        plot: Any = self._anchor
        # The grid the map shows: its ranges, at the plot's size once it has one
        view = None
        if (
            x_range is not None
            and y_range is not None
            and is_known_range(x_range)
            and is_known_range(y_range)
        ):
            shape = DEFAULT_VIEW_SHAPE
            if self.size.width and self.size.height:
                shape = (self.size.height, self.size.width)
            view = create_view(x_range, y_range, shape)
        for layer in explorer.select_drawn_rasters():
            plot = plot * layer.draw(view)
        for layer in explorer.layers:
            if isinstance(layer, VectorLayer) and layer.visible:
                plot = plot * layer.draw(view)
        tile = BASEMAPS[explorer.basemap]
        # Tiles render beneath every glyph wherever they sit in the overlay.
        if tile is not None:
            plot = plot * tile
        return plot.opts(
            projection=gv.tile_sources.OSM.crs,
            responsive=True,
            height=None,
            xlim=self.ranges.x_range,
            ylim=self.ranges.y_range,
            xaxis=None,
            yaxis=None,
            show_legend=False,
            active_tools=["pan", "wheel_zoom"],
            tools=["tap"],
            hooks=[scope_hover, self._apply_framing, self._set_up_figure],
            backend="bokeh",
        )

    def _set_up_figure(self, plot, _):
        # Runs on every redraw; only a new figure is set up, so later redraws
        # keep what was drawn and the toolbar choices made since.
        figure = plot.state
        # Bokeh keeps metres square by widening the ranges. HoloViews'
        # data_aspect instead resizes the figure, first to thousands of pixels
        # tall; tiles fetched for that never draw, so the page never idles.
        # HoloViews resets it on each redraw.
        set_map_aspect(plot, _)
        if self.region.has_tools(figure):
            return
        # Hover starts off; the toolbar's hover button turns it on.
        figure.toolbar.active_inspect = None
        self.region.add_tools(figure)

    def _resize(self, width=None, height=None, **_):
        if not width or not height or self._shape == (width, height):
            return
        assert self.ranges.x_range is not None
        assert self.ranges.y_range is not None
        left, right = self.ranges.x_range
        bottom, top = self.ranges.y_range
        if self._shape is not None:
            center = (left + right) / 2
            half_width = (top - bottom) * width / height / 2
            left, right = center - half_width, center + half_width
        self._shape = (width, height)
        self._frame_extent(BoundingBox(left, bottom, right, top, crs="EPSG:3857"))

    def _frame_extent(self, extent: BoundingBox) -> None:
        # Widen the extent to the plot's aspect, so it shows whole.
        width, height = self.size.width, self.size.height
        if width and height:
            ratio = width / height
            if extent.span_x / extent.span_y < ratio:
                extent = extent.buffered((extent.span_y * ratio - extent.span_x) / 2, 0)
            else:
                extent = extent.buffered(0, (extent.span_x / ratio - extent.span_y) / 2)
        self._extent = extent
        self.ranges.event(
            x_range=(extent.left, extent.right), y_range=(extent.bottom, extent.top)
        )

    def _apply_framing(self, plot, _):
        # RangeXY reports navigation; emitting it does not move a drawn map.
        # Use the public Bokeh ranges only for an explicit framing request.
        if self._extent is None:
            return
        extent = self._extent
        plot.state.x_range.update(start=extent.left, end=extent.right)
        plot.state.y_range.update(start=extent.bottom, end=extent.top)
        self._extent = None

    def _identify(self, x=None, y=None, **_):
        if x is None or y is None or self._explorer.is_closed:
            return
        x_range = self.ranges.x_range
        if x_range is None:
            return
        width = self.size.width or self.height
        tolerance = (x_range[1] - x_range[0]) / width * CLICK_TOLERANCE
        values = {}
        drawn = self._explorer.select_drawn_rasters()
        for label, layer in reversed(self._explorer.layers_by_label.items()):
            if layer in drawn or (isinstance(layer, VectorLayer) and layer.visible):
                found = layer.identify(x, y, tolerance)
                if found:
                    values[label] = found
        self.clicked_values = values

    def release(self) -> None:
        """Detach the click, resize and redraw callbacks."""
        self._explorer.param.unwatch(self._watcher)
        for layer, watcher in self._layer_watchers.items():
            layer.param.unwatch(watcher)
        self._layer_watchers.clear()
        self._refresh.clear()
        self.tap.clear()
        self.size.clear()
        for stream in self._settings:
            stream.unwatch()
