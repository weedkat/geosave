"""The explorer's map: its layers over a basemap, the viewport and map clicks."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

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


def scope_hover(plot, element):
    """Keep dynamically added glyphs out of unrelated attribute hover tools.

    HoloViews attaches new overlay renderers to its first hover tool. Only
    renderers carrying that tool's declared fields can answer those hovers.
    """
    if isinstance(element, hv.Overlay) and plot.dynamic:
        # HoloViews appends each new renderer to this overlay-wide hover handle
        # before hooks run. A TileRenderer fails Bokeh's validation at that
        # point. Keep the toolbar tools and bind their data renderers below.
        plot.handles.pop("hover", None)
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
        self.ranges.add_subscriber(self._navigate)
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
        self.pane.object = hv.DynamicMap(
            Callable(self._draw, memoize=False),
            streams=[self.size, self._refresh, *self._settings],
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
        self.redraw()

    def snapshot(self) -> hv.Overlay:
        """Freeze the current view without live streams or drawing tools."""
        plot = self._draw()
        # The invisible anchor carries the live viewport streams. Detach every
        # element so rendering the snapshot cannot subscribe to this map.
        snapshot = cast(
            "hv.Overlay",
            plot.map(lambda element: element.clone(link=False), hv.Element),
        )
        assert self.ranges.x_range is not None and self.ranges.y_range is not None
        left, right = self.ranges.x_range
        bottom, top = self.ranges.y_range
        # A standalone map has no Python resize callback to adjust its ranges.
        # Fix the frame's aspect to the captured extent instead.
        snapshot.opts(
            clone=False,
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
        return snapshot

    def _view(self):
        x_range, y_range = self.ranges.x_range, self.ranges.y_range
        if (
            x_range is None
            or y_range is None
            or not (is_known_range(x_range) and is_known_range(y_range))
        ):
            return None
        shape = DEFAULT_VIEW_SHAPE
        if self.size.width and self.size.height:
            shape = (self.size.height, self.size.width)
        return create_view(x_range, y_range, shape)

    def _navigate(self, **_):
        """Let Bokeh move existing glyphs until a visible layer needs new pixels."""
        explorer = self._explorer
        view = self._view()
        if view is None:
            return
        if any(
            not layer.covers_view(view) for layer in explorer.select_drawn_rasters()
        ):
            self.redraw()
        elif any(
            isinstance(layer, VectorLayer) and layer.visible and layer.is_sampled
            for layer in explorer.layers
        ):
            self.redraw()

    def _draw(self, **_: Any) -> hv.Overlay:
        # The anchor leads so its navigation streams keep the same subplot.
        explorer = self._explorer
        plot: Any = self._anchor
        view = self._view()
        for layer in explorer.select_drawn_rasters():
            plot = plot * layer.draw(view)
        for layer in explorer.layers:
            if isinstance(layer, VectorLayer) and layer.visible:
                plot = plot * layer.draw(view)
        tile = BASEMAPS[explorer.basemap]
        if tile is not None:
            # Background tiles have no data extent to contribute to the viewport.
            plot = plot * tile.opts(apply_ranges=False, clone=True, backend="bokeh")
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
        # HoloViews recomputes sizing_mode on refresh. Keep the live figure
        # stretched to its pane in both dimensions, including after PlotSize.
        figure.sizing_mode = self.pane.sizing_mode
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
        self.ranges.update(
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
        self.ranges.clear()
        self.tap.clear()
        self.size.clear()
        for stream in self._settings:
            stream.unwatch()
