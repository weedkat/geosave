"""The explorer: the layers shown on one map, which is selected, and the basemap."""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

import geopandas as gpd

import holoviews as hv
import pandas as pd
import panel as pn
import param
import shapely
from odc.geo.geom import BoundingBox, Geometry, bbox_union
from panel.viewable import Viewer

from .layers import (
    Layer,
    RasterLayer,
    VectorLayer,
)
from .page import create_page
from .comparison import Comparison
from .page.map import BASEMAPS, Map
from .page.table import AttributeTables
from .page.sidebar import LayerDetails, LayerList
from .server import serve_explorer, stop_explorer


class Explorer(Viewer):
    """Explore native geodata on one map, with a sidebar and attribute tables.

    The explorer owns which layers are shown and selected, and the basemap;
    each layer owns its display settings, and each table owns row selection
    and raster loading. The page follows them. Display it in a notebook, or
    call `.serve()` to launch a web server.
    """

    # Layers added to the explorer, in order.
    added_layers = param.List(default=[], item_type=Layer, precedence=-1)
    # Layers on the map: each source, then the rasters a catalog opened.
    layers = param.List(default=[], item_type=Layer, precedence=-1)
    active_layer = param.ClassSelector(class_=Layer, allow_None=True, precedence=-1)
    basemap = param.Selector(default="Light", objects=list(BASEMAPS))
    # Step of the date slider, as pandas spells a period's frequency.
    freq = param.Selector(
        default="M", objects={"Year": "Y", "Month": "M", "Day": "D"}, label="Step"
    )
    # The period the map shows: each dated raster draws its last date in it.
    period = param.Selector(default=None, objects=[], allow_None=True, label="Date")
    # Periods offered by switched-on rasters; none hides the date controls.
    period_count = param.Integer(default=0, precedence=-1)

    def __init__(
        self,
        *,
        layers: Sequence[Layer] = (),
        tables: Mapping[str, gpd.GeoDataFrame] | None = None,
        basemap: str | None = None,
        height: int = 600,
    ):
        """Show configured layers and independently browsable feature tables.

        Args:
            layers: RasterLayer or VectorLayer objects displayed immediately.
            tables: Named GeoDataFrames. No rows are selected initially.
                Selecting rows displays their footprints; the Raster control
                opens supported Item assets separately.
            basemap: Basemap name; None uses Light.
            height: Inline page height in pixels. Served pages fill the browser.
        """
        if not layers and not tables:
            raise ValueError("an explorer needs layers or tables")
        if any(not isinstance(layer, Layer) for layer in layers):
            raise TypeError("layers must contain RasterLayer or VectorLayer objects")
        backend = hv.Store.current_backend
        # GeoViews registering a renderer does not initialize notebook resources.
        if not hv.extension._loaded:
            hv.extension("bokeh")
        pn.extension("tabulator", comms="ipywidgets")
        hv.renderer("bokeh")
        hv.Store.set_current_backend("bokeh")
        super().__init__(basemap=basemap or "Light", added_layers=list(layers))
        self.height = height
        self.server = None
        self.port: int | None = None
        self.is_closed = False
        # The served copies of this explorer, whose pages draw their own regions.
        self._sessions = []
        # Callbacks on each shown layer, removed when it is no longer shown.
        self._layer_watchers: dict[Layer, list[param.parameterized.Watcher]] = {}
        self.layers = list(self.added_layers)

        # The parts of the page that keep state of their own.
        self.status = pn.ui.Alert("", alert_type="danger", visible=False)
        self.comparison = Comparison()
        self.tables = AttributeTables(self, tables or {})
        self.map = Map(self, height)
        self.details = LayerDetails(self)
        self.layer_list = LayerList(self)
        self._page = None

        self._watchers = [
            self.param.watch(self._show_layers, "added_layers"),
            self.param.watch(self._watch_layers, "layers"),
        ]
        self._watch_layers()
        self._show_period()
        self.active_layer = self.layers[0] if self.layers else None
        if backend in hv.Store.renderers:
            hv.Store.set_current_backend(backend)

    def __panel__(self):
        if self._page is None:
            self._page = create_page(self)
        return self._page

    def zoom(self, bbox: BoundingBox) -> None:
        """Frame a layer without changing its display settings."""
        self.map.zoom(bbox)

    def snapshot(self) -> hv.Overlay:
        """Capture the current map as a standalone HoloViews overlay.

        Captures visible layers, date, display settings and viewport. Reading
        uncached pixels computes them. Pan and zoom remain available after
        HTML export, but cannot load new pixels. Basemap tiles need network
        access; sidebar controls, tables and region drawings are not included.

        Returns:
            Detached map, usable after this Explorer is changed or closed.

        Raises:
            RuntimeError: This Explorer has been closed.

        Examples:
            >>> snapshot = viewer.snapshot()
            >>> hv.save(snapshot, "map.html", backend="bokeh", resources="inline")
        """
        if self.is_closed:
            raise RuntimeError("cannot capture a closed explorer")
        return self.map.snapshot()

    def fit_visible(self) -> None:
        """Frame the rasters and vectors currently drawn on the map."""
        bounds = [layer.bbox for layer in self.select_drawn_rasters()]
        for layer in self.layers:
            if isinstance(layer, VectorLayer) and layer.visible:
                bounds.append(layer.bbox)
        if bounds:
            self.zoom(bbox_union(bounds))

    def add(self, *layers: Layer) -> None:
        """Add configured layers to the map."""
        if any(not isinstance(layer, Layer) for layer in layers):
            raise TypeError("add expects Layer objects")
        self.added_layers = [*self.added_layers, *layers]

    def remove(self, layer: Layer) -> None:
        """Remove a layer; removing a catalog's raster unchecks its row."""
        if layer in self.added_layers:
            remaining = []
            for source in self.added_layers:
                if source is not layer:
                    remaining.append(source)
            self.added_layers = remaining
            return

        for table in self.tables.tables.values():
            if layer is table.footprints:
                table.selection = []
                return
            if isinstance(layer, RasterLayer) and layer in table.rasters.values():
                table.uncheck_row_of(layer)
                return

    def __mul__(self, other: Explorer | Layer) -> Explorer:
        """Compose independent layers and named tables into a new viewer."""
        if isinstance(other, Layer):
            copied = self.copy()
            copied.add(other.copy())
            return copied
        if not isinstance(other, Explorer):
            return NotImplemented

        overlap = self.tables.tables.keys() & other.tables.tables.keys()
        if overlap:
            raise ValueError(f"table names must be unique: {sorted(overlap)}")
        # Table composition needs the original records as well as selections.
        frames = {}
        for viewer in (self, other):
            for name, table in viewer.tables.tables.items():
                frames[name] = table.data
        layers = []
        for viewer in (self, other):
            for layer in viewer.added_layers:
                layers.append(layer.copy())
        combined = Explorer(
            layers=layers, tables=frames, basemap=self.basemap, height=self.height
        )
        self._copy_tables(combined)
        other._copy_tables(combined)
        return combined

    def copy(self) -> Explorer:
        """Copy presentation state without reopening any raster assets."""
        layers = [layer.copy() for layer in self.added_layers]
        frames = {}
        for name, table in self.tables.tables.items():
            frames[name] = table.data
        copied = Explorer(
            layers=layers, tables=frames, basemap=self.basemap, height=self.height
        )
        self._copy_tables(copied)
        copied.tables._tabs.active = self.tables._tabs.active
        return copied

    def _copy_tables(self, viewer: Explorer) -> None:
        for name, table in self.tables.tables.items():
            copied = viewer.tables.tables[name]
            copied.footprints.param.update(
                column=table.footprints.column,
                color=table.footprints.color,
                alpha=table.footprints.alpha,
                visible=table.footprints.visible,
                period=table.footprints.period,
            )
            copied.filters = list(table.filters)
            copied.sorters = list(table.sorters)
            copied.page_size = table.page_size
            copied.page = table.page
            copied.selection = list(table.selection)
            copied.rasters = {
                position: raster.copy() for position, raster in table.rasters.items()
            }
            copied.raster_rows = list(table.rasters)

    @property
    def layers_by_label(self) -> dict[str, Layer]:
        """Return the layers by a unique label: the name, numbered when repeated."""
        layers_by_label = {}
        for layer in self.layers:
            label = layer.name
            suffix = 2
            while label in layers_by_label:
                label = f"{layer.name} ({suffix})"
                suffix += 1
            layers_by_label[label] = layer
        return layers_by_label

    @property
    def table(self):
        """Return the table in the active tab."""
        return self.tables.table

    @property
    def region(self) -> Geometry | None:
        """Return the shapes drawn with the map's box and polygon tools.

        Returns:
            Union of the drawn shapes in EPSG:4326, or None when none is drawn.

        Examples:
            >>> region = explorer.region
            >>> shape = region.to_crs(catalog.crs).geom
            >>> catalog.loc[catalog.geometry.intersects(shape)]
            >>> scene.gs.crop(region)
        """
        shapes = self._collect_shapes()
        if not shapes:
            return None
        drawn = Geometry(shapely.union_all(shapes), crs="EPSG:3857")
        return drawn.to_crs("EPSG:4326")

    def _collect_shapes(self):
        shapes = self.map.region.collect_shapes()
        for session in self._sessions:
            shapes.extend(session._collect_shapes())
        return shapes

    def _list_shown_layers(self) -> list[Layer]:
        return [*self.added_layers, *self.tables.selected_layers()]

    def _show_layers(self, *_: Any) -> None:
        self.layers = self._list_shown_layers()

    def _watch_layers(self, *_: Any) -> None:
        """Follow the shown layers' changes, and fit the selection and comparison."""
        # 1. Stop following layers no longer shown
        for layer in list(self._layer_watchers):
            if layer not in self.layers:
                if isinstance(layer, RasterLayer):
                    layer.clear_cache()
                for watcher in self._layer_watchers.pop(layer):
                    layer.param.unwatch(watcher)

        # Follow membership and temporal state; components own display subscriptions.
        for layer in self.layers:
            if layer in self._layer_watchers:
                continue
            watchers = []
            if isinstance(layer, RasterLayer):
                watchers.append(layer.param.watch(self._offer_periods, "visible"))
            self._layer_watchers[layer] = watchers

        self._repair_selection()

    def _repair_selection(self):
        """Keep comparison choices and selection within the shown layers."""
        rasters_by_label = {}
        for label, layer in self.layers_by_label.items():
            if isinstance(layer, RasterLayer):
                rasters_by_label[label] = layer
        self.comparison.rasters = rasters_by_label
        if self.active_layer not in self.layers:
            self.active_layer = (
                self.layers[0] if self.layers else None if self.layers else None
            )

    @param.depends("layers", "freq", watch=True)
    def _offer_periods(self, *_) -> None:
        """Offer each period holding a date of a switched-on raster; keep the shown one."""
        times = []
        for layer in self.layers:
            if isinstance(layer, RasterLayer) and layer.visible:
                times.extend(layer.times)
        periods = list(
            pd.DatetimeIndex(times).to_period(self.freq).unique().sort_values()
        )
        self.period_count = len(periods)
        # A slider cannot hold no options, so the last periods stay while no
        # raster has dates; they then filter nothing.
        if not periods:
            return
        if periods != self.param.period.objects:
            self.param.period.objects = periods
        if self.period not in periods:
            self.period = periods[0]

    @param.depends("period", watch=True)
    def _show_period(self) -> None:
        """Show each dated raster's last date in the period."""
        if self.period is None:
            return
        # Initial parameters are assigned before the map is constructed.
        if not hasattr(self, "map"):
            return
        with self.map.defer_draw():
            for layer in self.layers:
                if isinstance(layer, RasterLayer):
                    time = layer.select_time(self.period)
                    if time is not None and time != layer.time:
                        layer.time = time
            self.map.redraw()

    def select_drawn_rasters(self) -> list[RasterLayer]:
        """Return the rasters the map draws, in order.

        A raster draws when visible, shown by a comparison when one runs, and
        either timeless or dated inside the period.
        """
        compared = self.comparison.shown_raster
        drawn = []
        for layer in self.layers:
            if not isinstance(layer, RasterLayer) or not layer.visible:
                continue
            if self.comparison.enabled and layer is not compared:
                continue
            if layer.times and (
                self.period is None or layer.select_time(self.period) is None
            ):
                continue
            drawn.append(layer)
        return drawn

    def _show_error(self, event) -> None:
        self.status.object = f"Could not open catalog row: {event.new}"
        self.status.visible = bool(event.new)

    def serve(
        self,
        title: str | None = None,
        port: int = 5006,
        address: str | None = "localhost",
        websocket_origin: str | list[str] | None = None,
        threaded: bool | None = None,
        verbose: bool = True,
        open: bool = True,
        location: bool = True,
        **kwargs: Any,
    ):
        """Serve the explorer from a script, notebook, or Google Colab.

        Notebooks run the server in a background thread, so the cell stays
        free. Each browser session shows its own copy of this explorer, so
        later changes to this one do not reach an open page. Serving again on
        a port an explorer of this kernel serves, as rerunning the cell does,
        closes that explorer and replaces its server. Scripts block until
        interrupted. Colab displays the server through its port-proxy iframe.

        Args:
            title: Browser title.
            port: Local server port, default 5006.
            address: Listening address, default localhost.
            websocket_origin: Additional allowed browser origins (host:port).
            threaded: Override automatic notebook threading.
            verbose: Print the server address.
            open: Open a browser tab, or display an iframe in Colab.
            location: Include Panel's browser location component.
            **kwargs: Additional Panel server options.

        Returns:
            Native Panel server or background thread, with a `.stop()` method.

        Raises:
            ValueError: `port` is outside 1-65535.
            OSError: Another program listens on `port`.

        Examples:
            >>> explorer = catalog.gs.explore()
            >>> server = explorer.serve()
            >>> explorer.stop()
        """
        return serve_explorer(
            self,
            title=title,
            port=port,
            address=address,
            websocket_origin=websocket_origin,
            threaded=threaded,
            verbose=verbose,
            open=open,
            location=location,
            **kwargs,
        )

    def _create_session(self) -> Explorer:
        # Panel shares one served object, and its streams and widgets, across
        # every session, so each browser session gets its own copy to close.
        session = self.copy()
        # Code reads `region` off this explorer, not off the page's copy.
        self._sessions.append(session)
        pn.state.on_session_destroyed(lambda _: self._end_session(session))
        return session

    def _end_session(self, session):
        if session in self._sessions:
            self._sessions.remove(session)
        session._release()

    def stop(self) -> None:
        """Stop the server `serve` started; nothing happens when none runs."""
        stop_explorer(self)

    def close(self):
        """Stop serving and release the map and table callbacks."""
        if self.is_closed:
            return
        self.stop()
        self._release()
        self.map.pane.object = None
        self.added_layers = []
        self.layers = []
        self.active_layer = None

    def _release(self):
        # A closed page has already destroyed its models, so this releases
        # callbacks only, never widgets or the pane.
        self.is_closed = True
        self.map.release()
        self.tables.release()
        self.details.release()
        self.layer_list.release()
        for watcher in self._watchers:
            watcher.inst.param.unwatch(watcher)
        self._watchers.clear()
        for layer, watchers in self._layer_watchers.items():
            if isinstance(layer, RasterLayer):
                layer.clear_cache()
            for watcher in watchers:
                layer.param.unwatch(watcher)
        self._layer_watchers.clear()
        for session in self._sessions:
            session._release()
        self._sessions.clear()
