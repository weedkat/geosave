"""The explorer's attribute tables: one tab per vector layer, below the map."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import panel as pn
import param
from panel.viewable import Viewer

from ..layers import VectorLayer

if TYPE_CHECKING:
    from ..explorer import Explorer


class AttributeTable(pn.ui.Tabulator):
    """A layer's table, owning its row synchronization and subscriptions."""

    def __init__(self, layer: VectorLayer) -> None:
        self._layer = layer
        self._released = False
        frame = pd.DataFrame(layer.data[layer.columns])
        self._raster_column = "raster"
        while self._raster_column in frame.columns:
            self._raster_column += "_"
        formatters = {}
        titles = {}
        if layer.has_rasters:
            frame.insert(0, self._raster_column, frame.index.isin(layer.raster_rows))
            formatters[self._raster_column] = {"type": "tickCross"}
            titles[self._raster_column] = "Raster"
        super().__init__(
            frame,
            selectable="checkbox",
            selection=list(layer.footprint_rows),
            formatters=formatters,
            titles=titles,
            header_tooltips={
                self._raster_column: "Show or hide this row's raster. The left checkbox controls its footprint."
            }
            if layer.has_rasters
            else {},
            disabled=True,
            show_index=False,
            header_filters=True,
            pagination="local",
            page_size=8,
            row_height=30,
            height=280,
            layout="fit_data_stretch",
            sizing_mode="stretch_width",
        )
        # Copy current state, rather than event.new: opening a raster can fail
        # and roll back its selection during the same event dispatch.
        self._subscriptions = [
            self.param.watch(self._select_footprints, "selection"),
            layer.param.watch(self._tick_footprints, "footprint_rows"),
            self.param.watch(self._show_filtered_rows, ["filters", "value"]),
            layer.param.watch(self._filter_period, "period"),
        ]
        if layer.has_rasters:
            self.on_click(self._toggle_raster, column=self._raster_column)
            self._subscriptions.append(
                layer.param.watch(self._tick_rasters, "raster_rows")
            )
        # A plain filter keeps the period subscription here. Panel's bound
        # filters install a watcher that remove_filter does not detach.
        self.add_filter(self._filter_dates)
        self._show_filtered_rows()

    def _select_footprints(self, *_):
        self._layer.footprint_rows = list(self.selection)

    def _tick_footprints(self, *_):
        self.selection = list(self._layer.footprint_rows)

    def _toggle_raster(self, event):
        if self._released:
            return
        rows = list(self._layer.raster_rows)
        if event.row in rows:
            rows.remove(event.row)
        else:
            rows.append(int(event.row))
        self._layer.raster_rows = rows

    def _tick_rasters(self, *_):
        ticked = self.value.index.isin(self._layer.raster_rows)
        if not (self.value[self._raster_column].to_numpy(dtype=bool) == ticked).all():
            value = self.value.copy()
            value[self._raster_column] = ticked
            self.value = value

    def _filter_dates(self, frame):
        return self._layer.filter_dates(frame, self._layer.period)

    def _filter_period(self, *_):
        self.param.trigger("value")

    def _show_filtered_rows(self, *_):
        self._layer.shown_rows = list(self.current_view.index)

    def release(self) -> None:
        """Detach this table from its layer; later clicks have no effect."""
        self._released = True
        for watcher in self._subscriptions:
            watcher.inst.param.unwatch(watcher)
        self._subscriptions.clear()


class AttributeTables(Viewer):
    """Show each vector layer's table in a tab, the last selected one's tab open.

    Choosing a tab selects its layer. Tables are built once per layer and
    kept, so a notebook keeps their models as tabs switch.
    """

    # The vector layer whose table is open: the last one selected.
    open_layer = param.ClassSelector(class_=VectorLayer, allow_None=True)

    def __init__(self, explorer: Explorer) -> None:
        """Show the vector layers of `explorer`."""
        super().__init__()
        self._explorer = explorer
        # Each vector layer's table, and its tab's content, built when the layer
        # is first shown and kept.
        self.tables: dict[VectorLayer, AttributeTable] = {}
        self._tab_contents: dict[VectorLayer, pn.ui.Column] = {}
        # One tab per vector layer, in layer order.
        self._tabs = pn.ui.Tabs(dynamic=False, sizing_mode="stretch_width", margin=0)
        self._tab_watcher = self._tabs.param.watch(self._select_tab_layer, "active")
        self._watchers = [
            explorer.param.watch(self._follow_layers, "layers"),
            explorer.param.watch(self._follow_selection, "active_layer"),
        ]
        self._follow_layers()
        self._card = self._create_card()

    @property
    def table(self) -> pn.ui.Tabulator | None:
        """Return the open table."""
        return self.tables.get(self.open_layer)

    def _list_vectors(self, *_: Any) -> list[VectorLayer]:
        vectors = []
        for layer in self._explorer.layers:
            if isinstance(layer, VectorLayer):
                vectors.append(layer)
        return vectors

    def _follow_layers(self, *_: Any) -> None:
        """Reconcile table membership, preserving surviving models and filters."""
        # 1. Build each new vector layer's table, and its tab
        vectors = self._list_vectors()
        for layer in list(self.tables):
            if layer not in vectors:
                self.tables.pop(layer).release()
                self._tab_contents.pop(layer)
        for layer in vectors:
            if layer not in self.tables:
                self.tables[layer] = AttributeTable(layer)
                self._tab_contents[layer] = self._create_tab_content(layer)

        # 3. Retab only when the vector layers change, so tables keep their models
        contents = []
        for layer in vectors:
            contents.append(self._tab_contents[layer])
        if contents != list(self._tabs.objects):
            tabs = []
            for layer in vectors:
                tabs.append((layer.name, self._tab_contents[layer]))
            self._tabs[:] = tabs
            # A single table needs no tab bar.
            self._tabs.sx = {
                ".MuiTabs-root": {"display": "none" if len(tabs) == 1 else "flex"}
            }
        self._follow_selection()

    def _follow_selection(self, *_: Any) -> None:
        """Open the selected vector's table without rebuilding its contents."""
        vectors = self._list_vectors()
        if self._explorer.active_layer in vectors:
            self.open_layer = self._explorer.active_layer
        elif self.open_layer not in vectors:
            self.open_layer = vectors[0] if vectors else None
        if self.open_layer is not None:
            self._tabs.active = vectors.index(self.open_layer)

    def __panel__(self):
        return self._card

    def _create_card(self):
        return pn.ui.Card(
            objects=[self._tabs],
            title=self.view_title,
            visible=pn.bind(lambda layer: layer is not None, self.param.open_layer),
            collapsed=False,
            collapsible=True,
            sizing_mode="stretch_width",
            sx={
                "boxShadow": "none",
                "borderTop": 1,
                "borderColor": "divider",
                "borderRadius": 0,
            },
        )

    def _create_tab_content(self, layer: VectorLayer) -> pn.ui.Column:
        # The layer's date filter, when it has dates, above its table.
        controls = []
        if layer.period is not None:
            controls.append(
                pn.ui.DateRangePicker.from_param(
                    layer.param.period, size="small", width=300
                )
            )
        if layer.has_rasters:
            controls.append(
                pn.ui.Typography(
                    "Left checkbox: footprint · Raster: image. If loading fails, check Raster again to retry.",
                    variant="caption",
                    color="text.secondary",
                )
            )
        controls.append(self.tables[layer])
        return pn.ui.Column(*controls, sizing_mode="stretch_width")

    @param.depends("open_layer")
    def view_title(self) -> str:
        """Return the open table's title."""
        layer = self.open_layer
        if layer is None:
            return ""
        noun = "Catalog" if layer.heading == "Catalogs" else "Attributes"
        return f"{noun} · {layer.name}"

    def _select_tab_layer(self, event):
        vectors = self._list_vectors()
        if event.new is not None and 0 <= event.new < len(vectors):
            layer = vectors[event.new]
            if self._explorer.active_layer is not layer:
                self._explorer.active_layer = layer

    def release(self) -> None:
        """Stop following the explorer's layers and selection."""
        for watcher in self._watchers:
            watcher.inst.param.unwatch(watcher)
        self._watchers.clear()
        self._tabs.param.unwatch(self._tab_watcher)
        for table in self.tables.values():
            table.release()
        self.tables.clear()
        self._tab_contents.clear()
