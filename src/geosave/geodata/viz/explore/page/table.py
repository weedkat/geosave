"""The explorer's attribute tables: one tab per vector layer, below the map."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING

from geosave.geodata.stac.catalog import FOOTER_KEY, ITEM_COLUMNS

import pandas as pd
import panel as pn
import param
from panel.viewable import Viewer

from ..layers import VectorLayer, RasterLayer

if TYPE_CHECKING:
    from ..explorer import Explorer


class AttributeTable(pn.ui.Tabulator):
    """Browse records and own their selected footprints and opened rasters."""

    _rename = {
        **pn.ui.Tabulator._rename,
        "raster_rows": None,
        "rasters": None,
        "error": None,
    }

    raster_rows = param.List(default=[], item_type=int)
    rasters = param.Dict(default={})
    error = param.String(default="")

    def __init__(self, frame, *, name: str, explorer: Explorer) -> None:
        self._explorer = explorer
        self._load_generation = 0
        self._released = False
        frame.gs.validate(crs=True)
        self.has_rasters = FOOTER_KEY in frame.attrs or all(
            column in frame for column in ITEM_COLUMNS
        )
        if self.has_rasters:
            frame.gs.validate(stac_item=True)

        layer = VectorLayer(frame, name=name)
        layer.footprint_rows = []
        self.footprints = layer
        self.data = layer.data
        frame = pd.DataFrame(layer.data[layer.columns])
        self._raster_column = "raster"
        while self._raster_column in frame.columns:
            self._raster_column += "_"
        formatters = {}
        titles = {}
        if self.has_rasters:
            frame.insert(0, self._raster_column, frame.index.isin(self.raster_rows))
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
            if self.has_rasters
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
            self.param.watch(self._explorer._show_layers, "rasters"),
            self.param.watch(self._explorer._show_error, "error"),
        ]
        if self.has_rasters:
            self.on_click(self._toggle_raster, column=self._raster_column)
            self._subscriptions.append(
                self.param.watch(self._tick_rasters, "raster_rows")
            )
        # A plain filter keeps the period subscription here. Panel's bound
        # filters install a watcher that remove_filter does not detach.
        self.add_filter(self._filter_dates)
        self._show_filtered_rows()

    def _select_footprints(self, *_):
        self.footprints.footprint_rows = list(self.selection)
        self._explorer._show_layers()

    def _tick_footprints(self, *_):
        self.selection = list(self.footprints.footprint_rows)

    def _toggle_raster(self, event):
        if self._released:
            return
        rows = list(self.raster_rows)
        if event.row in rows:
            rows.remove(event.row)
        else:
            rows.append(int(event.row))
        self.raster_rows = rows

    def _tick_rasters(self, *_):
        ticked = self.value.index.isin(self.raster_rows)
        if not (self.value[self._raster_column].to_numpy(dtype=bool) == ticked).all():
            value = self.value.copy()
            value[self._raster_column] = ticked
            self.value = value

    def _filter_dates(self, frame):
        return self.footprints.filter_dates(frame, self.footprints.period)

    def _filter_period(self, *_):
        self.param.trigger("value")

    def _show_filtered_rows(self, *_):
        self.footprints.shown_rows = list(self.current_view.index)

    def open_row_raster(self, position: int) -> RasterLayer:
        """Open one catalog row lazily, titled by its collection and date."""
        row = self.data.iloc[position]
        title = str(row.get("collection") or row.get("id", position))
        date = row.get("datetime", row.get("start_datetime"))
        if pd.notna(date) and hasattr(date, "strftime"):
            title += f" · {date:%Y-%m-%d}"
        tree = row.gs.to_datatree()
        # Viewport reads resample onto the screen anyway, so the row is shown on
        # its finest node grid to keep the most detail; reads stay lazy.
        nodes = tree.gs.datasets.values()
        finest = min(nodes, key=lambda node: abs(node.gs.resolution.x))
        scene = tree.gs.to_dataset(grid=finest.gs.geobox)
        return RasterLayer(scene, row=row, name=title)

    @param.depends("raster_rows", watch=True)
    def open_checked_rows(self) -> None:
        """Open the rasters of newly checked rows, and drop the unchecked rows'."""
        self._load_generation += 1
        generation = self._load_generation
        checked_rows = list(self.raster_rows)
        new_rows = []
        for position in checked_rows:
            if position not in self.rasters:
                new_rows.append(position)
        if not new_rows:
            self._keep_rasters(checked_rows, {})
            return
        # Metadata reads can wait on remote files; a running page stays live.
        pn.state.execute(
            partial(self._open_new_rows, checked_rows, new_rows, generation)
        )

    async def _open_new_rows(
        self, checked_rows: list[int], new_rows: list[int], generation: int
    ) -> None:
        if generation != self._load_generation:
            return
        try:
            rasters_by_row = await asyncio.to_thread(self._open_rows, new_rows)
        except (OSError, ValueError, KeyError, TypeError, NotImplementedError) as error:
            if generation != self._load_generation:
                return
            # Preserve already open rows when this request fails.
            self.error = str(error)
            self.raster_rows = sorted(self.rasters)
            return
        # A later check superseded this one while its rows opened.
        if generation != self._load_generation:
            return
        self.error = ""
        self._keep_rasters(checked_rows, rasters_by_row)

    def cancel_loads(self) -> None:
        """Invalidate pending reads and keep only successfully opened rows checked."""
        self._load_generation += 1
        self.raster_rows = list(self.rasters)

    def _open_rows(self, positions: list[int]) -> dict[int, RasterLayer]:
        rasters_by_row = {}
        for position in positions:
            rasters_by_row[position] = self.open_row_raster(position)
        return rasters_by_row

    def _keep_rasters(
        self, checked_rows: list[int], rasters_by_row: dict[int, RasterLayer]
    ) -> None:
        # The checked rows' rasters already open, then the newly opened ones.
        rasters = {}
        for position, raster in self.rasters.items():
            if position in checked_rows:
                rasters[position] = raster
        rasters.update(rasters_by_row)
        if rasters.keys() != self.rasters.keys():
            self.rasters = rasters

    def uncheck_row_of(self, raster: RasterLayer) -> None:
        """Uncheck the row that opened `raster`, closing its raster."""
        for position, opened_raster in self.rasters.items():
            if opened_raster is raster:
                remaining = []
                for row in self.raster_rows:
                    if row != position:
                        remaining.append(row)
                self.raster_rows = remaining
                return

    def release(self) -> None:
        """Detach this table from its layer; later clicks have no effect."""
        self._released = True
        self._load_generation += 1
        for watcher in self._subscriptions:
            watcher.inst.param.unwatch(watcher)
        self._subscriptions.clear()


class AttributeTables(Viewer):
    """Keep named tables and their selections independent of map membership."""

    def __init__(self, explorer: Explorer, frames) -> None:
        super().__init__()
        self.tables: dict[str, AttributeTable] = {}
        tabs = []
        for name, frame in frames.items():
            table = AttributeTable(frame, name=name, explorer=explorer)
            self.tables[name] = table
            controls = []
            if table.footprints.period is not None:
                controls.append(
                    pn.ui.DateRangePicker.from_param(
                        table.footprints.param.period, size="small", width=300
                    )
                )
            controls.append(table)
            tabs.append((name, pn.ui.Column(*controls)))

        self._tabs = pn.ui.Tabs(*tabs, dynamic=False, sizing_mode="stretch_width")
        self._card = pn.ui.Card(
            objects=[self._tabs],
            title="Tables",
            visible=bool(tabs),
            collapsed=False,
            sizing_mode="stretch_width",
        )

    @property
    def table(self) -> AttributeTable | None:
        """Return the active table without changing map selection."""
        if not self.tables:
            return None
        return list(self.tables.values())[self._tabs.active]

    def selected_layers(self) -> list:
        """Return selected footprints and opened rasters from every table."""
        layers = []
        for table in self.tables.values():
            if table.selection:
                layers.append(table.footprints)
            layers.extend(table.rasters.values())
        return layers

    def __panel__(self):
        return self._card

    def release(self) -> None:
        """Detach tables and invalidate their outstanding loads."""
        for table in self.tables.values():
            table.release()
        self.tables.clear()
