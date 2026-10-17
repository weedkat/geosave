"""Vector styling, table filtering and feature identification."""

from __future__ import annotations

from typing import Any

import datashader as ds
from holoviews.operation.datashader import datashade, rasterize
import geoviews as gv
import holoviews as hv
import pandas as pd
import panel as pn
import param
from matplotlib.colors import to_hex
from odc.geo.geom import BoundingBox
from shapely.geometry import Point

from geosave.geodata.core import GeoCatalog, GeoSamples
from geosave.geodata.viz import shapes
from geosave.geodata.viz.style import (
    DEFAULT_CMAP,
    FEATURE_COLOR,
    MATERIAL_STYLESHEET,
    SELECTION_COLOR,
    get_category_colors,
)


class VectorLayer(param.Parameterized):
    """Keep a native frame, its display choices and its table's row identity."""

    visible = param.Boolean(True)
    alpha = param.Number(0.3, bounds=(0.0, 1.0), label="Opacity")
    column = param.Selector(default=None, allow_None=True, label="Color by")
    color = param.Color(FEATURE_COLOR, label="Color")
    selected = param.List(default=[], item_type=int)
    rows = param.List(default=[], item_type=int)
    period = param.CalendarDateRange(default=None, label="Dates")

    def __init__(
        self, frame, *, name=None, column=None, color=FEATURE_COLOR, alpha=0.3
    ):
        if frame.crs is None:
            raise ValueError("a map needs a CRS; use set_crs first")
        shapes.select_drawable(frame)
        self.data = frame.reset_index(drop=True)
        self.frame = self.data.to_crs(3857)
        self.is_catalog = isinstance(frame.gs, GeoCatalog | GeoSamples)
        self.columns = []
        for key in self.data.columns:
            if key == self.data.geometry.name:
                continue
            values = self.data[key]
            if (
                pd.api.types.is_numeric_dtype(values)
                or pd.api.types.is_datetime64_any_dtype(values)
                or values.dropna().map(lambda value: isinstance(value, str)).all()
            ):
                self.columns.append(str(key))
        color_columns = [
            name
            for name in self.columns
            if not pd.api.types.is_datetime64_any_dtype(self.data[name])
        ]
        if column is not None and column not in color_columns:
            raise KeyError(f"{column!r} is not a scalar attribute")
        super().__init__(
            name=name or ("catalog" if self.is_catalog else "vector"),
            column=column,
            color=to_hex(color),
            alpha=alpha,
        )
        solid = "Single color"
        while solid in color_columns:
            solid += " only"
        self.param.column.objects = {
            solid: None,
            **{name: name for name in color_columns},
        }
        self.rows = list(range(len(self.data)))
        starts, ends = self.get_dates()
        dates = pd.concat([starts, ends]).dropna()
        if not dates.empty:
            self.param.period.bounds = (dates.min(), dates.max())
            self.period = (dates.min(), dates.max())
        self._table = None
        self._plot = None
        self._watchers = []

    @property
    def bbox(self):
        """Return the features' extent in map metres."""
        return BoundingBox(*self.frame.total_bounds, crs="EPSG:3857")

    @property
    def category_colors(self):
        """Return stable colors for text attributes."""
        if self.column is None or pd.api.types.is_numeric_dtype(self.data[self.column]):
            return None
        return get_category_colors(self.data[self.column].astype("string"))

    @property
    def table(self):
        """Build one checkbox table whose selection uses original row positions."""
        if self._table is None:
            self._table = pn.widgets.Tabulator(
                pd.DataFrame(self.data[self.columns]),
                selectable="checkbox",
                selection=self.selected,
                disabled=True,
                show_index=False,
                header_filters=True,
                pagination="local",
                page_size=8,
                row_height=30,
                height=280,
                layout="fit_data_stretch",
                sizing_mode="stretch_width",
                theme="materialize",
                stylesheets=[MATERIAL_STYLESHEET],
            )
            self._watchers.append(
                self._table.param.watch(
                    lambda event: setattr(self, "selected", list(event.new)),
                    "selection",
                )
            )
            self._watchers.append(
                self.param.watch(
                    lambda event: setattr(self._table, "selection", list(event.new)),
                    "selected",
                )
            )
            self._table.add_filter(pn.bind(self.filter_dates, period=self.param.period))
            self._watchers.append(
                self._table.param.watch(self._filter, ["filters", "value"])
            )
            self._watchers.append(self.param.watch(self._filter, "period"))
            self._filter()
        return self._table

    def get_dates(self):
        """Read instants or interval bounds as calendar dates."""
        missing = pd.Series(pd.NaT, index=self.data.index)
        instant = pd.to_datetime(self.data.get("datetime", missing), utc=True)
        starts = pd.to_datetime(self.data.get("start_datetime", missing), utc=True)
        ends = pd.to_datetime(self.data.get("end_datetime", missing), utc=True)
        return instant.fillna(starts).dt.date, instant.fillna(ends).dt.date

    def filter_dates(self, table, period):
        """Keep rows overlapping the chosen dates, including undated rows."""
        if period is None:
            return table
        starts, ends = self.get_dates()
        mask = (starts.isna() | (starts <= period[1])) & (
            ends.isna() | (ends >= period[0])
        )
        return table.loc[mask.reindex(table.index)]

    def _filter(self, *_):
        assert self._table is not None
        self.rows = list(self._table.current_view.index)

    def render(
        self, column, color, alpha, visible, rows, selected, x_range=None, y_range=None
    ):
        """Draw the filtered geometry as native GeoViews elements."""
        frame = self.frame.iloc[rows]
        frame = frame[~(frame.geometry.is_empty | frame.geometry.isna())]
        if frame.empty:
            return hv.Overlay([gv.Points([], crs=gv.tile_sources.OSM.crs)])
        elements = []
        constructors = {
            "Polygon": gv.Polygons,
            "LineString": gv.Path,
            "Point": gv.Points,
        }
        for kind, group in shapes.group_by_type(frame).items():
            # Keep dates readable without relying on a backend-specific epoch
            # formatter; the table and source retain their native datetimes.
            for name in self.columns:
                if pd.api.types.is_datetime64_any_dtype(group[name]):
                    group = group.copy()
                    group[name] = (
                        group[name].dt.strftime("%Y-%m-%d %H:%M:%S %z").str.strip()
                    )
            element = constructors[kind](
                group, vdims=self.columns, crs=gv.tile_sources.OSM.crs
            )
            options: dict[str, Any] = {"tools": ["hover"], "visible": visible}
            if column is None:
                options["color"] = color
            else:
                options["color"] = hv.dim(column)
                options["cmap"] = self.category_colors or DEFAULT_CMAP
                if self.category_colors is None:
                    options["clim"] = (
                        float(self.data[column].min()),
                        float(self.data[column].max()),
                    )
            threshold = 50_000 if kind == "Point" else 5_000
            if len(group) > threshold:
                sampling = dict(
                    dynamic=False,
                    width=900,
                    height=600,
                    x_range=x_range,
                    y_range=y_range,
                )
                categories = self.category_colors
                if categories is not None:
                    group = group[group[column].notna()].copy()
                    if group.empty:
                        continue
                    group[column] = group[column].astype("category")
                    element = constructors[kind](
                        group, vdims=self.columns, crs=gv.tile_sources.OSM.crs
                    )
                    element = datashade(
                        element,
                        aggregator=ds.count_cat(column),
                        color_key=categories,
                        **sampling,
                    )
                    options.pop("cmap", None)
                elif column is None:
                    element = datashade(
                        element,
                        aggregator=ds.count(),
                        cmap=[color],
                        min_alpha=255,
                        **sampling,
                    )
                else:
                    element = rasterize(element, aggregator=ds.mean(column), **sampling)
                options.pop("color", None)
                options["alpha"] = alpha
                options["tools"] = ["hover"] if column and categories is None else []
            elif kind == "Polygon":
                options["fill_alpha"] = alpha
                options["line_alpha"] = 1
            else:
                options["alpha"] = alpha
                if kind == "Point":
                    options["size"] = 7
            elements.append(element.opts(**options, backend="bokeh"))
        chosen = frame.loc[frame.index.intersection(selected)]
        if not chosen.empty and visible:
            for kind, group in shapes.group_by_type(chosen).items():
                if kind == "Polygon":
                    group = group.copy()
                    group.geometry = group.boundary
                    kind = "LineString"
                element = constructors[kind](
                    group[[group.geometry.name]], crs=gv.tile_sources.OSM.crs
                )
                elements.append(
                    element.opts(color=SELECTION_COLOR, line_width=3, backend="bokeh")
                )
        return hv.Overlay(elements)

    def draw(self, ranges=None):
        """Bind vector changes independently from raster pixel reads."""
        if self._plot is None:
            self._plot = hv.DynamicMap(
                pn.bind(
                    self.render,
                    column=self.param.column,
                    color=self.param.color,
                    alpha=self.param.alpha,
                    visible=self.param.visible,
                    rows=self.param.rows,
                    selected=self.param.selected,
                ),
                streams=[ranges] if ranges is not None else [],
            )
        return self._plot

    def identify(self, x, y, tolerance):
        """Identify nearby visible rows; catalog opening stays with checkboxes."""
        positions = self.frame.sindex.query(
            Point(x, y).buffer(tolerance), predicate="intersects"
        )
        rows = sorted(set(int(row) for row in positions) & set(self.rows))
        if not self.is_catalog:
            self.selected = rows
        if not rows:
            return {}
        return {name: str(self.data.iloc[rows[0]][name]) for name in self.columns}

    def get_info(self):
        """Return feature count, CRS and source metadata."""
        return {
            "Rows": len(self.data),
            "CRS": str(self.data.crs),
            "Geometry": shapes.summarize_geometry(self.data),
            "Attributes": dict(self.data.attrs),
        }

    def close(self):
        """Detach table subscriptions without constructing new widgets."""
        for watcher in self._watchers:
            watcher.inst.param.unwatch(watcher)
        self._watchers.clear()
