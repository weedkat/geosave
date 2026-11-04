"""A vector on the map: features coloured by a column, filtered by dates, picked by click."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, cast

import datashader as ds
from bokeh.models import HoverTool
from holoviews.operation.datashader import datashade, rasterize
from holoviews.core.util import dimension_sanitizer
import geoviews as gv
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from matplotlib.colors import to_hex
from odc.geo.geom import BoundingBox
from shapely.geometry import Point

from geosave.geodata.stac.catalog import ITEM_COLUMNS
from geosave.geodata.utils.datetime import find_overlapping, read_row_spans
from geosave.geodata.viz import shapes
from geosave.geodata.utils.color import CATEGORY_PALETTE
from geosave.geodata.viz.colors import DEFAULT_CMAP

from ..mercator import is_known_range, to_web_mercator
from .layer import Layer
from .raster import RasterLayer

# Colour of features no column colours, and of selected features.
FEATURE_COLOR = "#1f77b4"
SELECTION_COLOR = "#ffd400"

# Element drawing each geometry type.
ELEMENTS: dict[str, type[hv.Element]] = {
    "Polygon": gv.Polygons,
    "LineString": gv.Path,
    "Point": gv.Points,
}

# Features of one geometry type above which the map draws them as an
# aggregated image of the viewport rather than as individual glyphs.
SAMPLED_COUNTS = {"Point": 50_000, "LineString": 5_000, "Polygon": 5_000}


class VectorLayer(Layer):
    """One vector on the map: its features, coloured by a column or one colour.

    Row lists index `data`: `shown_rows` are the rows the table's filters
    show, `footprint_rows` those whose polygons the table's checkboxes draw,
    `picked_rows` those a map click picked, and `raster_rows` those whose
    rasters are on the map, offered when the rows carry STAC assets. Checking
    a raster row opens its raster lazily, off the page's thread when a page
    is running; a row already open keeps its raster.
    """

    heading = "Vectors"
    icon = "polyline"
    # Kind of row, in the layer list's summary.
    row_noun = "feature"
    # Columns hover leaves out.
    hidden_hover_columns: set[str] = set()
    # Name of a layer given none.
    default_name = "vector"

    draw_params = (
        *Layer.draw_params,
        "column",
        "color",
        "picked_rows",
        "shown_rows",
        "footprint_rows",
    )
    legend_params = ("column", "color")

    alpha = param.Number(0.3, bounds=(0.0, 1.0), label="Opacity")
    column = param.Selector(default=None, allow_None=True, label="Color by")
    color = param.Color(FEATURE_COLOR, label="Color")
    picked_rows = param.List(default=[], item_type=int)
    shown_rows = param.List(default=[], item_type=int)
    footprint_rows = param.List(default=[], item_type=int)
    raster_rows = param.List(default=[], item_type=int)
    # Rasters of the checked raster rows, by row position, in check order.
    rasters = param.Dict(default={}, precedence=-1)
    # Why the last checked raster row failed to open; empty when it opened.
    error = param.String(default="", precedence=-1)
    period = param.CalendarDateRange(default=None, label="Dates")

    def __init__(
        self,
        frame,
        *,
        name=None,
        column=None,
        color=FEATURE_COLOR,
        alpha=0.3,
    ):
        """Show a frame's features, coloured by `column` or one `color`.

        Args:
            frame: Features with a CRS.
            name: Name in the layer list. None uses the kind's default name.
            column: Number or text column colouring the features. None
                colours them `color`.
            color: Colour of every feature when no column colours them.
            alpha: Opacity of polygon fills, lines and points.

        Raises:
            ValueError: The frame has no CRS, holds no geometry, or holds a
                GeometryCollection.
            KeyError: `column` is no number or text column.
        """
        self._load_generation = 0
        # 1. The rows, in the map's Web Mercator, by geometry type
        if frame.crs is None:
            raise ValueError("a map needs a CRS; use set_crs first")
        self.data = frame.reset_index(drop=True)
        # Rows carrying STAC assets are catalog rows, whose rasters can open.
        self.has_rasters = True
        for key in ITEM_COLUMNS:
            if key not in self.data:
                self.has_rasters = False
        if self.has_rasters:
            self.heading = "Catalogs"
            self.icon = "layers"
            self.row_noun = "item"
            self.default_name = "catalog"
            self.hidden_hover_columns = {"id", "type", "stac_version"}
        # The map's extent of these features, in Web Mercator metres.
        self.bbox = to_web_mercator(
            BoundingBox(*self.data.total_bounds, crs=self.data.crs).polygon
        )
        self.frame = self.data.to_crs(3857)
        self.columns = self._list_shown_columns()
        # Hover shows dates as text, rather than through a backend-specific
        # epoch formatter; the table and source keep their native datetimes.
        for key in self.columns:
            if pd.api.types.is_datetime64_any_dtype(self.frame[key]):
                self.frame[key] = (
                    self.frame[key].dt.strftime("%Y-%m-%d %H:%M:%S %z").str.strip()
                )
        # Rows by geometry type, split once since filters only pick rows.
        self._groups = shapes.split_by_geometry_type(self.frame)

        # 2. The columns that can colour the features: numbers and text, not dates
        color_columns = []
        for shown_column in self.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.data[shown_column]):
                color_columns.append(shown_column)
        if column is not None and column not in color_columns:
            raise KeyError(f"{column!r} is not a scalar attribute")
        super().__init__(
            name=name or self.default_name,
            column=column,
            color=to_hex(color),
            alpha=alpha,
            footprint_rows=list(range(len(self.data))),
        )
        # "Single color" stands for no column, unless a column carries the name.
        solid = "Single color"
        while solid in color_columns:
            solid += " only"
        columns_by_label = {solid: None}
        for color_column in color_columns:
            columns_by_label[color_column] = color_column
        self.param.column.objects = columns_by_label

        # 3. Every row shown, and the date filter spanning the rows' dates
        self.shown_rows = list(range(len(self.data)))
        self._spans = self.get_dates()
        dates = pd.concat(self._spans).dropna()
        if not dates.empty:
            span = (dates.min().date(), dates.max().date())
            self.param.period.bounds = span
            self.period = span
        # The last drawn element, and the settings and view it was drawn for.
        self._drawn_element = None
        self._drawn_settings = None

    def _list_shown_columns(self) -> list[str]:
        """Return the columns a table and hover show: numbers, dates and text."""
        columns = []
        for key in self.data.columns:
            if key == self.data.geometry.name:
                continue
            values = self.data[key]
            # Python's all(), since an empty pyarrow timestamp column has no `all` reduction.
            holds_text = all(isinstance(value, str) for value in values.dropna())
            if (
                pd.api.types.is_numeric_dtype(values)
                or pd.api.types.is_datetime64_any_dtype(values)
                or holds_text
            ):
                columns.append(str(key))
        return columns

    @property
    def summary(self) -> str:
        """Return the row count, and open rasters of a catalog, such as `12 items · 2 open`."""
        plural = "" if len(self.data) == 1 else "s"
        summary = f"{len(self.data):,} {self.row_noun}{plural}"
        if self.has_rasters:
            summary += f" · {len(self.rasters)} open"
        return summary

    @property
    def swatches(self) -> dict[str, str] | None:
        """Return each category's colour, or the single colour when none colours."""
        if self.category_colors:
            swatches = {}
            for category, color in self.category_colors.items():
                swatches[str(category)] = color
            return swatches
        if self.column is None:
            return {"Features": self.color}
        return None

    @property
    def ramp(self) -> tuple[str, str, tuple[float, float] | None] | None:
        """Return the numeric column, colormap and value range colouring the features."""
        if self.column is None or self.category_colors:
            return None
        return str(self.column), DEFAULT_CMAP, self.get_limits(self.column)

    def _create_copy(self):
        return type(self)(self.data, name=self.name)

    def copy(self) -> VectorLayer:
        """Return a layer on the same rows, its open rasters copied with their settings."""
        copied = super().copy()
        # The copy shows rasters of its own, so their settings stay independent.
        rasters = {}
        for position, raster in self.rasters.items():
            rasters[position] = raster.copy()
        copied.rasters = rasters
        return copied

    def open_row_raster(self, position: int) -> RasterLayer:
        """Open one catalog row lazily, titled by its collection and date."""
        row = self.data.iloc[position]
        title = str(row.get("collection") or row.get("id", position))
        date = row.get("datetime", row.get("start_datetime"))
        if pd.notna(date) and hasattr(date, "strftime"):
            title += f" · {date:%Y-%m-%d}"
        return RasterLayer(row.gs.to_datatree().gs.to_dataset(), row=row, name=title)

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

    @property
    def is_sampled(self):
        """Return whether any geometry type draws as an image of the viewport."""
        for kind, group in self._groups.items():
            if len(group) > SAMPLED_COUNTS[kind]:
                return True
        return False

    @property
    def category_colors(self) -> dict[str, str] | None:
        """Return the colour of each category of the text column colouring the features.

        None when no column or a number colours them. The map and the legend
        both read this, so a category draws in the colour its legend entry
        shows. Categories take colours in sorted order, so one keeps its colour
        across geometry types and redraws.
        """
        column = self.column
        if column is None or pd.api.types.is_numeric_dtype(self.data[column]):
            return None
        categories = sorted(self.data[column].dropna().astype(str).unique())
        colors = {}
        for position, category in enumerate(categories):
            colors[category] = CATEGORY_PALETTE[position % len(CATEGORY_PALETTE)]
        return colors

    def get_limits(self, column):
        """Return a numeric attribute's finite range, None when it holds none."""
        values = self.data[column].replace([np.inf, -np.inf], np.nan)
        if values.isna().all():
            return None
        return float(values.min()), float(values.max())

    def get_dates(self):
        """Read each row's first and last instant, to whole days."""
        starts, ends = read_row_spans(self.data)
        return starts.dt.normalize(), ends.dt.normalize()

    def filter_dates(self, table, period):
        """Keep rows overlapping the chosen dates, including undated rows."""
        if period is None:
            return table
        starts, ends = self._spans
        mask = find_overlapping(starts, ends, period)
        return table.loc[pd.Series(mask, index=self.data.index).reindex(table.index)]

    def draw(self, view):
        """Return the shown features, drawn again only when a setting or a dense view changes."""
        x_range = y_range = None
        if view is not None:
            bounds = view.boundingbox
            x_range, y_range = (bounds.left, bounds.right), (bounds.bottom, bounds.top)
        # Glyphs draw the same at any viewport; only images of dense features resample.
        view = (x_range, y_range) if self.is_sampled else None
        settings = ({name: getattr(self, name) for name in self.draw_params}, view)
        if self._drawn_element is None or settings != self._drawn_settings:
            self._drawn_element = self.render(x_range, y_range)
            self._drawn_settings = settings
        return self._drawn_element

    def render(self, x_range=None, y_range=None):
        """Draw the shown features as GeoViews elements, the picked ones outlined."""
        shown_index = pd.Index(self.shown_rows).intersection(self.footprint_rows)
        if not is_known_range(x_range) or not is_known_range(y_range):
            x_range = y_range = None
        category_colors = self.category_colors
        elements = []

        # 1. One element per geometry type: glyphs, or an image of the view when dense
        for kind, group in self._groups.items():
            group = group.loc[group.index.intersection(shown_index)]
            if group.empty:
                continue
            if len(group) > SAMPLED_COUNTS[kind]:
                element = self._draw_dense_features(
                    kind, group, category_colors, x_range, y_range
                )
            else:
                element = self._draw_features(kind, group, category_colors)
            if element is not None:
                elements.append(element)

        # 2. Outline the picked features
        if self.visible:
            elements.extend(
                self._draw_picked_features(shown_index.intersection(self.picked_rows))
            )
        if not elements:
            return hv.Overlay([gv.Points([], crs=gv.tile_sources.OSM.crs)])
        return hv.Overlay(elements)

    def _create_hover(self) -> HoverTool:
        """Show up to four columns on hover, the colouring column first."""
        column = self.column
        hover_columns = []
        for name in self.columns:
            if name not in self.hidden_hover_columns and name != column:
                hover_columns.append(name)
        if column is not None:
            hover_columns.insert(0, column)
        tooltips = []
        for name in hover_columns[:4]:
            tooltips.append((name, f"@{{{dimension_sanitizer(name)}}}"))
        return HoverTool(tooltips=tooltips)

    def _draw_features(self, kind, group, category_colors) -> hv.Element:
        """Draw features of one geometry type as glyphs, coloured by `column` or `color`."""
        column = self.column
        element = ELEMENTS[kind](group, vdims=self.columns, crs=gv.tile_sources.OSM.crs)
        options: dict[str, Any] = {
            "tools": [self._create_hover()],
            "visible": self.visible,
        }

        # Colour: one colour, the categories' colours, or a numeric column's ramp
        if column is None:
            options["color"] = self.color
        else:
            options["color"] = hv.dim(column)
            options["cmap"] = category_colors or DEFAULT_CMAP
            limits = None if category_colors else self.get_limits(column)
            if limits is not None:
                options["clim"] = limits

        # Opacity: polygons fill at it and keep solid outlines
        if kind == "Polygon":
            options["fill_alpha"] = self.alpha
            options["line_alpha"] = 1
        else:
            options["alpha"] = self.alpha
            if kind == "Point":
                options["size"] = 7
        return cast("hv.Element", element.opts(**options, backend="bokeh"))

    def _draw_dense_features(
        self, kind, group, category_colors, x_range, y_range
    ) -> hv.Element | None:
        """Draw many features of one geometry type as an image of the view.

        Returns:
            Datashaded categories or single colour, or a rasterized numeric
            column; None when no feature has a category to colour.
        """
        column = self.column
        sampling = dict(
            dynamic=False, width=900, height=600, x_range=x_range, y_range=y_range
        )
        options: dict[str, Any] = {
            "visible": self.visible,
            "alpha": self.alpha,
            "tools": [],
        }
        if category_colors is not None:
            # Each category's count, in its colour
            group = group[group[column].notna()].copy()
            if group.empty:
                return None
            group[column] = group[column].astype("category")
            element = datashade(
                ELEMENTS[kind](group, vdims=self.columns, crs=gv.tile_sources.OSM.crs),
                aggregator=ds.count_cat(column),
                color_key=category_colors,
                **sampling,
            )
        elif column is None:
            # Where any feature lies, in one colour
            element = datashade(
                ELEMENTS[kind](group, vdims=self.columns, crs=gv.tile_sources.OSM.crs),
                aggregator=ds.count(),
                cmap=[self.color],
                min_alpha=255,
                **sampling,
            )
        else:
            # The column's mean per pixel, through the colormap; hover reads it
            element = rasterize(
                ELEMENTS[kind](group, vdims=self.columns, crs=gv.tile_sources.OSM.crs),
                aggregator=ds.mean(column),
                **sampling,
            )
            options["cmap"] = DEFAULT_CMAP
            limits = self.get_limits(column)
            if limits is not None:
                options["clim"] = limits
            options["tools"] = ["hover"]
        return cast("hv.Element", element.opts(**options, backend="bokeh"))

    def _draw_picked_features(self, picked_rows: pd.Index) -> list[hv.Element]:
        """Outline the picked features in the selection colour; a polygon by its boundary."""
        picked = self.frame.loc[picked_rows]
        picked = picked[~(picked.geometry.is_empty | picked.geometry.isna())]
        if picked.empty:
            return []
        elements = []
        for kind, group in shapes.split_by_geometry_type(picked).items():
            if kind == "Polygon":
                group = group.copy()
                group.geometry = group.boundary
                kind = "LineString"
            element = ELEMENTS[kind](
                group[[group.geometry.name]], crs=gv.tile_sources.OSM.crs
            )
            elements.append(
                element.opts(color=SELECTION_COLOR, line_width=3, backend="bokeh")
            )
        return elements

    def find_rows_at(self, x, y, tolerance):
        """Return the shown rows within `tolerance` of a map point."""
        positions = self.frame.sindex.query(
            Point(x, y).buffer(tolerance), predicate="intersects"
        )
        drawn_rows = set(self.shown_rows) & set(self.footprint_rows)
        return sorted(set(int(row) for row in positions) & drawn_rows)

    def identify(self, x, y, tolerance):
        """Identify nearby visible rows, and pick them."""
        rows = self.find_rows_at(x, y, tolerance)
        self.picked_rows = rows
        if not rows:
            return {}
        return {name: str(self.data.iloc[rows[0]][name]) for name in self.columns}

    def get_info(self):
        """Return feature count, CRS and source metadata."""
        return {
            "Rows": len(self.data),
            "CRS": str(self.data.crs),
            "Geometry": self.data.geom_type.value_counts().to_dict(),
            "Attributes": dict(self.data.attrs),
        }
