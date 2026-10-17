"""One native Panel layout for notebooks and locally served exploration."""

from __future__ import annotations

import copy
import html
import json
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import geopandas as gpd
import geoviews as gv
import geoviews.plotting.bokeh  # noqa: F401 -- register the geographic Bokeh plots
import holoviews as hv
from holoviews.streams import PlotSize, RangeXY, SingleTap, Stream
import pandas as pd
import panel as pn
import panel_material_ui as mui
from panel.io.notebook import load_notebook
import param
import xarray as xr
from odc.geo.geom import bbox_union

from geosave.geodata.core import GeoCatalog
from geosave.geodata.viz.map import BASEMAPS, frame_bbox, scope_hover
from geosave.geodata.viz.style import EXPLORER_THEME, MATERIAL_STYLESHEET, get_ramp
from .raster import RasterLayer
from .vector import VectorLayer

type GeoData = xr.Dataset | xr.DataArray | gpd.GeoDataFrame
type Layer = RasterLayer | VectorLayer
type Overlay = (
    GeoData | Layer | Sequence[GeoData | Layer] | Mapping[str, GeoData | Layer]
)


def create_layer(data: GeoData | Layer, name: str | None = None, **style: Any) -> Layer:
    """Create display settings for native raster or vector data."""
    if isinstance(data, RasterLayer | VectorLayer):
        return data
    if isinstance(data, xr.Dataset | xr.DataArray):
        return RasterLayer(data, name=name, **style)
    if isinstance(data, gpd.GeoDataFrame):
        return VectorLayer(data, name=name, **style)
    raise TypeError(f"cannot explore {type(data).__name__}")


def create_layers(overlay: Overlay | None) -> list[Layer]:
    """Create the named or ordered overlays of an accessor call."""
    if overlay is None:
        return []
    if isinstance(overlay, Mapping):
        return [create_layer(data, name=str(name)) for name, data in overlay.items()]
    if isinstance(
        overlay,
        xr.Dataset | xr.DataArray | gpd.GeoDataFrame | RasterLayer | VectorLayer,
    ):
        return [create_layer(overlay)]
    return [create_layer(data) for data in overlay]


class Explorer(pn.viewable.Viewer):
    """Explore native geodata with a sidebar, map and checkbox catalog table.

    Raster settings belong to each raster. Checking catalog rows reads their
    metadata lazily and opens them; unchecking releases their display layers.
    Use `.servable()` with Panel, or display the explorer in a notebook.
    """

    layers = param.List(default=[])
    selected = param.Parameter(default=None)
    basemap = param.Selector(default="Light", objects=list(BASEMAPS))
    compare = param.Boolean(False, label="Compare rasters")
    left = param.Selector(default=None, allow_None=True, label="Left")
    right = param.Selector(default=None, allow_None=True, label="Right")
    side = param.Selector(default="left", objects=["left", "right"], label="Showing")

    def __init__(
        self, *layers: GeoData | Layer, basemap: str | None = None, height: int = 600
    ):
        if not layers:
            raise ValueError("an explorer needs a raster or vector")
        backend = hv.Store.current_backend
        pn.extension("tabulator", comms="ipywidgets")
        hv.renderer("bokeh")
        hv.Store.set_current_backend("bokeh")
        super().__init__(basemap=basemap or "Light")
        self.height = height
        self._watchers = []
        self._catalog_watchers = {}
        self._layer_watchers = {}
        self._opened = {}
        self._updating = False
        self._closed = False
        self._extent = None
        self._vector = None
        self.layers = [create_layer(value) for value in layers]
        extent = frame_bbox(bbox_union(layer.bbox for layer in self.layers))
        self._anchor = gv.Points(
            [(extent.left, extent.bottom), (extent.right, extent.top)],
            crs=gv.tile_sources.OSM.crs,
        ).opts(alpha=0, size=0, backend="bokeh")
        self._ranges = RangeXY(
            source=self._anchor,
            x_range=(extent.left, extent.right),
            y_range=(extent.bottom, extent.top),
        )
        self._tap = SingleTap(source=self._anchor)
        self._size = PlotSize(source=self._anchor)
        self._tap.add_subscriber(self._identify)
        self.map = mui.HoloViews(
            backend="bokeh", sizing_mode="stretch_width", height=height
        )
        self.status = mui.Alert("", alert_type="danger", visible=False)
        self._layout = mui.Page(
            title="GeoSave",
            height=height + 480,
            sidebar_width=300,
            sidebar_resizable=True,
            sizing_mode="stretch_width",
            theme_config=EXPLORER_THEME,
            sx={
                ".main-content": {"padding": 0},
                ".header": {
                    "backgroundColor": "var(--mui-palette-background-paper) !important",
                    "color": "text.primary",
                    "boxShadow": "none",
                    "borderBottom": 1,
                    "borderColor": "divider",
                },
                ".header .title": {
                    "fontSize": 16,
                    "fontWeight": 650,
                    "color": "var(--mui-palette-text-primary) !important",
                },
                ".sidebar": {"backgroundColor": "background.paper"},
            },
        )
        self._menus = {}
        self._groups = {}
        self._layer_list = mui.Column(
            sizing_mode="stretch_width", margin=0, max_height=280, scroll=True
        )
        for title in ("Catalogs", "Rasters", "Vectors"):
            menu = mui.MenuList(
                dense=True,
                sizing_mode="stretch_width",
                sx={
                    ".MuiListItemText-primary": {
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "display": "-webkit-box",
                        "WebkitLineClamp": 2,
                        "WebkitBoxOrient": "vertical",
                        "fontSize": 12,
                        "fontWeight": 550,
                        "lineHeight": 1.4,
                    },
                    ".MuiListItemText-secondary": {"fontSize": 11},
                    ".MuiListItemIcon-root": {"minWidth": 26},
                    ".MuiListItemButton-root": {"padding": "4px 8px"},
                    ".MuiIconButton-root": {
                        "color": "text.secondary",
                        "padding": "4px",
                    },
                    ".material-icons": {"fontSize": 18},
                },
            )
            menu.on_action("Visibility", self._toggle_item)
            menu.on_action("Zoom to layer", self._zoom_item)
            menu.on_action("Remove layer", self._remove_item)
            self._watchers.append(menu.param.watch(self._choose_item, "value"))
            self._menus[title] = menu
            group = mui.Column(
                mui.Typography(
                    title,
                    variant="overline",
                    color="text.secondary",
                    margin=(4, 12, 0, 12),
                ),
                menu,
                sizing_mode="stretch_width",
                margin=0,
            )
            self._groups[title] = group
            self._layer_list.append(group)
        self._controls = mui.Column(sizing_mode="stretch_width", margin=0)
        self._color_by = None
        self._legend = pn.pane.HTML(
            "",
            sizing_mode="stretch_width",
            margin=(8, 0),
            stylesheets=[MATERIAL_STYLESHEET],
        )
        self._info = pn.pane.HTML(
            "", sizing_mode="stretch_width", stylesheets=[MATERIAL_STYLESHEET]
        )
        self._identified = pn.pane.HTML(
            "<p>Click the map to inspect visible layers.</p>",
            sizing_mode="stretch_width",
            stylesheets=[MATERIAL_STYLESHEET],
        )
        self._table_layers = []
        self._attributes = mui.Tabs(
            dynamic=False, sizing_mode="stretch_width", margin=0
        )
        self._watchers.append(
            self._attributes.param.watch(self._select_table, "active")
        )
        self._catalog = mui.Card(
            objects=[self._attributes],
            title="Catalog / attributes",
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
        self._comparison = mui.Card(
            objects=[
                mui.Checkbox.from_param(self.param.compare),
                mui.Select.from_param(self.param.left, size="small"),
                mui.Select.from_param(self.param.right, size="small"),
                mui.RadioButtonGroup.from_param(self.param.side),
            ],
            title="Compare",
            collapsed=True,
            visible=False,
            collapsible=True,
            sizing_mode="stretch_width",
            sx={"boxShadow": "none", "borderRadius": 0},
        )
        sidebar = mui.Column(
            mui.Select.from_param(
                self.param.basemap,
                name="Basemap",
                size="small",
                sizing_mode="stretch_width",
                margin=(12, 12, 8, 12),
            ),
            mui.Divider(),
            mui.Row(
                mui.Typography("Layers", variant="subtitle2", margin=(6, 0)),
                pn.Spacer(sizing_mode="stretch_width"),
                mui.Button(
                    label="Fit all",
                    icon="fullscreen",
                    size="small",
                    variant="text",
                    on_click=lambda _: self.zoom(
                        bbox_union(layer.bbox for layer in self.layers)
                    ),
                    disabled=pn.bind(lambda layers: not layers, self.param.layers),
                ),
                margin=(4, 12, 0, 12),
                sizing_mode="stretch_width",
            ),
            self._layer_list,
            mui.Divider(),
            self._comparison,
            mui.Tabs(
                (
                    "Style",
                    mui.Column(
                        self._controls,
                        self._legend,
                        margin=(12, 8),
                        sizing_mode="stretch_width",
                    ),
                ),
                ("Info", self._info),
                ("Identify", self._identified),
                sizing_mode="stretch_width",
            ),
            width=300,
            min_width=260,
            scroll=True,
            height=height + 350,
            margin=0,
        )
        main = mui.Column(
            self.status,
            self.map,
            self._catalog,
            sizing_mode="stretch_width",
            min_width=300,
            styles={"flex": "1 1 300px", "min-width": "0"},
            margin=0,
        )
        self._layout.sidebar = [sidebar]
        self._layout.main = [main]
        self._refresh = Stream.define("Refresh", tick=0)()
        self.map.object = hv.DynamicMap(
            self._draw, streams=[self._ranges, self._refresh]
        )
        self._watchers.extend(
            [
                self.param.watch(self._sync_layers, "layers"),
                self.param.watch(self._select, "selected"),
                self.param.watch(
                    self._compose, ["basemap", "compare", "left", "right", "side"]
                ),
            ]
        )
        self._sync_layers()
        self.selected = self.layers[0]
        if backend in hv.Store.renderers:
            hv.Store.set_current_backend(backend)

    def __panel__(self):
        return self._layout

    def _repr_mimebundle_(self, include=None, exclude=None):
        # Captured or cleared setup output can leave Python's extension flags
        # set while the notebook frontend has none of the core resources.
        load_notebook(inline=pn.config.inline, reloading=False)
        return super()._repr_mimebundle_(include=include, exclude=exclude)

    @property
    def table(self):
        """Return the selected vector's table or the last catalog table."""
        return self._vector.table if self._vector is not None else None

    def _choices(self):
        choices = {}
        for layer in self.layers:
            label = layer.name
            suffix = 2
            while label in choices:
                label = f"{layer.name} ({suffix})"
                suffix += 1
            choices[label] = layer
        return choices

    def _sync_layers(self, *_):
        for layer in list(self._layer_watchers):
            if layer not in self.layers:
                layer.param.unwatch(self._layer_watchers.pop(layer))
        for layer in list(self._catalog_watchers):
            if layer not in self.layers:
                layer.param.unwatch(self._catalog_watchers.pop(layer))
        for layer in self.layers:
            if layer not in self._layer_watchers:
                layer.draw(self._ranges)
                settings = (
                    [
                        "visible",
                        "alpha",
                        "display",
                        "cmap",
                        "stretch",
                        "clim",
                        "time",
                    ]
                    if isinstance(layer, RasterLayer)
                    else ["visible", "alpha", "column", "color", "selected", "rows"]
                )
                self._layer_watchers[layer] = layer.param.watch(self._compose, settings)
            if isinstance(layer, VectorLayer):
                layer.table
                if layer.is_catalog and layer not in self._catalog_watchers:
                    self._catalog_watchers[layer] = layer.param.watch(
                        partial(self._open, layer), "selected"
                    )
        choices = self._choices()
        rasters = [
            label for label, layer in choices.items() if isinstance(layer, RasterLayer)
        ]
        self.param.left.objects = self.param.right.objects = rasters
        self.left = (
            self.left if self.left in rasters else (rasters[0] if rasters else None)
        )
        self.right = (
            self.right if self.right in rasters else (rasters[-1] if rasters else None)
        )
        if len(rasters) < 2:
            self.compare = False
        self._comparison.visible = len(rasters) >= 2
        if self.selected not in self.layers:
            self.selected = self.layers[0] if self.layers else None
        if self._vector not in self.layers:
            self._vector = next(
                (layer for layer in self.layers if isinstance(layer, VectorLayer)), None
            )
        self._show_table()
        self._update_menus()
        self._compose()

    def _choose_item(self, event):
        layer = self._choices().get(event.new["label"]) if event.new else None
        if layer is not None and self.selected is not layer:
            self.selected = layer

    def _toggle_item(self, item):
        layer = self._choices()[item["label"]]
        layer.visible = not layer.visible

    def _zoom_item(self, item):
        self.zoom(self._choices()[item["label"]].bbox)

    def _remove_item(self, item):
        self.remove(self._choices()[item["label"]])

    def _update_menus(self):
        groups = {title: [] for title in self._menus}
        for label, layer in self._choices().items():
            if isinstance(layer, RasterLayer):
                title, icon = "Rasters", "grid_on"
                grid = layer.data.odc.geobox
                resolution = abs(grid.resolution.x)
                unit = "°" if grid.crs.geographic else f" {grid.crs.units[1]}"
                if unit == " metre":
                    unit = " m"
                bands = "band" if len(layer.names) == 1 else "bands"
                subtitle = f"{len(layer.names)} {bands} · {resolution:.4g}{unit}"
            else:
                title = "Catalogs" if layer.is_catalog else "Vectors"
                icon = "layers" if layer.is_catalog else "polyline"
                noun = "item" if layer.is_catalog else "feature"
                subtitle = (
                    f"{len(layer.data):,} {noun}{'' if len(layer.data) == 1 else 's'}"
                )
                if layer.is_catalog:
                    subtitle += f" · {len(layer.selected)} open"
            groups[title].append(
                {
                    "label": label,
                    "icon": icon,
                    "secondary": subtitle,
                    "tooltip": label,
                    "actions": [
                        {
                            "label": "Visibility",
                            "action": "Visibility",
                            "icon": "visibility" if layer.visible else "visibility_off",
                            "inline": True,
                        },
                        {
                            "label": "Zoom to layer",
                            "icon": "zoom_in_map",
                            "inline": True,
                        },
                        {"label": "Remove layer", "icon": "delete_outline"},
                    ],
                }
            )
        for title, menu in self._menus.items():
            items = groups[title]
            if menu.items != items:
                menu.items = items
            active = next(
                (
                    (index,)
                    for index, item in enumerate(items)
                    if self._choices()[item["label"]] is self.selected
                ),
                None,
            )
            if menu.active != active:
                menu.active = active
            self._groups[title].visible = bool(items)

    def _compose(self, *_):
        if self._closed:
            return
        if self._color_by is not None and isinstance(self.selected, VectorLayer):
            for label, column in self.selected.param.column.objects.items():
                if column == self.selected.column:
                    self._color_by.value = label
                    break
        self._update_menus()
        self._refresh.event(tick=self._refresh.tick + 1)

    def _draw(self, **_):
        plot: Any = self._anchor
        tile = BASEMAPS[self.basemap]
        if tile is not None:
            plot = tile * plot
        comparison = self._choices().get(
            self.left if self.side == "left" else self.right
        )
        for layer in self.layers:
            if (
                isinstance(layer, RasterLayer)
                and layer.visible
                and (not self.compare or layer is comparison)
            ):
                plot = plot * layer.draw(self._ranges)[()]
        for layer in self.layers:
            if isinstance(layer, VectorLayer) and layer.visible:
                plot = plot * layer.draw(self._ranges)[()]
        x_range, y_range = self._ranges.x_range, self._ranges.y_range
        return plot.opts(
            projection=gv.tile_sources.OSM.crs,
            responsive=True,
            min_height=self.height,
            max_height=self.height,
            data_aspect=1,
            aspect=None,
            xlim=x_range,
            ylim=y_range,
            xaxis=None,
            yaxis=None,
            show_legend=False,
            active_tools=["pan", "wheel_zoom"],
            tools=["tap"],
            hooks=[scope_hover, self._frame_map],
            backend="bokeh",
        )

    def _open(self, catalog, *_):
        if self._updating:
            return
        self._updating = True
        previous = self._opened.get(catalog, {})
        opened = {}
        try:
            for position in catalog.selected:
                row = catalog.data.iloc[position]
                if isinstance(catalog.data.gs, GeoCatalog):
                    groups = {None: None}
                else:
                    groups = row.gs.items
                for collection in groups:
                    key = (position, collection)
                    if key in previous:
                        opened[key] = previous[key]
                    else:
                        if collection is None:
                            data = row.gs.to_xarray()
                        else:
                            data = row.gs.to_xarray(collections=[collection])[
                                collection
                            ].to_dataset()
                        title = str(
                            row.get("collection")
                            or collection
                            or row.get("id", position)
                        )
                        date = row.get("datetime", row.get("start_datetime"))
                        if pd.notna(date) and hasattr(date, "strftime"):
                            title += f" · {date:%Y-%m-%d}"
                        opened[key] = RasterLayer(data, name=title)
            self._opened[catalog] = opened
            self.layers = [
                layer for layer in self.layers if layer not in previous.values()
            ] + list(opened.values())
            self.status.visible = False
        except (OSError, ValueError, KeyError, TypeError, NotImplementedError) as error:
            catalog.selected = sorted({key[0] for key in previous})
            self.status.object = f"Could not open catalog row: {error}"
            self.status.visible = True
        finally:
            self._updating = False

    def _select(self, *_):
        layer = self.selected
        self._update_menus()
        self._controls.clear()
        self._color_by = None
        self._legend.object = ""
        if layer is None:
            self._info.object = ""
            return
        self._controls.append(mui.Typography(layer.name, variant="subtitle2"))
        if isinstance(layer, VectorLayer):
            self._vector = layer
            choices = dict(layer.param.column.objects.items())
            self._color_by = mui.Select(
                label="Color by",
                options=list(choices),
                size="small",
                value=next(
                    label for label, column in choices.items() if column == layer.column
                ),
                sizing_mode="stretch_width",
            )
            self._color_by.param.watch(
                lambda event: setattr(layer, "column", choices[event.new]), "value"
            )
            self._controls.extend(
                [
                    self._color_by,
                    mui.ColorPicker.from_param(
                        layer.param.color,
                        visible=pn.bind(
                            lambda column: column is None, layer.param.column
                        ),
                        sizing_mode="stretch_width",
                    ),
                ]
            )
        else:

            def continuous(display):
                return not layer.is_rgb and not (layer.legend and layer.legend.classes)

            def stretchable(display):
                return not (layer.legend and layer.legend.classes)

            self._controls.extend(
                [
                    mui.Select.from_param(
                        layer.param.display, size="small", sizing_mode="stretch_width"
                    ),
                    mui.Select.from_param(
                        layer.param.cmap,
                        size="small",
                        sizing_mode="stretch_width",
                        visible=pn.bind(continuous, layer.param.display),
                    ),
                    mui.RangeSlider.from_param(
                        layer.param.stretch,
                        throttled=True,
                        sizing_mode="stretch_width",
                        visible=pn.bind(stretchable, layer.param.display),
                    ),
                    mui.Card(
                        objects=[
                            mui.TupleInput.from_param(
                                layer.param.clim,
                                label="Manual range",
                                sizing_mode="stretch_width",
                            )
                        ],
                        title="Advanced contrast",
                        collapsible=True,
                        collapsed=True,
                        visible=pn.bind(stretchable, layer.param.display),
                        sizing_mode="stretch_width",
                        margin=0,
                        sx={"boxShadow": "none", "border": 1, "borderColor": "divider"},
                    ),
                ]
            )
            if len(layer.param.time.objects) > 1:
                self._controls.append(
                    mui.Select.from_param(
                        layer.param.time, size="small", sizing_mode="stretch_width"
                    )
                )
        self._controls.append(
            mui.FloatSlider.from_param(layer.param.alpha, sizing_mode="stretch_width")
        )
        values = (
            ["column", "color"]
            if isinstance(layer, VectorLayer)
            else ["display", "cmap", "clim", "limits"]
        )
        self._controls.append(
            pn.bind(self._show_legend, layer, *[layer.param[name] for name in values])
        )
        self._info.object = self._format_info(layer.get_info())
        self._show_table()

    def _format_info(self, values):
        rows = []
        for name, value in values.items():
            title = html.escape(str(name))
            if isinstance(value, dict | list | tuple):
                text = html.escape(json.dumps(value, indent=2, default=str))
                rows.append(
                    f'<details><summary>{title}</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:280px;overflow:auto">{text}</pre></details>'
                )
            else:
                rows.append(f"<div><b>{title}</b><br>{html.escape(str(value))}</div>")
        return (
            '<div style="overflow-wrap:anywhere;font-size:13px;line-height:1.6">'
            + "".join(rows)
            + "</div>"
        )

    def _show_legend(self, layer, *_):
        if layer is None:
            return ""
        if isinstance(layer, VectorLayer):
            colors = layer.category_colors or (
                {"Features": layer.color} if layer.column is None else {}
            )
            limits = (
                None
                if colors
                else (
                    float(layer.data[layer.column].min()),
                    float(layer.data[layer.column].max()),
                )
            )
        else:
            legend = layer.legend
            colors = (
                {}
                if legend is None or not legend.classes
                else {
                    str(title): str(layer.colors[code])
                    for code, title in legend.classes.items()
                }
            )
            if layer.is_rgb:
                channels = " / ".join(html.escape(name) for name in layer.rgb)
                self._legend.object = f"<small>RGB composite · {channels}</small>"
                return ""
            limits = layer.clim or layer.limits
        if colors:
            self._legend.object = (
                '<div style="display:grid;gap:5px;font-size:13px">'
                + "".join(
                    f'<div style="display:flex;align-items:center;gap:8px"><span style="flex:none;width:14px;height:14px;border:1px solid #8888;border-radius:2px;background:{html.escape(color)}"></span><span>{html.escape(str(name)).replace("_", " ")}</span></div>'
                    for name, color in colors.items()
                )
                + "</div>"
            )
        else:
            cmap = layer.cmap if isinstance(layer, RasterLayer) else "viridis"
            ramp = get_ramp(cmap)
            palette = ",".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in ramp[::16])
            low, high = limits or (0.0, 1.0)
            title = layer.display if isinstance(layer, RasterLayer) else layer.column
            self._legend.object = f'<div style="font-size:13px"><b>{html.escape(str(title))}</b><div style="height:10px;margin:6px 0;border-radius:3px;background:linear-gradient(to right,{palette})"></div><small>{low:.4g} — {high:.4g}</small></div>'
        return ""

    def _show_table(self):
        layer = self._vector
        vectors = [value for value in self.layers if isinstance(value, VectorLayer)]
        if vectors != self._table_layers:
            self._table_layers = vectors
            tables = []
            for vector in vectors:
                controls = []
                if vector.period is not None:
                    controls.append(
                        mui.DateRangePicker.from_param(
                            vector.param.period, size="small", width=300
                        )
                    )
                controls.append(vector.table)
                tables.append(
                    (vector.name, mui.Column(*controls, sizing_mode="stretch_width"))
                )
            self._attributes[:] = tables
            self._attributes.sx = {
                ".MuiTabs-root": {"display": "none" if len(tables) == 1 else "flex"}
            }
        self._catalog.visible = layer is not None
        if layer is not None:
            self._attributes.active = vectors.index(layer)
            self._catalog.title = (
                f"Catalog · {layer.name}"
                if layer.is_catalog
                else f"Attributes · {layer.name}"
            )

    def _select_table(self, event):
        if self._table_layers and event.new is not None:
            layer = self._table_layers[event.new]
            if self.selected is not layer:
                self.selected = layer

    def zoom(self, bbox):
        """Frame a layer without changing its display settings."""
        extent = frame_bbox(bbox)
        width, height = self._size.width, self._size.height
        if width and height:
            ratio = width / height
            if extent.span_x / extent.span_y < ratio:
                extent = extent.buffered((extent.span_y * ratio - extent.span_x) / 2, 0)
            else:
                extent = extent.buffered(0, (extent.span_x / ratio - extent.span_y) / 2)
        self._ranges.update(
            x_range=(extent.left, extent.right), y_range=(extent.bottom, extent.top)
        )
        self._extent = extent
        self._compose()

    def _frame_map(self, plot, _):
        # RangeXY reports navigation; emitting it does not move a drawn map.
        # Use the public Bokeh ranges only for an explicit framing request.
        if self._extent is None:
            return
        extent = self._extent
        plot.state.x_range.update(start=extent.left, end=extent.right)
        plot.state.y_range.update(start=extent.bottom, end=extent.top)
        self._extent = None

    def add(self, *items):
        """Add native rasters or frames to the map."""
        self.layers = [*self.layers, *create_layers(items)]

    def remove(self, layer):
        """Remove a layer, including catalog rasters and their checked rows."""
        if isinstance(layer, VectorLayer) and layer in self._opened:
            removed = list(self._opened.pop(layer).values()) + [layer]
        else:
            removed = [layer]
            for catalog, opened in list(self._opened.items()):
                positions = {key[0] for key, value in opened.items() if value is layer}
                if positions:
                    catalog.selected = [
                        row for row in catalog.selected if row not in positions
                    ]
        self.layers = [value for value in self.layers if value not in removed]
        if isinstance(layer, VectorLayer):
            layer.close()

    def _identify(self, x=None, y=None, **_):
        if x is None or y is None or self._closed:
            return
        x_range = self._ranges.x_range
        if x_range is None:
            return
        tolerance = (x_range[1] - x_range[0]) / 900 * 5
        values = {}
        for layer in reversed(self.layers):
            if layer.visible:
                found = layer.identify(x, y, tolerance)
                if found:
                    values[layer.name] = found
        self._identified.object = self._format_info(values)

    def __mul__(self, other):
        """Compose a new explorer with independent display choices."""
        groups = [(self.layers, self._opened)]
        if isinstance(other, Explorer):
            groups.append((other.layers, other._opened))
        else:
            groups.append((create_layers(other), {}))
        layers = []
        opened = {}
        for originals, catalogs in groups:
            copies = {}
            for layer in originals:
                if isinstance(layer, RasterLayer):
                    copied = RasterLayer(layer.data, name=layer.name, rgb=layer.rgb)
                else:
                    copied = VectorLayer(layer.data, name=layer.name)
                settings = {
                    name: copy.copy(value)
                    for name, value in layer.param.values().items()
                    if name in copied.param and name != "name"
                }
                copied.param.update(**settings)
                layers.append(copied)
                copies[layer] = copied
            for catalog, rasters in catalogs.items():
                opened[copies[catalog]] = {
                    key: copies[layer] for key, layer in rasters.items()
                }
        composed = Explorer(*layers, basemap=self.basemap, height=self.height)
        composed._opened.update(opened)
        return composed

    def close(self):
        """Release the native pane and table callbacks."""
        if self._closed:
            return
        self._closed = True
        for watcher in self._watchers:
            watcher.inst.param.unwatch(watcher)
        self._watchers.clear()
        for layer, watcher in self._catalog_watchers.items():
            layer.param.unwatch(watcher)
        self._catalog_watchers.clear()
        for layer, watcher in self._layer_watchers.items():
            layer.param.unwatch(watcher)
        self._layer_watchers.clear()
        for layer in self.layers:
            if isinstance(layer, VectorLayer):
                layer.close()
        self._tap.clear()
        self.map.object = None
        self.layers = []
        self.selected = None
