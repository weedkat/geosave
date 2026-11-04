"""The explorer's sidebar: the layer list, and the selected layer's style and info."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import panel as pn
import param
from panel.viewable import Viewer

from ..layers import Layer

from .controls import ColorBySelect, create_controls
from .legend import format_info, format_legend
from .theme import MATERIAL_STYLESHEET, MENU_STYLE

if TYPE_CHECKING:
    from ..explorer import Explorer

# Layer list headings, in the order they are listed.
HEADINGS = ("Catalogs", "Rasters", "Vectors")

IDENTIFY_PROMPT = "<p>Click the map to inspect visible layers.</p>"


class LayerList(Viewer):
    """List the explorer's layers under their headings; choosing one selects it.

    Each entry shows the layer's summary, and toggles, frames or removes it.
    """

    def __init__(self, explorer: Explorer) -> None:
        """List the layers of `explorer`."""
        super().__init__()
        self._explorer = explorer
        self._menus = {}
        self._groups = {}
        self._subscriptions = []
        self._layer_watchers = {}
        for heading in HEADINGS:
            menu = pn.ui.MenuList(
                items=[],
                dense=True,
                sizing_mode="stretch_width",
                sx=MENU_STYLE,
            )
            menu.on_action("Visibility", self._toggle_layer)
            menu.on_action("Zoom to layer", self._zoom_to_layer)
            menu.on_action("Remove layer", self._remove_layer)
            self._subscriptions.append(menu.param.watch(self._select_layer, "value"))
            self._menus[heading] = menu
            self._groups[heading] = pn.ui.Column(
                pn.ui.Typography(
                    heading,
                    variant="overline",
                    color="text.secondary",
                    margin=(4, 12, 0, 12),
                ),
                menu,
                sizing_mode="stretch_width",
                margin=0,
            )
        self._layout = pn.ui.Column(
            *self._groups.values(),
            sizing_mode="stretch_width",
            margin=0,
            max_height=280,
            scroll=True,
        )
        self._subscriptions.extend(
            [
                explorer.param.watch(self._follow_layers, "layers"),
                explorer.param.watch(self._follow_selection, "active_layer"),
            ]
        )
        self._released = False
        self._follow_layers()

    def __panel__(self):
        return self._layout

    def _follow_layers(self, *_):
        for layer in list(self._layer_watchers):
            if layer not in self._explorer.layers:
                layer.param.unwatch(self._layer_watchers.pop(layer))
        for layer in self._explorer.layers:
            if layer not in self._layer_watchers:
                self._layer_watchers[layer] = layer.param.watch(
                    self._update_entries, list(layer.summary_params)
                )
        self._update_entries()

    def _update_entries(self, *_):
        for heading, menu in self._menus.items():
            menu.items = self._list_entries(heading)
            self._groups[heading].visible = bool(menu.items)
        self._follow_selection()

    def _follow_selection(self, *_):
        for heading, menu in self._menus.items():
            menu.active = self._find_selected_entry(heading)

    def release(self):
        """Detach navigation from the explorer and its layers."""
        self._released = True
        for watcher in self._subscriptions:
            watcher.inst.param.unwatch(watcher)
        self._subscriptions.clear()
        for layer, watcher in self._layer_watchers.items():
            layer.param.unwatch(watcher)
        self._layer_watchers.clear()

    def _list_entries(self, heading: str, *_: Any) -> list[dict[str, Any]]:
        entries = []
        for label, layer in self._explorer.layers_by_label.items():
            if layer.heading != heading:
                continue
            entries.append(
                {
                    "label": label,
                    # Kept on the Python side only; the browser sees the label.
                    "layer": layer,
                    "icon": layer.icon,
                    "secondary": layer.summary,
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
        return entries

    def _find_selected_entry(self, heading: str, *_: Any) -> tuple[int] | None:
        entries = self._menus[heading].items
        for position, entry in enumerate(entries):
            if self._find_layer(entry) is self._explorer.active_layer:
                return (position,)
        return None

    @staticmethod
    def _find_layer(entry: dict[str, Any]) -> Layer | None:
        return entry.get("layer")

    def _select_layer(self, event):
        layer = self._find_layer(event.new) if event.new else None
        if layer is not None and self._explorer.active_layer is not layer:
            self._explorer.active_layer = layer

    def _toggle_layer(self, entry):
        if self._released:
            return
        layer = self._find_layer(entry)
        if layer is not None:
            layer.visible = not layer.visible

    def _zoom_to_layer(self, entry):
        if self._released:
            return
        layer = self._find_layer(entry)
        if layer is not None:
            self._explorer.zoom(layer.bbox)

    def _remove_layer(self, entry):
        if self._released:
            return
        layer = self._find_layer(entry)
        if layer is not None:
            self._explorer.remove(layer)


class LayerDetails(Viewer):
    """Show the selected layer's style widgets and legend, its info, and map clicks.

    Each layer's widgets are built once and kept, so switching layers neither
    rebuilds nor leaks them.
    """

    layer = param.ClassSelector(class_=Layer, allow_None=True, allow_refs=True)
    clicked_values = param.Dict(default=None, allow_None=True, allow_refs=True)

    def __init__(self, explorer: Explorer) -> None:
        """Follow the selected layer and the map's identified values."""
        self._explorer = explorer
        self._styles = {}
        self._legend_watchers = {}
        self._released = False
        self._empty = pn.ui.Column(margin=0)
        super().__init__(
            layer=explorer.param.active_layer,
            clicked_values=explorer.map.param.clicked_values,
        )
        self._watcher = explorer.param.watch(self._prune_styles, "layers")
        self._layout = pn.ui.Tabs(
            (
                "Style",
                pn.ui.Column(
                    self.view_style, margin=(12, 8), sizing_mode="stretch_width"
                ),
            ),
            (
                "Info",
                pn.ui.HTML(
                    self.view_info,
                    sizing_mode="stretch_width",
                    stylesheets=[MATERIAL_STYLESHEET],
                ),
            ),
            (
                "Identify",
                pn.ui.HTML(
                    self.view_identified,
                    sizing_mode="stretch_width",
                    stylesheets=[MATERIAL_STYLESHEET],
                ),
            ),
            sizing_mode="stretch_width",
        )

    def __panel__(self):
        return self._layout

    @param.depends("layer")
    def view_style(self):
        """Return the selected layer's existing controls and legend."""
        layer = self.layer
        if self._released or layer not in self._explorer.layers:
            return self._empty
        if layer not in self._styles:
            legend = pn.ui.HTML(
                format_legend(layer),
                sizing_mode="stretch_width",
                margin=(18, 0, 8, 0),
                stylesheets=[MATERIAL_STYLESHEET],
            )

            def update_legend(*_):
                legend.object = format_legend(layer)

            self._legend_watchers[layer] = layer.param.watch(
                update_legend, list(layer.legend_params)
            )
            self._styles[layer] = pn.ui.Column(
                pn.ui.Typography(layer.name, variant="subtitle2"),
                *create_controls(layer),
                legend,
                sizing_mode="stretch_width",
                margin=0,
            )
        return self._styles[layer]

    @param.depends("layer")
    def view_info(self) -> str:
        """Return the selected layer's metadata."""
        return "" if self.layer is None else format_info(self.layer.get_info())

    @param.depends("clicked_values")
    def view_identified(self) -> str:
        """Return values from the last map click."""
        return (
            IDENTIFY_PROMPT
            if self.clicked_values is None
            else format_info(self.clicked_values)
        )

    def _prune_styles(self, *_):
        for layer in list(self._styles):
            if layer not in self._explorer.layers:
                self._release_style(layer)

    def _release_style(self, layer):
        layer.param.unwatch(self._legend_watchers.pop(layer))
        for widget in self._styles.pop(layer).select(ColorBySelect):
            widget.release()

    def release(self) -> None:
        """Detach references and custom subscriptions without updating destroyed UI."""
        if self._released:
            return
        self._released = True
        self._explorer.param.unwatch(self._watcher)
        # Assigning plain values detaches Param references. Suppress display
        # events because a server session may already have destroyed its models.
        with param.discard_events(self):
            self.layer = None
            self.clicked_values = None
        for layer in list(self._styles):
            self._release_style(layer)
