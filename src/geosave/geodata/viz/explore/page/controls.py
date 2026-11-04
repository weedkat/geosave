"""The Style tab's widgets for one layer, each bound to that layer's settings."""

from __future__ import annotations

import panel as pn
from panel.viewable import Viewable

from ..layers import Layer, RasterLayer, VectorLayer


def create_controls(layer: Layer) -> list[Viewable]:
    """Create the widgets that change `layer`'s settings, opacity last."""
    if isinstance(layer, RasterLayer):
        controls = create_raster_controls(layer)
    elif isinstance(layer, VectorLayer):
        controls = create_vector_controls(layer)
    else:
        controls = []
    controls.append(
        pn.ui.FloatSlider.from_param(layer.param.alpha, sizing_mode="stretch_width")
    )
    return controls


def create_raster_controls(layer: RasterLayer) -> list[Viewable]:
    """Create the band, colormap, contrast and date widgets of a raster."""

    # A colormap and a stretch apply to continuous values only.
    def has_colormap(band):
        return not layer.is_rgb and not (layer.legend and layer.legend.classes)

    def has_stretch(band):
        return not (layer.legend and layer.legend.classes)

    controls: list[Viewable] = [
        pn.ui.Select.from_param(
            layer.param.band, size="small", sizing_mode="stretch_width"
        ),
        pn.ui.Select.from_param(
            layer.param.cmap,
            size="small",
            sizing_mode="stretch_width",
            visible=pn.bind(has_colormap, layer.param.band),
        ),
        pn.ui.RangeSlider.from_param(
            layer.param.stretch_percentiles,
            throttled=True,
            sizing_mode="stretch_width",
            visible=pn.bind(has_stretch, layer.param.band),
        ),
        pn.ui.Card(
            objects=[
                pn.ui.TupleInput.from_param(
                    layer.param.clim,
                    label="Manual range",
                    sizing_mode="stretch_width",
                )
            ],
            title="Advanced contrast",
            collapsible=True,
            collapsed=True,
            visible=pn.bind(has_stretch, layer.param.band),
            sizing_mode="stretch_width",
            margin=0,
            sx={"boxShadow": "none", "border": 1, "borderColor": "divider"},
        ),
    ]
    return controls


def create_vector_controls(layer: VectorLayer) -> list[Viewable]:
    """Create the colour-by select, and the single colour when no column colours."""
    return [
        ColorBySelect(layer),
        pn.ui.ColorPicker.from_param(
            layer.param.color,
            visible=pn.bind(lambda column: column is None, layer.param.column),
            sizing_mode="stretch_width",
        ),
    ]


class ColorBySelect(pn.ui.Select):
    """Select a vector attribute with a visible label for single-color drawing."""

    def __init__(self, layer: VectorLayer) -> None:
        self._layer = layer
        self._columns = dict(layer.param.column.objects.items())
        self._labels = {}
        for label, column in self._columns.items():
            self._labels[column] = label
        super().__init__(
            label="Color by",
            options=list(self._columns),
            value=self._labels[layer.column],
            size="small",
            sizing_mode="stretch_width",
        )
        self._subscriptions = [
            self.param.watch(self._select_column, "value"),
            layer.param.watch(self._show_column, "column"),
        ]

    def _select_column(self, *_):
        self._layer.column = self._columns[self.value]

    def _show_column(self, *_):
        self.value = self._labels[self._layer.column]

    def release(self) -> None:
        """Stop synchronizing this control with its layer."""
        for watcher in self._subscriptions:
            watcher.inst.param.unwatch(watcher)
        self._subscriptions.clear()
