"""Download the current map and its legends without a running Explorer."""

from __future__ import annotations

from html import escape
from io import StringIO
from typing import TYPE_CHECKING

import holoviews as hv
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.resources import INLINE

from ..layers import RasterLayer, VectorLayer
from .legend import format_legend

if TYPE_CHECKING:
    from ..explorer import Explorer


def export_map(explorer: Explorer) -> StringIO:
    """Render the current view and shown-layer legends into an HTML download."""
    snapshot = explorer.snapshot()
    figure = hv.render(snapshot, backend="bokeh")
    rasters = explorer.select_drawn_rasters()
    legends = []
    for label, layer in explorer.layers_by_label.items():
        if layer not in rasters and not (
            isinstance(layer, VectorLayer) and layer.visible
        ):
            continue
        legend = f"<h3>{escape(label)}</h3>"
        if isinstance(layer, RasterLayer) and layer.time is not None:
            legend += f"<p>Date: {escape(str(layer.time))}</p>"
        legend += format_legend(layer)
        legends.append(legend)

    note = Div(
        text=(
            "<h2>GeoSave map</h2>"
            "<p>This map captures the displayed layers and detail. "
            "Pan and zoom explore the captured pixels; they do not load new detail. "
            "Basemap tiles require an internet connection.</p>"
        ),
        width=600,
    )
    legend = Div(text="".join(legends), width=300)
    document = column(note, figure, legend)
    return StringIO(file_html(document, INLINE, "GeoSave map"))
