"""The explorer's page: the sidebar beside the map, and the attribute tables below it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn
from panel.util.parameters import edit_readonly

from .comparison import create_comparison
from .export import export_map
from .reflow import Reflow
from .theme import EXPLORER_THEME, PAGE_STYLE

if TYPE_CHECKING:
    from ..explorer import Explorer

# Width of the sidebar, in pixels.
SIDEBAR_WIDTH = 300


def create_page(explorer: Explorer) -> pn.ui.Page:
    """Lay out the explorer: basemap, layers, comparison and details beside the map."""
    page = pn.ui.Page(
        title="GeoSave",
        height=explorer.height,
        sidebar_width=SIDEBAR_WIDTH,
        sidebar_resizable=True,
        sizing_mode="stretch_width",
        theme_config=EXPLORER_THEME,
        sx=PAGE_STYLE,
    )
    # Mirroring global busy state in inline pages can create a notebook
    # widget feedback loop that starves raster updates.
    with edit_readonly(page):
        page.busy = False
    page.header = [
        pn.ui.FileDownload(
            callback=lambda: export_map(explorer),
            filename="geosave-map.html",
            label="Export map",
            icon="download",
            variant="text",
            size="small",
            description="Download the current map and legends as HTML.",
        )
    ]
    page.sidebar = [create_sidebar(explorer)]
    page.main = [
        pn.ui.Column(
            explorer.status,
            Reflow(object=explorer.map, sizing_mode="stretch_both"),
            explorer.tables,
            sizing_mode="stretch_both",
            margin=0,
        )
    ]
    return page


def create_sidebar(explorer: Explorer) -> pn.ui.Column:
    """Stack the basemap, the layer list, the comparison and the selected layer's details."""
    fit_all = pn.ui.Button(
        label="Fit visible layers",
        icon="fullscreen",
        size="small",
        variant="text",
        on_click=lambda _: explorer.fit_visible(),
        disabled=pn.bind(lambda layers: not layers, explorer.param.layers),
    )
    return pn.ui.Column(
        pn.ui.Select.from_param(
            explorer.param.basemap,
            name="Basemap",
            size="small",
            sizing_mode="stretch_width",
            margin=(12, 12, 8, 12),
        ),
        create_period_controls(explorer),
        pn.ui.Divider(),
        pn.ui.Row(
            pn.ui.Typography("Layers", variant="subtitle2", margin=(6, 0)),
            pn.ui.Spacer(sizing_mode="stretch_width"),
            fit_all,
            margin=(4, 12, 0, 12),
            sizing_mode="stretch_width",
        ),
        explorer.layer_list,
        pn.ui.Divider(),
        create_comparison(explorer.comparison),
        explorer.details,
        sizing_mode="stretch_width",
        margin=0,
    )


def create_period_controls(explorer: Explorer) -> pn.ui.Column:
    """Create the date step and the date slider, shown while a raster has dates."""
    return pn.ui.Column(
        pn.ui.Select.from_param(
            explorer.param.freq, size="small", sizing_mode="stretch_width"
        ),
        pn.ui.DiscreteSlider.from_param(
            explorer.param.period,
            # Material UI cannot render an empty discrete slider, even hidden.
            # Param replaces these options when dated layers become available.
            options=explorer.param.period.objects or {"No dates": None},
            sizing_mode="stretch_width",
        ),
        visible=pn.bind(lambda count: count > 0, explorer.param.period_count),
        margin=(0, 12, 8, 12),
        sizing_mode="stretch_width",
    )
