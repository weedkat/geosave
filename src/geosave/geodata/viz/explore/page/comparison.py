"""Controls for choosing the rasters being compared."""

import panel as pn

from ..comparison import Comparison


def create_comparison(comparison: Comparison):
    """Return controls for the comparison state."""
    return pn.ui.Card(
        objects=[
            pn.ui.Checkbox.from_param(comparison.param.enabled),
            pn.ui.Select.from_param(comparison.param.left, size="small"),
            pn.ui.Select.from_param(comparison.param.right, size="small"),
            pn.ui.RadioButtonGroup.from_param(comparison.param.side),
        ],
        title="Compare",
        collapsed=True,
        visible=pn.bind(lambda rasters: len(rasters) >= 2, comparison.param.rasters),
        collapsible=True,
        sizing_mode="stretch_width",
        sx={"boxShadow": "none", "borderRadius": 0},
    )
