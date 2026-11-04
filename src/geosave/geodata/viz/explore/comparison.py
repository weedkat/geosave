"""Comparing two rasters: the map shows one side at a time."""

from __future__ import annotations

from typing import TYPE_CHECKING

import param

if TYPE_CHECKING:
    from .layers import RasterLayer


class Comparison(param.Parameterized):
    """Choose two rasters and which one the map shows.

    The explorer offers its rasters, by label, in `rasters`. Page controls
    edit these choices; map eligibility remains on Explorer.
    """

    enabled = param.Boolean(False, label="Compare rasters")
    left = param.Selector(default=None, allow_None=True, label="Left")
    right = param.Selector(default=None, allow_None=True, label="Right")
    side = param.Selector(default="left", objects=["left", "right"], label="Showing")
    # The rasters that can be compared, by label.
    rasters = param.Dict(default={}, precedence=-1)

    @property
    def shown_raster(self) -> RasterLayer | None:
        """Return the raster the map shows, when comparing."""
        return self.left if self.side == "left" else self.right

    @param.depends("rasters", watch=True)
    def _offer_rasters(self) -> None:
        # Keep each side's choice while it is still offered.
        self.param.left.objects = self.param.right.objects = self.rasters
        offered = list(self.rasters.values())
        if self.left not in offered:
            self.left = offered[0] if offered else None
        if self.right not in offered:
            self.right = offered[-1] if offered else None
        if len(offered) < 2:
            self.enabled = False
