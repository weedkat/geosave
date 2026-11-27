"""Preprocessing steps, one class per `DataTree.gs` method and named after it.

A step keeps the method's arguments under the same names, so a config selects
it by `class_path` with those arguments as `init_args`. Calling the step runs
the method on a tree.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Literal

import xarray as xr

from geosave.geodata.core.raster.transform.temporal import Interpolation, Reducer
from geosave.geodata.utils.datetime import Freq


class Step(ABC):
    """One preprocessing operation from a tree to a tree."""

    @abstractmethod
    def __call__(self, tree: xr.DataTree) -> xr.DataTree:
        """Return the tree after this operation."""


class MaskAndScale(Step):
    """Decode every raster node's CF values with `DataTree.gs.mask_and_scale`.

    Examples:
        >>> decoded = MaskAndScale()(sources)
    """

    def __call__(self, tree: xr.DataTree) -> xr.DataTree:
        """Return the tree with each node's packing and fill decoded."""
        return tree.gs.mask_and_scale()


class Composite(Step):
    """Reduce every dated node into calendar bins with `DataTree.gs.composite`.

    Args:
        freq: Pandas offset alias, such as "MS".
        method: One native reducer for every dated node; class IDs permit
            only first or last.
        dim: Datetime dimension to resample.
        interpolation: Fill interior pixel gaps after reduction. None keeps gaps.
        closed: Inclusive bin edge, following native resample defaults.
        label: Bin label edge, following native resample defaults.
        origin: Native resample origin.
        offset: Shift applied to the origin.

    Examples:
        >>> monthly = Composite("MS", "median")(decoded)
    """

    def __init__(
        self,
        freq: Freq,
        method: Reducer,
        *,
        dim: str = "time",
        interpolation: Interpolation | None = None,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        origin: str | datetime = "start_day",
        offset: str | timedelta | None = None,
    ) -> None:
        self.freq = freq
        self.method: Reducer = method
        self.dim = dim
        self.interpolation: Interpolation | None = interpolation
        self.closed: Literal["left", "right"] | None = closed
        self.label: Literal["left", "right"] | None = label
        self.origin = origin
        self.offset = offset

    def __call__(self, tree: xr.DataTree) -> xr.DataTree:
        """Return the tree with each dated node on its calendar bins."""
        return tree.gs.composite(
            self.freq,
            self.method,
            dim=self.dim,
            interpolation=self.interpolation,
            closed=self.closed,
            label=self.label,
            origin=self.origin,
            offset=self.offset,
        )
