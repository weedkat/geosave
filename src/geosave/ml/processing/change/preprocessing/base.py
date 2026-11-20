"""The change preprocessor role and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import xarray as xr


class ChangePreprocessor(ABC):
    """Turn two acquisitions of one place into one model-ready sample."""

    @abstractmethod
    def __call__(self, before: xr.DataTree, after: xr.DataTree) -> xr.DataTree:
        """Build the sample a change model is run over, lazily.

        Args:
            before: One node per raw raster of the earlier acquisition, as a
                raster ingestor returns them.
            after: The same nodes for the later acquisition.

        Returns:
            One node per model input on one grid, such as `before` and
            `after`, each as float32 in model units, under the names the
            model takes them by. Each node keeps its acquisition's scalar
            `time`.
        """


PREPROCESSORS = Registry(ChangePreprocessor)
