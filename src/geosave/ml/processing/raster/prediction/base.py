"""The predictor role and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import xarray as xr
    from torch import nn


class RasterPredictor(ABC):
    """Run a model over a sample of any size and assemble what it returns.

    A subclass declares its `__call__` with `@chain_step()`. The name and
    type of the variable it returns are what its task's postprocessor takes;
    a pipeline refuses a predictor and a postprocessor that disagree.
    """

    @abstractmethod
    def __call__(self, model: nn.Module, sample: xr.DataTree) -> Any:
        """Return the model's output over the whole sample.

        Args:
            model: Called with its inputs by name: `model(**inputs)`.
            sample: Model-ready rasters by name, lazy, on one grid.

        Returns:
            `output`: the model's own values, undecoded. For a per-pixel
            task, a float32 `(band, y, x)` DataArray on the sample's grid,
            one band per model output channel (logits for segmentation), NaN
            where the model predicted nothing.
        """

    @abstractmethod
    def measure_chips(self) -> tuple[int, int]:
        """Return the chip size and the stride between chip starts, in pixels.

        `RasterPipeline.predict_to_zarr` reads each chunk with the pixels of
        every chip touching it, so a chunk is predicted as the whole raster is.
        """


PREDICTORS = Registry(
    RasterPredictor, imports=("geosave.ml.processing.raster.prediction.dense",)
)
