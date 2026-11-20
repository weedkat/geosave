"""The predictor role and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import xarray as xr
    from torch import nn


class RasterPredictor[Output](ABC):
    """Run a model over a sample of any size and place what it returns in space.

    `Output` is the type the task's model output takes once it is placed back
    on the map: a `(band, y, x)` DataArray for a per-pixel task, a
    GeoDataFrame for detection. A postprocessor taking the same `Output`
    decodes it.

    Examples:
        >>> class LandCoverPredictor(RasterPredictor[xr.DataArray]):
        ...     def __call__(self, model: nn.Module, sample: xr.DataTree) -> xr.DataArray:
        ...         ...
    """

    @abstractmethod
    def __call__(self, model: nn.Module, sample: xr.DataTree) -> Output:
        """Return the model's output over the whole sample.

        Args:
            model: Called with its inputs by name: `model(**inputs)`.
            sample: Model-ready rasters by name, lazy, as the preprocessor
                returns them.

        Returns:
            The model's own values, undecoded, placed back in space. For a
            per-pixel task, a float32 `(band, y, x)` DataArray on the
            sample's grid, one band per model output channel (logits for
            segmentation), NaN where the model predicted nothing.
        """

    def measure_chips(self) -> tuple[int, int]:
        """Return the chip size and the stride between chip starts, in pixels.

        Only `RasterPipeline.predict_to_zarr` asks: it reads each chunk with
        the pixels of every chip touching it, so a chunk is predicted as the
        whole raster is. A predictor that cuts no chips leaves this out.

        Raises:
            NotImplementedError: The predictor cuts no chips.
        """
        raise NotImplementedError(
            f"{type(self).__name__} states no chip size, so its pipeline cannot "
            f"predict chunk by chunk; implement measure_chips"
        )


PREDICTORS = Registry(
    RasterPredictor, imports=("geosave.ml.processing.raster.prediction.dense",)
)
