"""The preprocessor role and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import xarray as xr


class RasterPreprocessor(ABC):
    """Turn raw rasters into a model-ready sample; shared by training and inference.

    A subclass declares its `__call__` with `@chain_step()` and returns a
    variable named `sample`: a pipeline hands that value to the predictor
    and the postprocessor, checked against what each declares it takes.
    """

    @abstractmethod
    def __call__(self, rasters: xr.DataTree) -> xr.DataTree:
        """Build the sample a model is run over from whole raw rasters, lazily.

        Everything the model needs done to its pixels happens here, band
        selection, decoding and standardizing included, so training and
        inference prepare a raster alike.

        Args:
            rasters: One node per raw raster, as an ingestor returns them.

        Returns:
            `sample`: one node per model input, under the name the model
            takes it by, each node's variables its bands in model order, as
            float32 in model units. A node the model does not take, such as
            a boolean mask, is for the postprocessor. Every node sits on one
            grid and keeps the acquisition's scalar `time`.
        """


PREPROCESSORS = Registry(
    RasterPreprocessor, imports=("geosave.ml.processing.raster.preprocessing.dense",)
)
