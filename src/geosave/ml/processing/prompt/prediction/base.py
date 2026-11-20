"""The prompt predictor role and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr
    from torch import nn


class PromptPredictor[Output](ABC):
    """Run a promptable model at each prompt and place what it returns in space.

    The predictor owns the move between map and pixel coordinates both ways:
    prompts arrive in map coordinates and answers leave in them.
    """

    @abstractmethod
    def __call__(
        self, model: nn.Module, sample: xr.DataTree, prompts: gpd.GeoDataFrame
    ) -> Output:
        """Return the model's answer to every prompt over the sample.

        Args:
            model: Called with the pixels and the prompts in pixel coordinates.
            sample: Model-ready rasters by name, as a raster preprocessor
                returns them.
            prompts: Points or boxes in map coordinates, one row per prompt,
                in any CRS.

        Returns:
            The model's own answers, undecoded, in the sample's CRS: a
            `(prompt, y, x)` DataArray of mask logits on the sample's grid,
            or a GeoDataFrame indexed like `prompts`.
        """


PREDICTORS = Registry(PromptPredictor)
