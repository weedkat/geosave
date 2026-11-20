"""The prompt postprocessor role and its registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr


class PromptPostprocessor[Output, Product: (xr.Dataset, gpd.GeoDataFrame)](ABC):
    """Turn a promptable model's answers into the product, one per prompt."""

    @abstractmethod
    def __call__(
        self, output: Output, sample: xr.DataTree, prompts: gpd.GeoDataFrame
    ) -> Product:
        """Decode the answers to each prompt.

        Args:
            output: The prompt predictor's output.
            sample: The sample the answers were predicted over; it holds the
                acquisition's scalar coordinates.
            prompts: The prompts as the caller gave them, whose attributes
                the product keeps.

        Returns:
            A GeoDataFrame with one row per prompt, such as the polygon of
            its mask, indexed like `prompts`; or a raster Dataset on the
            sample's grid.
        """


POSTPROCESSORS = Registry(PromptPostprocessor)
