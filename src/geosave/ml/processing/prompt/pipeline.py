"""The prompt pipeline: a caller gives rasters and the shapes to look at, and gets one answer per shape."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    import geopandas as gpd
    from torch import nn

    from geosave.ml.processing.prompt.postprocessing.base import PromptPostprocessor
    from geosave.ml.processing.prompt.prediction.base import PromptPredictor
    from geosave.ml.processing.raster.ingestion.base import RasterIngestor
    from geosave.ml.processing.raster.preprocessing.base import RasterPreprocessor


class PromptPipeline[Output, Product: (xr.Dataset, gpd.GeoDataFrame)]:
    """Orchestrate inference of a promptable model, such as SAM, over one acquisition.

        rasters ──preprocessor──▶ sample ──predictor (model, prompts)──▶ output
            ──postprocessor (output, sample, prompts)──▶ product

    Args:
        model: Promptable model the predictor calls.
        preprocessor: Prepares the pixels, as in a raster pipeline.
        predictor: Runs the model at each prompt.
        postprocessor: Turns the answers into the product.
        ingestor: Loads the raw rasters this pipeline accepts; None for a
            pipeline whose caller supplies every raster.

    Examples:
        >>> pipeline = PromptPipeline(model, RgbPreprocessor(), BoxPrompts(), MaskPolygons())
        >>> fields = pipeline.predict(rasters, prompts)
        >>> fields.geometry.area
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: RasterPreprocessor,
        predictor: PromptPredictor[Output],
        postprocessor: PromptPostprocessor[Output, Product],
        ingestor: RasterIngestor | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.ingestor = ingestor

    def predict(self, rasters: xr.DataTree, prompts: gpd.GeoDataFrame) -> Product:
        """Answer every prompt over the raw rasters of one acquisition, in memory.

        Args:
            rasters: One node per raw raster, as the ingestor returns them.
            prompts: Points or boxes in map coordinates, one row per prompt.

        Returns:
            The postprocessor's product.
        """
        sample = self.preprocessor(rasters)
        output = self.predictor(self.model, sample, prompts)
        product = self.postprocessor(output, sample, prompts)
        return product
