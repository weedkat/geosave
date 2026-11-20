"""The change pipeline: a caller gives two acquisitions of one place and gets the model's product."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    import geopandas as gpd
    from torch import nn

    from geosave.ml.processing.change.preprocessing.base import ChangePreprocessor
    from geosave.ml.processing.raster.ingestion.base import RasterIngestor
    from geosave.ml.processing.raster.postprocessing.base import RasterPostprocessor
    from geosave.ml.processing.raster.prediction.base import RasterPredictor


class ChangePipeline[Output, Product: (xr.Dataset, gpd.GeoDataFrame)]:
    """Orchestrate inference over two acquisitions of one place.

        before, after ──preprocessor──▶ sample ──predictor (model)──▶ output
            ──postprocessor (output, sample)──▶ product

    The preprocessor brings both acquisitions onto one grid as one sample
    tree; from there the flow is a raster pipeline's, so any raster
    predictor and postprocessor fit.

    Args:
        model: Model the predictor calls with its inputs by name, such as
            `model(before=..., after=...)`.
        preprocessor: Builds one model-ready sample from both acquisitions.
        predictor: Runs the model over a sample of any size.
        postprocessor: Turns the predictor's output into the product.
        ingestor: Loads one acquisition's raw rasters, called once per date
            by the caller; None for a pipeline whose caller supplies them.

    Examples:
        >>> pipeline = ChangePipeline(
        ...     model, BandDifference(bands), DensePredictor(size=256), ChangeMap()
        ... )
        >>> before = pipeline.ingestor(region.set_time("2023-06"))[0]
        >>> after = pipeline.ingestor(region.set_time("2024-06"))[0]
        >>> pipeline.predict(before, after)
        <xarray.Dataset> ...
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: ChangePreprocessor,
        predictor: RasterPredictor[Output],
        postprocessor: RasterPostprocessor[Output, Product],
        ingestor: RasterIngestor | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.ingestor = ingestor

    def predict(self, before: xr.DataTree, after: xr.DataTree) -> Product:
        """Predict the change between two acquisitions in memory.

        Args:
            before: Raw rasters of the earlier acquisition.
            after: Raw rasters of the later acquisition, with the same nodes.

        Returns:
            The postprocessor's product.
        """
        sample = self.preprocessor(before, after)
        output = self.predictor(self.model, sample)
        product = self.postprocessor(output, sample)
        return product
