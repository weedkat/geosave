"""Semantic segmentation from a region to a georeferenced class map."""

from collections.abc import Callable

import torch
import xarray as xr
from torch import nn

from geosave import GeoRegion
from geosave.geodata.io.storage import Location
from processing.ingestor import ingest
from processing.persistor import persist
from processing.postprocessor import postprocess
from processing.predictor import predict
from processing.preprocessing import Reflectance


class Pipeline:
    """Run the project's processing stages with a trained, evaluation-mode model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        size: int,
        context: dict[str, Callable[[xr.Dataset], torch.Tensor] | str] | None = None,
    ):
        self.model = model
        self.size = size
        self.context = context
        self.preprocessor = Reflectance()

    def predict(self, region: GeoRegion) -> xr.Dataset:
        """Read a region and return a class map on its source grid."""
        with ingest(region) as sources:
            prepared = self.preprocessor.preprocess(sources)
            image = prepared["image"].gs.to_dataset()
            logits = predict(self.model, image, size=self.size, context=self.context)
        return postprocess(logits)

    def run(self, region: GeoRegion, destination: Location) -> None:
        """Predict one region and save its class map."""
        persist(self.predict(region), destination)
