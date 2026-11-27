"""Regional inference for regression."""

from torch import nn

from geosave.geodata import GeoRegion
from geosave.geodata.io.storage import Location
from processing.ingestor import ingest
from processing.preprocessing import preprocess
from processing.predictor import predict
from processing.postprocessor import postprocess
from processing.persistor import persist


class Pipeline:
    """Run this project's five processing stages with a trained model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def predict(self, region: GeoRegion):
        """Return the project's prediction for a region."""
        sources = ingest(region)
        inputs = preprocess(sources)
        prediction = predict(self.model, inputs)
        return postprocess(prediction)

    def run(self, region: GeoRegion, destination: Location):
        """Predict and save the resulting product."""
        return persist(self.predict(region), destination)
