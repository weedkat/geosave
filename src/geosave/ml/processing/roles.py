"""Sub-processors: the interchangeable steps around a model, and their registries.

    region ──Ingestor──▶ rasters ──Preprocessor.preprocess──▶ image
        ──Predictor (Preprocessor.normalize ▶ model)──▶ logits ──Postprocessor──▶ product

Each role is one protocol with one registry; a class registers by name in
its role's registry, as `@PREPROCESSORS.register("sentinel2_reflectance")`.
`geosave.ml.processing.processor.Processor` runs the four roles around a model.
Training uses the preprocessor alone, so inference prepares its input
exactly as training did.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from geosave.ml.factory import Registry

if TYPE_CHECKING:
    import torch
    import xarray as xr

    from geosave.geodata import GeoRegion


@runtime_checkable
class Ingestor(Protocol):
    """Load what a model predicts over a region; inference only."""

    def ingest(self, region: GeoRegion) -> list[xr.Dataset]:
        """Return one lazy raster per acquisition over the region."""
        ...


@runtime_checkable
class Preprocessor(Protocol):
    """Turn a stored raster into model input; shared by training and inference."""

    def preprocess(self, raster: xr.Dataset) -> xr.Dataset:
        """Decode a whole raster lazily, before it is cropped or chipped."""
        ...

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Standardize a `(..., band, y, x)` batch, after augmentation."""
        ...


@runtime_checkable
class Predictor(Protocol):
    """Run a model over a prepared raster of any size."""

    def predict(
        self, forward: Callable[[torch.Tensor], torch.Tensor], image: xr.Dataset
    ) -> xr.DataArray:
        """Return `forward`'s output over the whole image, on the image's grid.

        Args:
            forward: Maps a `(batch, band, y, x)` image batch to the model's
                `(batch, band, y, x)` output; it normalizes and runs the model.
            image: Preprocessed raster, lazy.
        """
        ...


@runtime_checkable
class Postprocessor(Protocol):
    """Decode merged model output into the product a release publishes."""

    def postprocess(self, logits: xr.DataArray) -> xr.Dataset:
        """Decode merged `(band, y, x)` logits on their grid."""
        ...


INGESTORS = Registry(Ingestor, imports=("geosave.ml.processing.ingestors",))
PREPROCESSORS = Registry(Preprocessor, imports=("geosave.ml.processing.preprocessors",))
PREDICTORS = Registry(Predictor, imports=("geosave.ml.processing.predictors",))
POSTPROCESSORS = Registry(
    Postprocessor, imports=("geosave.ml.processing.postprocessors",)
)
