"""The processor: a caller gives a region or a raster and gets the model's product."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import xarray as xr
from pydantic import BaseModel, ConfigDict

from geosave.geodata.io import open_raster
from geosave.ml.factory import BuildSpec
from geosave.ml.processing.roles import (
    INGESTORS,
    POSTPROCESSORS,
    PREDICTORS,
    PREPROCESSORS,
    Ingestor,
    Postprocessor,
    Predictor,
    Preprocessor,
)

if TYPE_CHECKING:
    from geosave.geodata import GeoRegion
    from geosave.geodata.io.storage import Location


class ProcessorSpec(BaseModel):
    """Name the four sub-processors of a processor, as a release stores them.

    Attributes:
        ingestor: Loads the rasters of a region.
        preprocessor: Prepares a raster and its batches as training did.
        predictor: Runs the model over a prepared raster.
        postprocessor: Decodes the model's output into the product.
    """

    model_config = ConfigDict(extra="forbid")

    ingestor: BuildSpec
    preprocessor: BuildSpec
    predictor: BuildSpec
    postprocessor: BuildSpec


class Processor:
    """Run a model over what a caller has: a region, a raster file or a raster.

        region ──ingestor──▶ rasters ──preprocessor──▶ image
            ──predictor (normalize ▶ model)──▶ logits ──postprocessor──▶ product

    Each step is a sub-processor and can be exchanged for another of its role.

    Args:
        model: Model called as `model(image=batch)`.
        ingestor: Loads the rasters of a region.
        preprocessor: Prepares a raster and its batches as training did.
        predictor: Runs the model over a prepared raster of any size.
        postprocessor: Decodes the model's output into the product.
        spec: Every sub-processor's name and arguments, as a release stores
            them; None for sub-processors built by hand, which a release
            cannot rebuild.

    Examples:
        >>> processor = load_processor("org/dynamic-world")
        >>> processor.predict_path("scene.tif")["class_id"].dtype
        dtype('uint8')
        >>> len(processor.predict_region(region))
        3
    """

    def __init__(
        self,
        model: Any,
        ingestor: Ingestor,
        preprocessor: Preprocessor,
        predictor: Predictor,
        postprocessor: Postprocessor,
        spec: ProcessorSpec | None = None,
    ) -> None:
        self.model = model
        self.ingestor = ingestor
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.spec = spec

    def predict_region(self, region: GeoRegion) -> list[xr.Dataset]:
        """Ingest a region and predict each of its rasters.

        Returns:
            One product per raster the ingestor loaded, in its order.
        """
        products = []
        for raster in self.ingestor.ingest(region):
            products.append(self.predict_raster(raster))
        return products

    def predict_path(self, path: Location) -> xr.Dataset:
        """Open one raster file lazily and predict it."""
        return self.predict_raster(open_raster(path))

    def predict_raster(self, raster: xr.Dataset) -> xr.Dataset:
        """Predict one raster as the ingestor returns it, before preprocessing.

        Returns:
            The postprocessor's product on the raster's grid, with the
            raster's scalar coordinates, such as its `time`.
        """
        image = self.preprocessor.preprocess(raster)
        logits = self.predictor.predict(self.forward, image)
        product = self.postprocessor.postprocess(logits)

        # The predictor rebuilds coordinates from the grid, so the raster's
        # identity (its acquisition time) is carried over here.
        for name, coord in raster.coords.items():
            if coord.ndim == 0 and name not in product.coords:
                product = product.assign_coords({name: coord})
        return product

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Normalize an image batch and run the model on its device, without gradients."""
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            return self.model(image=self.preprocessor.normalize(batch.to(device)))


def build_processor(spec: ProcessorSpec, model: Any) -> Processor:
    """Construct a processor around a model from registered sub-processors.

    Args:
        spec: The four sub-processors by registered name, with arguments.
        model: The model the processor runs.

    Returns:
        Processor whose `spec` holds every argument, defaults included, so a
        release rebuilds the same processor.

    Raises:
        KeyError: A sub-processor name is not registered.
        TypeError: An argument does not fit a constructor.

    Examples:
        >>> spec = ProcessorSpec(
        ...     ingestor={"name": "sentinel2_l2a", "init_args": {"bands": ["B04"]}},
        ...     preprocessor={"name": "sentinel2_reflectance", "init_args": {"bands": ["B04"]}},
        ...     predictor={"name": "chips", "init_args": {"size": 256}},
        ...     postprocessor={"name": "class_map", "init_args": {"classes": ["water"], "colors": ["#419bdf"]}},
        ... )
        >>> build_processor(spec, model).predict_path("scene.tif")
        <xarray.Dataset> ...
    """
    full_spec = ProcessorSpec(
        ingestor=INGESTORS.get_spec(spec.ingestor),
        preprocessor=PREPROCESSORS.get_spec(spec.preprocessor),
        predictor=PREDICTORS.get_spec(spec.predictor),
        postprocessor=POSTPROCESSORS.get_spec(spec.postprocessor),
    )
    return Processor(
        model,
        ingestor=INGESTORS.build(full_spec.ingestor),
        preprocessor=PREPROCESSORS.build(full_spec.preprocessor),
        predictor=PREDICTORS.build(full_spec.predictor),
        postprocessor=POSTPROCESSORS.build(full_spec.postprocessor),
        spec=full_spec,
    )
