"""The raster pipeline: a caller gives the rasters of one acquisition and gets the model's product."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xarray as xr

from geosave.geodata.io import create_zarr, write_zarr_region
from geosave.geodata.sampling import find_chip_span

if TYPE_CHECKING:
    import geopandas as gpd
    from torch import nn
    from upath import UPath

    from geosave.geodata.io.storage import Location
    from geosave.ml.processing.raster.ingestion.base import RasterIngestor
    from geosave.ml.processing.raster.postprocessing.base import RasterPostprocessor
    from geosave.ml.processing.raster.prediction.base import RasterPredictor
    from geosave.ml.processing.raster.preprocessing.base import RasterPreprocessor


class RasterPipeline[Output, Product: (xr.Dataset, gpd.GeoDataFrame)]:
    """Orchestrate inference over the rasters of one acquisition.

        rasters ──preprocessor──▶ sample ──predictor (model)──▶ output
            ──postprocessor (output, sample)──▶ product

    The sub-processors' signatures are the contract between steps: the
    preprocessor takes and returns a DataTree, the predictor returns the
    `Output` its postprocessor takes, and the postprocessor returns the
    `Product`, a raster Dataset or a GeoDataFrame. Several sensors on one
    acquisition are several nodes of the same tree. `predict` holds the whole
    product in memory; `predict_to_zarr` writes a raster product too large for
    that chunk by chunk.

    Args:
        model: Model the predictor calls with its inputs by name.
        preprocessor: Builds the model-ready sample as training did.
        predictor: Runs the model over a sample of any size.
        postprocessor: Turns the predictor's output into the product.
        ingestor: Loads the raw rasters this pipeline accepts; the pipeline
            carries it and never runs it. None for a pipeline whose caller
            supplies every raster.

    Examples:
        >>> pipeline = RasterPipeline(
        ...     model,
        ...     preprocessor=DensePreprocessor(bands=["B04", "B08"]),
        ...     predictor=DensePredictor(size=256),
        ...     postprocessor=SegmentationPostprocessor(["land", "water"], ["#397d49", "#419bdf"]),
        ...     ingestor=Sentinel2Ingestor(bands=["B04", "B08"]),
        ... )
        >>> for rasters in pipeline.ingestor(region):
        ...     pipeline.predict(rasters)["class_id"].dtype
        dtype('uint8')
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: RasterPreprocessor,
        predictor: RasterPredictor[Output],
        postprocessor: RasterPostprocessor[Output, Product],
        ingestor: RasterIngestor | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.ingestor = ingestor

    def predict(self, rasters: xr.DataTree) -> Product:
        """Predict the raw rasters of one acquisition in memory.

        Memory holds the predictor's output for the whole raster; write a
        larger raster with `predict_to_zarr`.

        Args:
            rasters: One node per raw raster, as the ingestor returns them.

        Returns:
            The postprocessor's product.
        """
        sample = self.preprocessor(rasters)
        output = self.predictor(self.model, sample)
        product = self.postprocessor(output, sample)
        return product

    def predict_to_zarr(
        self: RasterPipeline[Any, xr.Dataset],
        rasters: xr.DataTree,
        destination: Location,
        chunks: int = 2048,
    ) -> UPath:
        """Predict the raw rasters of one acquisition into a Zarr store, chunk by chunk.

        Each chunk is read with a halo: the pixels of every chip that touches
        it. The chunk is predicted with its halo and written without it, so
        the store holds what `predict` returns while memory holds one
        chunk's output.

        Args:
            rasters: One node per raw raster, lazy, as the ingestor
                returns them.
            destination: Store to create, ending in `.zarr`.
            chunks: Height and width in pixels of a chunk, also the store's
                chunk, rounded down to a multiple of the predictor's stride;
                choose the largest whose output fits memory.

        Returns:
            Path of the store.

        Raises:
            FileExistsError: The destination exists.
            NotImplementedError: The predictor states no chip size.
            TypeError: The postprocessor's product is not a raster Dataset.
        """
        sample = self.preprocessor(rasters)
        grid = sample.gs.common_geobox()
        height, width = grid.shape

        size, stride = self.predictor.measure_chips()
        # Chunks start on the chip stride, so a chunk's chips are the raster's.
        step = max(chunks // stride, 1) * stride

        store = None
        for top in range(0, height, step):
            for left in range(0, width, step):
                bottom = min(top + step, height)
                right = min(left + step, width)
                chunk_grid = grid[top:bottom, left:right]

                halo_top, halo_bottom = find_chip_span(
                    top, bottom, height, size=size, stride=stride
                )
                halo_left, halo_right = find_chip_span(
                    left, right, width, size=size, stride=stride
                )
                # A chunk is loaded once with its halo, not read chip by chip.
                halo_grid = grid[halo_top:halo_bottom, halo_left:halo_right]
                chunk = sample.gs.crop_to_grid(halo_grid).compute()
                output = self.predictor(self.model, chunk)
                product = self.postprocessor(output, chunk)
                # A loaded release is not type-checked; refuse a vector product here.
                if not isinstance(product, xr.Dataset):
                    raise TypeError(
                        "predict_to_zarr writes raster products; this "
                        f"postprocessor returns {type(product).__name__}"
                    )
                product = product.gs.crop_to_grid(chunk_grid)

                # The first chunk's product states the store's variables and attrs.
                if store is None:
                    store = create_zarr(product, grid, destination, chunks=step)
                write_zarr_region(product, store)
        return store
