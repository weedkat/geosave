"""The raster pipeline: a caller gives raw rasters and gets the model's product."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import xarray as xr

from geosave.geodata.io import create_zarr, write_zarr_region
from geosave.geodata.sampling import find_chip_span
from geosave.ml.chain.step import Step, type_name

if TYPE_CHECKING:
    from torch import nn
    from upath import UPath

    from geosave.geodata.io.storage import Location
    from geosave.ml.processing.raster.ingestion.base import RasterIngestor
    from geosave.ml.processing.raster.postprocessing.base import RasterPostprocessor
    from geosave.ml.processing.raster.prediction.base import RasterPredictor
    from geosave.ml.processing.raster.preprocessing.base import RasterPreprocessor

# Values a pipeline supplies itself; every other value a sub-processor takes
# is one an earlier sub-processor returned.
SUPPLIED = ("rasters", "model")


class RasterPipeline:
    """Orchestrate inference: raw rasters in, the model's product out.

        rasters ──preprocessor──▶ sample ──predictor (model)──▶ output
            ──postprocessor (output, sample)──▶ product

    Each step is a sub-processor called directly, and can be exchanged for
    another of its role. Its `__call__` declares with `@chain_step()` the
    values it takes and returns by name; the pipeline refuses sub-processors
    whose values do not connect, and checks every value's type as it is
    handed over. `predict` holds the whole product in memory;
    `predict_to_zarr` writes a raster too large for that chunk by chunk.

    Args:
        model: Model the predictor calls with its inputs by name.
        preprocessor: Builds the model-ready sample as training did.
        predictor: Runs the model over a sample of any size.
        postprocessor: Turns the model's output into the product.
        ingestor: Loads the raw rasters this pipeline accepts; the pipeline
            carries it and never runs it. None for a pipeline whose caller
            supplies every raster.

    Raises:
        TypeError: A sub-processor's `__call__` declares no `@chain_step()`,
            takes a value no earlier step returns, or takes it as another
            type than it is returned as.

    Examples:
        >>> pipeline = RasterPipeline(
        ...     model,
        ...     preprocessor=DensePreprocessor(bands=["B04", "B08"]),
        ...     predictor=DensePredictor(size=256),
        ...     postprocessor=SegmentationPostprocessor(["land", "water"], ["#397d49", "#419bdf"]),
        ...     ingestor=Sentinel2Ingestor(bands=["B04", "B08"]),
        ... )
        >>> rasters = pipeline.ingestor(region)[0]
        >>> pipeline.predict(rasters)["class_id"].dtype
        dtype('uint8')
        >>> pipeline.predict_to_zarr(rasters, "product.zarr", chunks=2048)
        UPath('/abs/product.zarr')
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: RasterPreprocessor,
        predictor: RasterPredictor,
        postprocessor: RasterPostprocessor,
        ingestor: RasterIngestor | None = None,
    ) -> None:
        check_steps(preprocessor, predictor, postprocessor)

        self.model = model
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.postprocessor = postprocessor
        self.ingestor = ingestor

    def predict(self, rasters: xr.DataTree) -> Any:
        """Predict the raw rasters of one acquisition in memory.

        Memory holds the model's output for the whole raster; write a larger
        raster with `predict_to_zarr`.

        Args:
            rasters: One node per raw raster, as the ingestor returns them.

        Returns:
            The postprocessor's product: for a per-pixel task a Dataset on
            the sample's grid, with the sample's scalar coordinates.

        Raises:
            TypeError: A sub-processor returned another type than it declares.
        """
        sample = self.preprocess(rasters)
        return self.predict_sample(sample)

    def predict_to_zarr(
        self, rasters: xr.DataTree, destination: Location, chunks: int = 2048
    ) -> UPath:
        """Predict the raw rasters of one acquisition into a Zarr store, chunk by chunk.

        Each chunk is read with a halo: the pixels of every chip that touches
        it. The chunk is predicted with its halo and written without it, so
        the store holds what `predict` returns while memory holds one
        chunk's model output.

        Args:
            rasters: One node per raw raster, lazy, as the ingestor
                returns them.
            destination: Store to create, ending in `.zarr`.
            chunks: Height and width in pixels of a chunk, also the store's
                chunk, rounded down to a multiple of the predictor's stride;
                choose the largest whose model output fits memory.

        Returns:
            Path of the store.

        Raises:
            FileExistsError: The destination exists.
            TypeError: The postprocessor's product is not a raster Dataset.
        """
        sample = self.preprocess(rasters)
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
                product = self.predict_sample(chunk)
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

    def preprocess(self, rasters: xr.DataTree) -> xr.DataTree:
        """Build the lazy model-ready sample of one acquisition's raw rasters."""
        values = run_step(self.preprocessor, {"rasters": rasters})
        return cast("xr.DataTree", values["sample"])

    def predict_sample(self, sample: xr.DataTree) -> Any:
        """Predict and decode one preprocessed sample whole."""
        values: dict[str, object] = {"model": self.model, "sample": sample}
        values.update(run_step(self.predictor, values))
        values.update(run_step(self.postprocessor, values))
        return values["product"]


def find_step(processor: object) -> Step:
    """Return what a sub-processor's `__call__` declares it takes and returns.

    Raises:
        TypeError: Its `__call__` carries no `@chain_step()`.
    """
    step = getattr(type(processor).__call__, "_chain_step", None)
    if not isinstance(step, Step):
        raise TypeError(
            f"{type(processor).__name__}.__call__ declares no step; decorate it "
            f"with @chain_step()"
        )
    return step


def run_step(processor: object, values: Mapping[str, object]) -> dict[str, object]:
    """Call a sub-processor with the values its `__call__` takes by name.

    Args:
        processor: Sub-processor whose `__call__` declares a step.
        values: Values available so far, by name.

    Returns:
        The values it returned, by the names its `__call__` declares.

    Raises:
        KeyError: A value it takes is absent.
        TypeError: A value it takes or returns has another type than declared.
    """
    step = find_step(processor)
    return cast("dict[str, object]", step.invoke(processor, values))


def check_steps(
    preprocessor: RasterPreprocessor,
    predictor: RasterPredictor,
    postprocessor: RasterPostprocessor,
) -> None:
    """Refuse sub-processors whose declared values do not connect in order.

    Raises:
        TypeError: A step takes a value no earlier step returns, or takes it
            as another type than it is returned as; the preprocessor returns
            no `sample`; or the postprocessor returns no `product`.
    """
    returned: dict[str, type] = {}
    for processor in (preprocessor, predictor, postprocessor):
        step = find_step(processor)
        owner = type(processor).__name__

        for name, kind in step.inputs.items():
            if name in SUPPLIED:
                continue
            if name not in returned:
                raise TypeError(
                    f"{owner} takes {name!r}, which no earlier step returns; "
                    f"returned so far: {list(returned)}"
                )
            if returned[name] != kind:
                raise TypeError(
                    f"{owner} takes {name!r} as {type_name(kind)}, but it is "
                    f"returned as {type_name(returned[name])}"
                )

        if processor is preprocessor and "sample" not in step.outputs:
            raise TypeError(f"{owner} must return a variable named `sample`")
        returned.update(step.outputs)

    if "product" not in returned:
        owner = type(postprocessor).__name__
        raise TypeError(f"{owner} must return a variable named `product`")
