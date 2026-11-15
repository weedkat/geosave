"""Make a processor's activity visible in Prefect without changing how it is called."""

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from geosave.ml.processing.processor import Processor


def monitor(processor: Processor, ingest_retries: int = 3) -> Processor:
    """Turn a processor's entry points into flows and its steps into tasks, in place.

        predict_region, predict_path, predict_raster   flow runs
        ingest, predict, postprocess                   task runs inside them

    Preprocessing is lazy and normalization runs per batch inside `predict`,
    so neither is a task of its own. Rasters stay in memory between tasks, so
    no task result is cached or persisted.

    Args:
        processor: Processor to monitor; its methods are replaced on this object.
        ingest_retries: Times a failed ingest is tried again.

    Returns:
        The same processor.

    Examples:
        >>> processor = monitor(load_processor("org/dynamic-world"))
        >>> processor.predict_region(region)
        [<xarray.Dataset> ...]
    """
    processor.ingestor.ingest = task(
        processor.ingestor.ingest,
        name="ingest",
        retries=ingest_retries,
        retry_delay_seconds=30,
        tags=["network"],
        cache_policy=NO_CACHE,
    )
    processor.predictor.predict = task(
        processor.predictor.predict,
        name="predict",
        tags=["gpu"],
        cache_policy=NO_CACHE,
    )
    processor.postprocessor.postprocess = task(
        processor.postprocessor.postprocess,
        name="postprocess",
        cache_policy=NO_CACHE,
    )

    # Regions and rasters are not JSON, so Prefect must not validate or coerce them.
    processor.predict_raster = flow(
        processor.predict_raster, name="predict_raster", validate_parameters=False
    )
    processor.predict_path = flow(
        processor.predict_path, name="predict_path", validate_parameters=False
    )
    processor.predict_region = flow(
        processor.predict_region, name="predict_region", validate_parameters=False
    )
    return processor
