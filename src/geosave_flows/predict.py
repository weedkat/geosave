"""Predict a region with a released pipeline, as a Prefect flow of monitored tasks."""

from __future__ import annotations

import xarray as xr
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from upath import UPath

from geosave.geodata import GeoRegion
from geosave.geodata.io import storage
from geosave.ml.processing.raster.pipeline import RasterPipeline
from geosave.ml.release import load_pipeline
from geosave_flows.configs import RegionConfig

# Rasters and pipelines stay in memory between tasks, so no task result is
# cached or persisted.


@task(retries=3, retry_delay_seconds=30, tags=["network"], cache_policy=NO_CACHE)
def ingest(pipeline: RasterPipeline, region: GeoRegion) -> list[xr.DataTree]:
    """Load the region's acquisitions, trying a failed load again.

    Returns:
        One lazy tree of raw rasters per acquisition, as the pipeline's
        ingestor returns them.

    Raises:
        ValueError: The pipeline has no ingestor.
    """
    if pipeline.ingestor is None:
        raise ValueError("the pipeline has no ingestor; its release names none")
    return pipeline.ingestor(region)


@task(tags=["gpu"], cache_policy=NO_CACHE)
def predict_to_zarr(
    pipeline: RasterPipeline, rasters: xr.DataTree, destination: UPath, chunks: int
) -> UPath:
    """Predict one acquisition into a Zarr store, as `RasterPipeline.predict_to_zarr` does."""
    return pipeline.predict_to_zarr(rasters, destination, chunks=chunks)


@flow
def predict_region(
    repo: str, region: RegionConfig, root: str, chunks: int = 2048
) -> list[UPath]:
    """Predict every acquisition over a region into one Zarr store each.

    Every argument is JSON, so the flow deploys and schedules as it is called.
    A flow run records one `ingest` task, retried on failure, and one
    `predict_to_zarr` task per acquisition.

    Args:
        repo: Hub repository id or local folder of a release with an ingestor.
        region: Place and timespan to predict.
        root: Folder or URL prefix the stores are written under.
        chunks: Height and width in pixels of a predicted chunk.

    Returns:
        One store per acquisition, at `<root>/<acquisition start>.zarr`.

    Raises:
        ValueError: The release has no ingestor, or no acquisition covers
            the region in its timespan.

    Examples:
        >>> predict_region(
        ...     repo="org/dynamic-world",
        ...     region={"kind": "coordinates", "latitude": -6.2, "longitude": 106.8,
        ...             "shape": 1024, "resolution": 10, "timespan": "2024-07"},
        ...     root="s3://bucket/land-cover",
        ... )
        [S3Path('s3://bucket/land-cover/20240703T030000.zarr'), ...]
        >>> predict_region.serve(name="land-cover", cron="0 6 * * *", parameters={...})
    """
    pipeline = load_pipeline(repo)
    area, _ = region.open()
    folder = storage.create_upath(root)

    stores = []
    for rasters in ingest(pipeline, area):
        start, _ = rasters.gs.region.timespan
        destination = folder / f"{start:%Y%m%dT%H%M%S}.zarr"
        stores.append(predict_to_zarr(pipeline, rasters, destination, chunks))
    return stores
