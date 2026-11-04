"""Ingest the rasters a model declares on an explicit anchor, as one catalog."""

from prefect import flow

from geosave.geodata.io import storage
from geosave.model.spec import ModelSpec
from geosave_flows.configs import AnchorConfig


@flow(name="ingest", persist_result=False)
def ingest(anchor: AnchorConfig, *, output: str, spec: str) -> str:
    """Load model rasters on one anchor, save them as published, and list them.

    Each raster is a COG per instant under `output/<raster>/`, and every one
    is an Item of `output/catalog.parquet`, the catalog `ModelSpec.prepare`
    takes.

    Args:
        anchor: Anchor configuration defining grid and time.
        output: New folder, as a path or URL.
        spec: Model specification defining required rasters and STAC recipes.

    Returns:
        Path to the written catalog.

    Raises:
        FileExistsError: If the output folder already exists.
        ValueError: If a raster declares no `stac` block or fails validation.
    """
    folder = storage.location(output)
    if folder.exists():
        raise FileExistsError(f"{folder} exists; ingest writes a new folder")
    model = ModelSpec.load(spec)
    region, grid = anchor.open()
    catalog = model.ingest(
        region, folder, geobox=grid, path_format="{collection}/{time:%Y%m%dT%H%M%S}.tif"
    )
    path = catalog.gs.to_geoparquet(folder / "catalog.parquet")
    return str(path)
