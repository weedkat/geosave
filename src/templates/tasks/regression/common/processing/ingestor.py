"""Read the Sentinel-2 source collection used by this project."""

import xarray as xr

from geosave.geodata import GeoRegion
from geosave.ml.pipeline.ingestion import Sentinel2L2AIngestor

from .preprocessing import BANDS


def ingest(region: GeoRegion) -> xr.DataTree:
    """Read reflectance observations covering the requested region and dates."""
    return Sentinel2L2AIngestor("s2", BANDS, groups={"reflectance": BANDS})(region)
