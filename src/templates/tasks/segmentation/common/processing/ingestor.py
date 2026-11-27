"""Read the project's imagery for a prediction region."""

import xarray as xr

from geosave import GeoRegion


def ingest(region: GeoRegion) -> xr.DataTree:
    """Return one observation under the image node, using the project's date policy."""
    raise NotImplementedError("Select an image source and observation for the region.")
