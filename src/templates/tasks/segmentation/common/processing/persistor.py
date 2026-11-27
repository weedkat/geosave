"""Write a finished class map to its chosen destination."""

import xarray as xr

from geosave.geodata.io import write_cog
from geosave.geodata.io.storage import Location


def persist(product: xr.Dataset, destination: Location) -> None:
    """Write one georeferenced class map as a COG."""
    write_cog(product, destination, overview_resampling="nearest")
