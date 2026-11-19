"""Grid requirements shared by native DataArrays and Datasets."""

from __future__ import annotations

import xarray as xr

from geosave.geodata.errors import RasterValidationError


def georeferenced(raster: xr.DataArray | xr.Dataset) -> None:
    """Require a grid ODC recognizes, with a CRS.

    Raises:
        RasterValidationError: ODC finds no grid, or the grid has no CRS.
        NotImplementedError: Pixels are placed by ground control points.
    """
    geobox = raster.gs.geobox
    if geobox is None:
        raise RasterValidationError(
            "ODC finds no grid. Name the spatial dimensions as ODC does, "
            "y/x or latitude/longitude, with their coordinates."
        )

    if geobox.crs is None:
        raise RasterValidationError(
            "The coordinate reference system (CRS) is missing. Declare the "
            "CRS of the existing coordinates with raster.odc.assign_crs(crs)."
        )
