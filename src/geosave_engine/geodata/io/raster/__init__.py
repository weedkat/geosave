"""Raster formats: GeoTIFF and COG, Zarr, NetCDF, and Sentinel SAFE products.

`gdal` reads any raster GDAL opens and holds its environment settings.
`geotiff` writes single files, and `cogs` arranges one raster as several of
them.
"""

from . import cogs, gdal, geotiff, netcdf, safe, zarr
from .cogs import CogFile

__all__ = ["CogFile", "cogs", "gdal", "geotiff", "netcdf", "safe", "zarr"]
