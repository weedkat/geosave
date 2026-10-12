"""Raster and vector I/O, one module per format.

Raster formats live in `io.raster` and vector formats in `io.vector`; both are
also reachable from here. `storage` resolves locations and writes single
files, and `readers` picks a format by suffix.

Format modules have a `read`; writable formats also have a `write`, except GeoTIFF
(`write_cog`, `write_gtiff`) and `cogs`, which arranges one raster as several
GeoTIFFs that `open_raster` reads back. Reach for them module-qualified:

Examples:
    >>> from geosave.geodata import io
    >>> raster = io.zarr.read("scene.zarr", mask_and_scale=True)
    >>> io.geotiff.write_cog(raster, "scene.tif")
    >>> io.open_raster("scene.zarr")  # by suffix, when the format is not known

The fluent `to_cog`, `to_zarr`, and `to_netcdf` names belong to the `gs`
accessors, not here.
"""

from . import raster, storage, vector
from .raster import cogs, gdal, geotiff, netcdf, zarr
from .vector import geojson, geopackage, geoparquet
from .raster.gdal import RasterioOpenOptions, configure_gdal
from .vector.geojson import GeoJSONOpenOptions
from .vector.geopackage import GeoPackageOpenOptions
from .vector.geoparquet import GeoParquetOpenOptions
from .raster.geotiff import (
    COGWriteOptions,
    GeoTIFFTags,
    GeoTIFFWriteOptions,
    GTiffWriteOptions,
)
from .raster.netcdf import NetCDFOpenOptions, NetCDFWriteOptions
from .raster.zarr import ZarrOpenOptions, ZarrWriteOptions
from .readers import open_raster, open_stack, open_vector

__all__ = [
    "COGWriteOptions",
    "GTiffWriteOptions",
    "GeoJSONOpenOptions",
    "GeoPackageOpenOptions",
    "GeoParquetOpenOptions",
    "GeoTIFFTags",
    "GeoTIFFWriteOptions",
    "NetCDFOpenOptions",
    "NetCDFWriteOptions",
    "RasterioOpenOptions",
    "ZarrOpenOptions",
    "ZarrWriteOptions",
    "configure_gdal",
    "cogs",
    "gdal",
    "geojson",
    "geopackage",
    "geoparquet",
    "geotiff",
    "netcdf",
    "raster",
    "open_raster",
    "open_stack",
    "open_vector",
    "storage",
    "vector",
    "zarr",
]
