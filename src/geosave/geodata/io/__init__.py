"""Raster and vector I/O, one module per format.

Raster formats live in `io.raster` and vector formats in `io.vector`; both are
also reachable from here. `storage` resolves locations and writes single
files, and `openers` picks a format by suffix.

Format modules have an `open_raster`; writable formats also have a `write`,
except GeoTIFF (`write_cog`, `write_gtiff`). Every raster writer returns the
`RasterFile` it wrote. `layout` arranges one raster as the files of several
`Scene`s and writes them through any writer; `assemble` reads them back.
Reach for them module-qualified:

Examples:
    >>> from geosave.geodata import io
    >>> raster = io.zarr.open_raster("scene.zarr", mask_and_scale=True)
    >>> io.geotiff.write_cog(raster, "scene.tif")
    >>> io.open_raster("scene.zarr")  # by suffix, when the format is not known

The fluent `to_cog`, `to_zarr`, and `to_netcdf` names belong to the `gs`
accessors, not here.
"""

from . import raster, storage, vector
from .raster import assemble, gdal, geotiff, layout, netcdf, zarr
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
from .openers import open_raster, open_tree, open_vector

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
    "assemble",
    "gdal",
    "geojson",
    "geopackage",
    "geoparquet",
    "geotiff",
    "layout",
    "netcdf",
    "raster",
    "open_raster",
    "open_tree",
    "open_vector",
    "storage",
    "vector",
    "zarr",
]
