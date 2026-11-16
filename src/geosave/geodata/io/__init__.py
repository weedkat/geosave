"""Public geospatial I/O: raster and vector formats, storage, suffix openers.

Use a format's own writer, such as `write_cog`, for a known format, and
`open_raster`, `open_raster_tree`, or `open_vector` to read by suffix.
Raster openers return a Dataset or a DataTree respectively. STAC Items load
through `geosave.geodata.stac`.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import gdal_env, raster, storage, vector
from .gdal_env import configure_gdal
from .openers import open_raster, open_raster_tree, open_vector
from .raster.geotiff import (
    COGWriteOptions,
    GeoTIFFOpenOptions,
    GeoTIFFWriteOptions,
    GTiffWriteOptions,
    open_geotiff,
    write_cog,
    write_geotiff,
)
from .raster.netcdf import (
    NetCDFOpenOptions,
    NetCDFWriteOptions,
    open_netcdf,
    open_netcdf_tree,
    write_netcdf,
    write_netcdf_tree,
)
from .raster.zarr import (
    ZarrOpenOptions,
    ZarrWriteOptions,
    create_zarr,
    open_zarr,
    open_zarr_tree,
    write_zarr,
    write_zarr_region,
    write_zarr_tree,
)
from .vector.geoparquet import (
    GeoParquetOpenOptions,
    GeoParquetWriteOptions,
    open_geoparquet,
    write_geoparquet,
)
from .vector.ogr import (
    OGROpenOptions,
    OGRWriteOptions,
    open_geojson,
    open_geopackage,
    write_geojson,
    write_geopackage,
)

if TYPE_CHECKING:
    from .vector.stac_geoparquet import write_stac_geoparquet


def __getattr__(name: str) -> Any:
    # stac_geoparquet imports the STAC package, which imports this one.
    if name == "write_stac_geoparquet":
        return getattr(import_module(f"{__name__}.vector.stac_geoparquet"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "open_raster",
    "open_raster_tree",
    "open_vector",
    "open_geotiff",
    "write_geotiff",
    "write_cog",
    "open_zarr",
    "open_zarr_tree",
    "write_zarr",
    "write_zarr_tree",
    "create_zarr",
    "write_zarr_region",
    "open_netcdf",
    "open_netcdf_tree",
    "write_netcdf",
    "write_netcdf_tree",
    "open_geojson",
    "write_geojson",
    "open_geopackage",
    "write_geopackage",
    "open_geoparquet",
    "write_geoparquet",
    "write_stac_geoparquet",
    "COGWriteOptions",
    "GTiffWriteOptions",
    "GeoTIFFOpenOptions",
    "GeoTIFFWriteOptions",
    "NetCDFOpenOptions",
    "NetCDFWriteOptions",
    "ZarrOpenOptions",
    "ZarrWriteOptions",
    "OGROpenOptions",
    "OGRWriteOptions",
    "GeoParquetOpenOptions",
    "GeoParquetWriteOptions",
    "configure_gdal",
    "gdal_env",
    "raster",
    "vector",
    "storage",
]
