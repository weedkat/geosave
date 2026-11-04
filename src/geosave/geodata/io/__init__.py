"""Public geospatial I/O: formats, storage, discovery-independent loading.

Use io.formats.geotiff.write_cog for a known format, open_raster for suffix
selection, and load_dataset/load_datatree for selected native STAC Items.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import formats, paths, rasterio, storage
from .formats.geojson import GeoJSONOpenOptions, open_geojson, write_geojson
from .formats.geopackage import GeoPackageOpenOptions, open_geopackage, write_geopackage
from .formats.geoparquet import GeoParquetOpenOptions, open_geoparquet, write_geoparquet
from .formats.geotiff import (
    COGWriteOptions,
    GeoTIFFWriteOptions,
    GTiffWriteOptions,
    open_geotiff,
    write_cog,
    write_geotiff,
)
from .formats.netcdf import (
    NetCDFOpenOptions,
    NetCDFWriteOptions,
    open_netcdf,
    write_netcdf,
)
from .formats.zarr import ZarrOpenOptions, ZarrWriteOptions, open_zarr, write_zarr
from .openers import open_raster, open_vector
from .rasterio import RasterioOpenOptions, configure_gdal

if TYPE_CHECKING:
    from .formats.stac_geoparquet import write_stac_geoparquet
    from .odc import StacLoadOptions, load_dataset, load_datatree


def __getattr__(name: str) -> Any:
    if name in {"StacLoadOptions", "load_dataset", "load_datatree"}:
        return getattr(import_module(f"{__name__}.odc"), name)
    if name == "write_stac_geoparquet":
        return getattr(import_module(f"{__name__}.formats.stac_geoparquet"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "open_geojson",
    "write_geojson",
    "open_geopackage",
    "write_geopackage",
    "open_geoparquet",
    "write_geoparquet",
    "open_geotiff",
    "write_geotiff",
    "open_zarr",
    "write_zarr",
    "open_netcdf",
    "write_netcdf",
    "write_cog",
    "write_stac_geoparquet",
    "StacLoadOptions",
    "load_dataset",
    "load_datatree",
    "COGWriteOptions",
    "GTiffWriteOptions",
    "GeoJSONOpenOptions",
    "GeoPackageOpenOptions",
    "GeoParquetOpenOptions",
    "GeoTIFFWriteOptions",
    "NetCDFOpenOptions",
    "NetCDFWriteOptions",
    "RasterioOpenOptions",
    "ZarrOpenOptions",
    "ZarrWriteOptions",
    "configure_gdal",
    "rasterio",
    "paths",
    "formats",
    "open_raster",
    "open_vector",
    "storage",
]
