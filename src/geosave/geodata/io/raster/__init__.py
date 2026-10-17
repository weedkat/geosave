"""Raster formats: GeoTIFF and COG, Zarr, NetCDF.

`gdal` reads any raster GDAL opens and holds its environment settings.
`geotiff`, `zarr` and `netcdf` each write one file and return it as a
`RasterFile`. `layout` arranges one raster as the files of several `Scene`s
and writes them through any of those writers; `assemble` reads the scenes
back as one raster.
"""

from . import assemble, gdal, geotiff, layout, netcdf, zarr

__all__ = ["assemble", "gdal", "geotiff", "layout", "netcdf", "zarr"]
