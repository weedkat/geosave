# Readers, writers, and format options

Use `gs.open_raster`, `gs.open_stack`, and `gs.open_vector` to select a reader
from a path. Use a format module when you need format-specific options.
See [Read and save data](../basics/io.md) for complete examples.

## Reader selection

::: geosave.geodata.io
    options:
      members:
        - open_raster
        - open_stack
        - open_vector

## COG files

::: geosave.geodata.io.raster.cogs
    options:
      members:
        - CogFile
        - read
        - write
        - write_files
        - layout
        - fields_named
        - split_keywords

## GeoTIFF and GDAL

::: geosave.geodata.io.raster.gdal
    options:
      members:
        - read
        - configure_gdal
        - RasterioOpenOptions
        - hf_to_s3

::: geosave.geodata.io.raster.geotiff
    options:
      members:
        - write_cog
        - write_gtiff
        - write_header
        - GeoTIFFWriteOptions
        - COGWriteOptions
        - GTiffWriteOptions

::: geosave.geodata.io.raster.tags.GeoTIFFTags

## Zarr

::: geosave.geodata.io.raster.zarr
    options:
      members:
        - read
        - write
        - ZarrOpenOptions
        - ZarrWriteOptions
        - ZarrChunkSpec
        - fill_encoding

## NetCDF

::: geosave.geodata.io.raster.netcdf
    options:
      members:
        - read
        - write
        - NetCDFOpenOptions
        - NetCDFWriteOptions
        - NetCDFEngine
        - NetCDFChunkSpec
        - NetCDFFormat

## Vendor products

::: geosave.geodata.io.raster.safe
    options:
      members:
        - read
        - read_stack

## GeoParquet

::: geosave.geodata.io.vector.geoparquet
    options:
      members:
        - read
        - write
        - GeoParquetOpenOptions
        - GeoParquetWriteOptions

## GeoJSON

::: geosave.geodata.io.vector.geojson
    options:
      members:
        - read
        - write
        - GeoJSONOpenOptions
        - GeoJSONWriteOptions
        - driver_read_options

## GeoPackage

::: geosave.geodata.io.vector.geopackage
    options:
      members:
        - read
        - write
        - GeoPackageOpenOptions
        - GeoPackageWriteOptions

## STAC GeoParquet

::: geosave.geodata.io.vector.stac_geoparquet
    options:
      members:
        - from_items
        - to_items
        - write
        - get_collections
        - resolve
        - parse_collections

## Locations and storage

::: geosave.geodata.io.storage
    options:
      members:
        - Location
        - StorageOptions
        - location
        - is_local
        - local_path
        - openable
        - absolute_location
        - write
