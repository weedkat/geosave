# Readers, writers, and format options

Use `gs.open_raster`, `gs.open_tree`, and `gs.open_vector` to select a reader
from a path. Use a format module when you need format-specific options.
See [Read and save data](../basics/io.md) for complete examples.

## Forwarded options

Every key in a format's options TypedDict is optional. Pass selected keys as
keyword arguments; the options class lists their supported names and types.
Omitted keys use the GeoSave defaults described in the entry, then the native
backend's defaults. For example, GeoSave's Zarr reader defaults `consolidated`
to `False`, while CF decoding and store layout follow its raster contract.

Use the backend references for option meanings, native defaults, and encoding
details. GeoSave's options classes describe the subset accepted by its readers
and writers.

| Format | Read options | Write options |
| --- | --- | --- |
| GDAL rasters / GeoTIFF | [rioxarray.open_rasterio](https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.open_rasterio), [rasterio.open](https://rasterio.readthedocs.io/en/stable/api/rasterio.html#rasterio.open) | [GDAL COG creation options](https://gdal.org/en/stable/drivers/raster/cog.html#creation-options), [GDAL GeoTIFF creation options](https://gdal.org/en/stable/drivers/raster/gtiff.html#creation-options) |
| Zarr | [xarray.open_zarr](https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html) | [Dataset.to_zarr](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_zarr.html) |
| NetCDF | [xarray.open_dataset](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html) | [Dataset.to_netcdf](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html) |
| GeoJSON / GeoPackage | [GeoPandas.read_file](https://geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html) | [GeoDataFrame.to_file](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_file.html) |
| GeoParquet | [GeoPandas.read_parquet](https://geopandas.org/en/stable/docs/reference/api/geopandas.read_parquet.html), [PyArrow read_table](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html) | [GeoDataFrame.to_parquet](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_parquet.html), [PyArrow write_table](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html) |

## Reader selection

::: geosave.geodata.io
    options:
      members:
        - open_raster
        - open_tree
        - open_vector

## Saved raster files

Writers return `Scene`s: the files of one collection at one instant, keyed by
asset, each a `RasterFile` carrying its path and pixel-free header. A scene is
one observation; native Items load real files through `io.raster.stac.open_raster`.

::: geosave.geodata.io.raster.file.Scene

::: geosave.geodata.io.raster.file.RasterFile

## Layout, writing and reading

::: geosave.geodata.io.raster.layout
    options:
      members:
        - layout
        - save

::: geosave.geodata.io.raster.assemble
    options:
      members:
        - open_raster

## GeoTIFF and GDAL

::: geosave.geodata.io.raster.gdal
    options:
      members:
        - open_raster
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

::: geosave.geodata.attrs.groups.tiff.GeoTIFFTags

## Zarr

::: geosave.geodata.io.raster.zarr
    options:
      members:
        - open_raster
        - write
        - ZarrOpenOptions
        - ZarrWriteOptions
        - ZarrChunkSpec
        - fill_encoding

## NetCDF

::: geosave.geodata.io.raster.netcdf
    options:
      members:
        - open_raster
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
        - open_raster
        - open_tree

## GeoParquet

::: geosave.geodata.io.vector.geoparquet
    options:
      members:
        - open_vector
        - write
        - GeoParquetOpenOptions
        - GeoParquetWriteOptions

## GeoJSON

::: geosave.geodata.io.vector.geojson
    options:
      members:
        - open_vector
        - write
        - GeoJSONOpenOptions
        - GeoJSONWriteOptions
        - driver_read_options

## GeoPackage

::: geosave.geodata.io.vector.geopackage
    options:
      members:
        - open_vector
        - write
        - GeoPackageOpenOptions
        - GeoPackageWriteOptions

## STAC GeoParquet

::: geosave.geodata.io.vector.stac_geoparquet
    options:
      members:
        - write
        - read_collections
        - resolve_asset_hrefs
        - parse_collections

## Static STAC JSON

::: geosave.geodata.io.vector.stac_json
    options:
      members:
        - write

## Locations and storage

::: geosave.geodata.io.storage
    options:
      members:
        - Location
        - StorageOptions
        - location
        - is_local
        - local_path
        - absolute_location
        - write

::: geosave.geodata.io.raster.stac
