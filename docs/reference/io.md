# Readers, writers, and format options

Use `gs.open_raster` and `gs.open_vector` to select a reader
from a path. Use a format module when you need format-specific options.
See [Read and save data](../basics/io.md) for complete examples.

## Forwarded options

Every key in a format's options TypedDict is optional. Pass selected keys as
keyword arguments; the options class lists their supported names and types.
Omitted keys use the GeoSave defaults described in the entry, then the native
backend's defaults. For example, GeoSave's Zarr reader defaults `consolidated`
to `False`, while CF decoding and store layout follow its raster contract.

The convenience opener docstrings include every supported format option and
the storage option reference for IDE help. Backend links provide further
details about extension dictionaries and version-specific SDK options.

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
        - open_vector

## STAC Item pixels

Use `stac.load_dataset(items, grid)` for one collection on a chosen grid,
`stac.load_datatree(items)` for native asset grids, or `rows.gs.to_datatree()`
for an Item table. Native-grid loading requires asset Projection metadata.
See [STAC loading](stac.md) and [catalog operations](catalog.md).

## Saved raster files

| Object and call | Default return | With `compute=False` |
| --- | --- | --- |
| DataArray `.gs.to_cog(path)` or `.gs.to_gtiff(path)` | Native rioxarray result (normally `None`) | Native Dask result with chunked data and `lock=True` |
| DataTree `.gs.to_cog(folder)` | List of asset UPaths | Delayed yielding that list |
| Dataset, DataArray, DataTree `.gs.to_zarr(path)` | One store `UPath` | Delayed yielding that UPath |
| Dataset, DataArray, DataTree `.gs.to_netcdf(path)` | One file `UPath` | Delayed yielding that UPath |
| DataTree `.gs.to_catalog(folder)` | Item GeoDataFrame; assets written | Not supported |

Dataset `.gs.to_cog()` is unavailable. Select one band, stack compatible bands
with `.gs.to_array()`, or expand dated data to a STAC tree. Tree asset paths
follow `folder/collection/Item/asset.tif`; path templates are not accepted by
these accessors. Zarr and NetCDF retain the full hierarchy in one store.

DataArray TIFF/COG methods delegate directly to `.rio.to_raster`, selecting
`GTiff` or `COG`. They use native options, overwrite behavior and return values;
`compute=False` requires chunked data and a write lock to defer pixels, and may
create the header immediately. They do not apply GeoSave metadata translation
or fsspec staging.

GeoSave Dataset/tree writers refuse existing destinations unless
`overwrite=True`. Their deferred writes return a Dask Delayed; keep sources
open until it has computed.
A returned Item table remains in memory until `.gs.to_geoparquet(...)` saves it.
See [Read and save data](../basics/io.md) for working round trips.

## GeoTIFF and GDAL

Dataset writers map variables to TIFF bands. Each band keeps its own CF storage
encoding, description, units and tags; the file shares one grid, stored dtype
and nodata value. Dask sources are written by source chunk. COG output is made
by converting a temporary disk TIFF with GDAL, avoiding Rasterio's in-memory
COG writer.

Pass creation options directly, for example `write_cog(ds, path, compress="LZW")`.
The former nested `creation_options` dictionary is removed. GDAL interprets
these keywords; GeoSave does not normalize or validate driver options.

Filesystem-backed reads pass `UPath.fs` to Rasterio's opener and remain lazy.
GTiff writes use that opener for seekable filesystems; object-store writes
stage a local file before uploading.

::: geosave.geodata.io.rasterio
    options:
      members:
        - open_raster
        - configure_gdal
        - RasterioOpenOptions
        - hf_to_s3

::: geosave.geodata.io.formats.geotiff
    options:
      members:
        - write_cog
        - write_geotiff
        - GeoTIFFWriteOptions
        - COGWriteOptions
        - GTiffWriteOptions


## Zarr

::: geosave.geodata.io.formats.zarr
    options:
      members:
        - open_zarr
        - write_zarr
        - ZarrOpenOptions
        - ZarrWriteOptions
        - ZarrChunkSpec
        - fill_encoding

## NetCDF

::: geosave.geodata.io.formats.netcdf
    options:
      members:
        - open_netcdf
        - write_netcdf
        - NetCDFOpenOptions
        - NetCDFWriteOptions
        - NetCDFEngine
        - NetCDFChunkSpec
        - NetCDFFormat

## Vendor products

SAFE loading is not implemented.

## GeoParquet

::: geosave.geodata.io.formats.geoparquet
    options:
      members:
        - open_geoparquet
        - write_geoparquet
        - GeoParquetOpenOptions
        - GeoParquetWriteOptions

## GeoJSON

::: geosave.geodata.io.formats.geojson
    options:
      members:
        - open_geojson
        - write_geojson
        - GeoJSONOpenOptions
        - GeoJSONWriteOptions

## GeoPackage

::: geosave.geodata.io.formats.geopackage
    options:
      members:
        - open_geopackage
        - write_geopackage
        - GeoPackageOpenOptions
        - GeoPackageWriteOptions

## STAC GeoParquet

::: geosave.geodata.io.formats.stac_geoparquet
    options:
      members:
        - write_stac_geoparquet
        - read_collections
        - parse_collections


::: geosave.geodata.io.formats.utils.geopandas
    options:
      members:
        - unpack_engine_options

## Locations and storage

::: geosave.geodata.io.storage
    options:
      members:
        - Location
        - StorageOptions
        - create_upath
        - check_destination
        - stage_file


## Load selected STAC Items

::: geosave.geodata.stac.load
    options:
      members: [load_dataset, load_datatree, StacLoadOptions]
