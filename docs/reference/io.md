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
Regenerate literal docstrings with `python scripts/opener_docstrings.py`;
use `--check` to detect drift.

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

## Saved raster files

Format writers return a UPath. Dataset/DataArray `to_cog` returns a list of
paths; `to_zarr` and `to_netcdf` also return lists of paths. DataTree writers return group
dictionaries. With `compute=False`, writers return one native Dask Delayed.
STAC publication requires one observation per Item. `to_catalog` writes and
describes actual files while retaining source observation properties.

The catalog table remains in memory until saved with `catalog.gs.to_geoparquet`.
Item metadata and identities are checked before writing raster assets.

::: geosave.geodata.io.stac
    options:
      members:
        - write_assets

## Path formats and raster combination

::: geosave.geodata.io.paths
    options:
      members:
        - default_path_format
        - expand_paths
        - split_by_collection
        - split_by_res
        - split_by_time
        - split_by_band

::: geosave.geodata.io.formats.geotiff
    options:
      members:
        - write_files

::: geosave.geodata.transform.combine
    options:
      members:
        - merge

## GeoTIFF and GDAL

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
        - write_header
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
        - resolve_asset_hrefs
        - parse_collections


::: geosave.geodata.io.formats._geopandas
    options:
      members:
        - unpack_engine_options

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
        - write_file

::: geosave.geodata.io.stac

## Load selected STAC Items

::: geosave.geodata.io.odc
    options:
      members: [load_dataset, load_datatree, StacLoadOptions]
