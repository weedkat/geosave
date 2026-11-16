# Readers, writers, and format options

Use `gs.open_raster`, `gs.open_raster_tree`, and `gs.open_vector` to select a
reader from a path. Use a format module when you need format-specific options.
See [Read and save data](../basics/io.md) for complete examples.

`open_raster` always returns a Dataset. Zarr and NetCDF open their root group
by default; `group="scene"` selects one group without opening its children.
`open_raster_tree` reads a whole Zarr or NetCDF store as a DataTree, including
root-only stores. GeoTIFF and other banded formats use `open_raster`.
Group selection follows native xarray semantics. To include coordinates inherited
from parent nodes, open the tree and use `to_dataset(inherit="all_coords")`.

```python
raster = gs.open_raster("archive.nc", group='scene')
pixels = raster["red"].compute()

tree = gs.open_raster_tree("archive.nc")
pixels = tree["scene"].to_dataset(inherit="all_coords")["red"].compute()
```

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
        - open_raster_tree
        - open_vector

## STAC Item pixels

Use `stac.load_dataset(items, geobox=grid)` for one collection on a chosen grid,
or `stac.load_native_datasets(items)` for one Dataset per native asset grid, which requires asset Projection metadata.
See [STAC loading](stac.md) and [catalog operations](catalog.md).

## Saved raster files

| Object and call | Default return | With `compute=False` |
| --- | --- | --- |
| DataArray `.gs.to_cog(path)` or `.gs.to_gtiff(path)` | Native rioxarray result (normally `None`) | Native Dask result with chunked data and `lock=True` |
| `io.write_cog(dataset, path)` or `io.write_geotiff(dataset, path)` | One file `UPath` | Delayed yielding that UPath |
| Dataset, DataArray, DataTree `.gs.to_zarr(path)` | One store `UPath` | Delayed yielding that UPath |
| Dataset, DataArray, DataTree `.gs.to_netcdf(path)` | One file `UPath` | Delayed yielding that UPath |
| `Dataset.gs.to_catalog(folder)` | Item GeoDataFrame; assets written | Not supported |
| `RasterStac.to_cog(folder)` | Native Items; assets written | Not supported |

Dataset `.gs.to_cog()` is unavailable. Use `io.write_cog(dataset, path)` for
one file preserving variable names and band metadata. For dated assets use
`Dataset.gs.to_catalog`; `RasterStac` stages assets on several grids. Asset paths
follow `folder/Item/asset.tif`. Zarr and NetCDF retain a whole hierarchy in one store.

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
A local destination is written in place; any other is staged as a local file
and uploaded when the write finishes.

::: geosave.geodata.io.raster.geotiff
    options:
      members:
        - open_geotiff
        - GeoTIFFOpenOptions
        - write_cog
        - write_geotiff
        - GeoTIFFWriteOptions
        - COGWriteOptions
        - GTiffWriteOptions
        - read_gdal_address
        - hf_to_s3

`configure_gdal` remains available as `geosave.configure_gdal` and
`geosave.geodata.configure_gdal`. Call it at startup to set the process-wide
GDAL environment; its explicit options document the corresponding variables.

::: geosave.geodata.io.gdal_env
    options:
      members:
        - configure_gdal

## Zarr

::: geosave.geodata.io.raster.zarr
    options:
      members:
        - open_zarr
        - open_zarr_tree
        - write_zarr
        - write_zarr_tree
        - create_zarr
        - write_zarr_region
        - ZarrOpenOptions
        - ZarrWriteOptions
        - ZarrChunkSpec

## NetCDF

::: geosave.geodata.io.raster.netcdf
    options:
      members:
        - open_netcdf
        - open_netcdf_tree
        - write_netcdf
        - write_netcdf_tree
        - NetCDFOpenOptions
        - NetCDFWriteOptions
        - NetCDFChunkSpec

## Codecs

A format lists the codecs that translate GeoSave's in-memory conventions to
its own; writing encodes in order and reading decodes in reverse, so a file
reads back as the raster written.

::: geosave.geodata.io.raster.codecs
    options:
      members:
        - Codec
        - encode_dataset
        - decode_dataset
        - encode_tree
        - decode_tree

## GeoParquet

::: geosave.geodata.io.vector.geoparquet
    options:
      members:
        - open_geoparquet
        - write_geoparquet
        - GeoParquetOpenOptions
        - GeoParquetWriteOptions

## GeoJSON and GeoPackage

A GeoPackage keeps the frame's CRS and stores its `attrs` as layer metadata.
GeoJSON is WGS84 by specification and has no slot for `attrs`.

::: geosave.geodata.io.vector.ogr
    options:
      members:
        - open_geojson
        - write_geojson
        - open_geopackage
        - write_geopackage
        - OGROpenOptions
        - OGRWriteOptions

## STAC GeoParquet

::: geosave.geodata.io.vector.stac_geoparquet
    options:
      members:
        - write_stac_geoparquet
        - parse_collections

## Locations and storage

::: geosave.geodata.io.storage
    options:
      members:
        - Location
        - StorageOptions
        - create_upath
        - check_destination
        - refuse_existing
        - stage_file
        - defer

## Load selected STAC Items

::: geosave.geodata.stac.load
    options:
      members: [load_dataset, load_native_datasets, group_bands_by_grid, StacLoadOptions]
