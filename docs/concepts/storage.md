# Storage and file layouts

See [Read and save data](../basics/io.md) for working round trips and
[writer return values](../reference/io.md#saved-raster-files) for exact output types.

## Recognized suffixes

| Reader | Recognized suffixes |
| --- | --- |
| Raster | `.tif`, `.tiff`, `.jp2`, `.png`, `.zarr`, `.nc`, `.nc4`, `.cdf`, `.safe` |
| Vector | `.parquet`, `.geoparquet`, `.geojson`, `.json`, `.gpkg` |

Suffix matching is case-insensitive. The whole-product SAFE reader is
unavailable; open individual `.jp2` files instead. A PNG can open without
georeferencing, but spatial operations still require a locatable grid.
Unsupported suffixes raise ValueError; use a format reader directly when needed.

## One image file or a tree of assets

A DataArray writes one explicit TIFF path:

```python title="Adaptation example — use one selected observation"
path = scene["red"].isel(time=0).gs.to_cog("output/red.tif")
path = scene[["red", "nir"]].isel(time=0).gs.to_array().gs.to_cog("output/image.tif")
```

One TIFF holds one instant on one grid. A band array needs compatible pixel
meaning across its bands. For different band metadata or several dates, use
a STAC tree of separate assets:

```python title="Adaptation example — use a Dataset with a time dimension"
tree = observations.gs.expand(id_format="{start:%Y%m%d}")
files = tree.gs.to_cog("output/assets")
```

The tree names the files: `/default/20250101/red` becomes
`output/assets/default/20250101/red.tif`. `id_format` controls Item names;
Collection and asset names come from the tree. The writer does not accept
file path templates. Returned paths are the files written by that call.
`tree.gs.to_catalog(...)` writes the same assets and returns their Item table.
See [catalog publication](../guides/catalogs-and-cuts.md).

## Multidimensional stores

Zarr and NetCDF write a Dataset or DataTree as one destination. Trees retain
one group per node; they are not split into separate files. `gs.open_raster`
returns a DataTree when the store has child groups and a Dataset otherwise.
Ordinary xarray trees can have arbitrary named groups; GeoSave's STAC combine
and asset-publication methods require the collection/Item/asset layout.

NetCDF stores observation cells as a CF bounds variable; readers restore
`start_datetime` and `end_datetime`. Zarr stores those coordinates directly.
Neither format chooses the analysis grid or prepares model inputs.

## Use remote storage

These examples require a real bucket and access credentials. They are setup
examples, with no captured output.

<!-- CREDENTIALS: S3 bucket and access keys -->
```python
import os

options = {
    "key": os.environ["AWS_ACCESS_KEY_ID"],
    "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
}
store = scene.gs.to_zarr("s3://my-bucket/scene.zarr", storage_options=options)
remote = gs.open_raster(store, storage_options=options, chunks={})
```

Zarr and GeoParquet use filesystem-backed access. COG, NetCDF, GeoJSON, and
GeoPackage writes stage a local temporary file before uploading. Remote NetCDF,
GeoJSON, and GeoPackage reads download a temporary copy; NetCDF retains it until
the returned object closes.

For remote TIFFs, supplying `storage_options` currently downloads the whole
file to temporary storage before opening it. The resulting pixels can be lazy,
but that does not make the download a windowed read. Without these options,
Rasterio uses GDAL's URL handling and configured environment, which can read
COG ranges. Choose that path when bounded remote reads matter:

<!-- CREDENTIALS: AWS S3 bucket and access keys -->
```python
gs.configure_gdal(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_default_region="us-east-1",
)
remote = gs.open_raster("s3://my-bucket/scene.tif", chunks={})
```

For Hugging Face **bucket** TIFFs, the S3 gateway needs S3 keys and
`aws_s3_endpoint="s3.hf.co", aws_virtual_hosting=False`. The endpoint setting
is process-wide; finish pending reads before changing it.

Reader options are forwarded to the selected format module. See the
[I/O reference](../reference/io.md) for chunking, format options, and remote
configuration.

Close file-backed objects when their dependent work finishes.
