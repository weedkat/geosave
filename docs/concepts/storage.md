# Storage and file layouts

See [Read and save data](../basics/io.md) for working round trips and
[writer return values](../reference/io.md#saved-raster-files) for exact output types.

## Recognized suffixes

| Reader | Recognized suffixes |
| --- | --- |
| Raster | `.tif`, `.tiff`, `.jp2`, `.png`, `.zarr`, `.nc`, `.nc4`, `.cdf` |
| Vector | `.parquet`, `.geoparquet`, `.geojson`, `.json`, `.gpkg` |

Suffix matching is case-insensitive; a URL's query string is ignored. For a
Sentinel-2 SAFE product, open its individual `.jp2` files. A PNG can open
without georeferencing, but spatial operations still require a locatable grid.
Unsupported suffixes raise ValueError; use a format reader directly when needed.

## One image file or a tree of assets

Select one instant on one grid, then write named bands to one TIFF:

```python title="Adaptation example — use one selected observation"
from pathlib import Path

from geosave.geodata import io

path = Path("output/image.tif")
path.parent.mkdir(parents=True, exist_ok=True)
io.write_cog(scene[["red", "nir"]].isel(time=0), path)
```

`io.write_cog` keeps each variable's name and value metadata as a band. The
bands of one file need compatible pixel meaning. `scene["red"].gs.to_cog(path)`
writes a single DataArray through native rioxarray. For different band
metadata or several dates, use a staged STAC publication of separate assets:

```python title="Adaptation example — use a Dataset with a time dimension"
from geosave.geodata import stac

staged = stac.RasterStac.from_dataset(observations, id_format="{start:%Y%m%d}")
items = staged.to_cog("output/assets")
```

An asset at `/20250101/red` becomes `output/assets/20250101/red.tif`.
`id_format` controls Item names; variable names identify assets. `to_cog`
returns native Items describing the written files. For an Item table in one
step, use `observations.gs.to_catalog("output/assets")` and save it separately
with `.gs.to_geoparquet`. See [catalog publication](../advanced/raster-catalog.md).

## Multidimensional stores

Zarr and NetCDF write a Dataset or DataTree as one destination. Trees retain
one group per node; they are not split into separate files. `gs.open_raster`
returns a Dataset for the root group, or the group selected by `group=`.
`gs.open_raster_tree` returns the complete hierarchy as a DataTree, even for
a root-only store. Neither opener merges groups or selects a child implicitly.
Ordinary xarray trees have caller-defined groups. `RasterStac` owns the
Item/asset layout used for COG publication separately.

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
GeoPackage writes stage a local temporary file before uploading. NetCDF reads
keep a filesystem file handle open until the returned object closes.
GeoJSON and GeoPackage readers pass file handles to GeoPandas; buffering and
transferred bytes depend on the filesystem and native backend.

For remote TIFFs, filesystem options use Rasterio's filesystem opener.
Without those options, supported URLs use GDAL's URL handling and configured
environment. Both can read windows; actual transferred bytes depend on the
file layout and storage backend. Configure GDAL credentials when using its URL path:

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
