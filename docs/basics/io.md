# Read and write data

Save a raster as a Cloud Optimized GeoTIFF (COG), then open it:

```python
import tempfile
from pathlib import Path

import numpy as np

import geosave as gs

root = Path(tempfile.mkdtemp())
anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=4, resolution=10, crs="EPSG:32633"
)
pixels = np.arange(1, 17, dtype="uint8").reshape(4, 4)
scene = gs.raster({"red": (("y", "x"), pixels)}, anchor.geobox, nodata=0)
files = scene.gs.to_cog(root / "scene.tif")
reopened = gs.open_raster(files)
reopened
```

Output:

```text
<xarray.Dataset> Size: 88B
Dimensions:      (x: 4, y: 4)
Coordinates:
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
    spatial_ref  int64 8B 0
Data variables:
    red          (y, x) uint8 16B dask.array<chunksize=(4, 4), meta=np.ndarray>
Attributes:
    AREA_OR_POINT:  Area
```

The reopened object is an xarray `Dataset`, with one variable per band.
The file preserves this raster's grid, `uint8` values, variable name, and nodata.
`to_cog` returns a list of `Scene`s, one per date. Each scene maps an asset key
to a `RasterFile`: the file's `path` and the `header` it states, its bands and
date, without pixels. A file holding one band is keyed by that band, and a file
holding several is keyed `"image"`. `gs.open_raster(scenes)` reads them back as
one raster.

Readers open data through the format named by the path suffix. Start with the
native object you need:

| Object | Reader | Writers |
| --- | --- | --- |
| Raster: xarray `Dataset` | `gs.open_raster(path)` | `.gs.to_cog`, `.gs.to_zarr`, `.gs.to_netcdf` |
| Tree: xarray `DataTree` | `gs.open_tree(paths)` | `.gs.to_cog`, `.gs.to_zarr`, `.gs.to_netcdf` |
| Vector: GeoPandas `GeoDataFrame` | `gs.open_vector(path)` | `.gs.to_geoparquet`, `.gs.to_geojson`, `.gs.to_geopackage` |

A location can be a string, `Path`, or `UPath`. Reader options such as `chunks`
and `mask_and_scale` go to the selected backend; they are not acquisition or
preprocessing instructions. Start with [existing files](../guides/ingest-files.md)
for selection, feature calculation, and saving in one workflow.

| Reader | Recognized suffixes |
| --- | --- |
| Raster | `.tif`, `.tiff`, `.jp2`, `.png`, `.zarr`, `.nc`, `.nc4`, `.cdf`, `.safe` |
| Vector | `.parquet`, `.geoparquet`, `.geojson`, `.json`, `.gpkg` |

Suffix matching is case-insensitive. `.SAFE` is a vendor product directory,
whose whole-product reader is recognized but currently raises
`NotImplementedError`; open individual `.jp2` files instead. A PNG without georeferencing can be opened,
but spatial operations still need a locatable grid. Unsupported suffixes raise
`ValueError`; use the format's reader directly when suffix dispatch is unsuitable.

## Choose a format

Use **COG** for geospatial image files and remote reads of selected areas.
Use **Zarr** for multidimensional arrays stored in chunks.
Use **NetCDF** for multidimensional scientific data in one file.
Use **GeoParquet** for vector tables and raster item catalogs.

```python
store = scene.gs.to_zarr(root / "scene.zarr")
zarr_scene = gs.open_raster(store, chunks={})
zarr_scene
```

Output:

```text
<xarray.Dataset> Size: 84B
Dimensions:      (y: 4, x: 4)
Coordinates:
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B ...
Data variables:
    red          (y, x) uint8 16B dask.array<chunksize=(4, 4), meta=np.ndarray>
```

```python
path = scene.gs.to_netcdf(root / "scene.nc")
netcdf_scene = gs.open_raster(path)
netcdf_scene
```

Output:

```text
<xarray.Dataset> Size: 84B
Dimensions:      (y: 4, x: 4)
Coordinates:
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B ...
Data variables:
    red          (y, x) uint8 16B ...
```

Zarr and NetCDF writers return a destination `UPath`, which readers accept.
They preserve variable order. Opening with `chunks={}` requests Dask arrays
using the backend's preferred chunks; `chunks=None` uses the backend's lazy
array access. GeoTIFFs default to `chunks={}`, so arithmetic on them stays
lazy. A dated GeoTIFF carries its date as a scalar `time` coordinate.
Writing normally computes the data needed for the
output. See [lazy data](../concepts/lazy-data.md) for when pixels are read.

## Save a time series as image files

```python
dated = scene.expand_dims(time=[np.datetime64("2025-01-01")])
items = dated.gs.to_cog(root / "dated/scene_{time:%Y%m%d}.tif")
[file.name for item in items for file in item.values()]
```

Output:

```text
['scene_20250101.tif']
```

The path template determines how data splits into files. `{time}` writes a
file per date; `{band}` writes a file per band. Bands sharing a file must have
the same dtype and nodata. For a tree, include `{collection}` to name each
group. Other fields such as `{prefix}` take caller-supplied keyword values;
callable values receive each date's raster. A scalar `time` coordinate is
accepted for one image. Several dates need distinct paths, or the writer
raises before writing them.

Existing destinations raise `FileExistsError` by default. Pass `overwrite=True`
to replace them. The returned paths are the files this call wrote, not an
inventory of the destination folder.
Build [STAC Items](../guides/stac.md) from them with `gs.stac.create_items`
when you need a searchable catalog.

## Read a tree

```python
sample = gs.open_tree({"optical": files[0], "label": store})
sample
```

Output:

```text
<xarray.DataTree>
Group: /
│   Dimensions:      (y: 4, x: 4)
│   Coordinates:
│     * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
│     * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
│       spatial_ref  int64 8B 0
├── Group: /optical
│       Dimensions:      (y: 4, x: 4)
│       Coordinates:
│           spatial_ref  int64 8B 0
│           time         datetime64[s] 8B 2025-01-01
│       Data variables:
│           red          (y, x) uint8 16B dask.array<chunksize=(4, 4), meta=np.ndarray>
│       Attributes:
│           TIFFTAG_DATETIME:  2025:01:01 00:00:00
│           GEOSAVE_DATETIME:  2025-01-01
│           AREA_OR_POINT:     Area
└── Group: /label
        Dimensions:      (y: 4, x: 4)
        Coordinates:
            spatial_ref  int32 4B ...
        Data variables:
            red          (y, x) uint8 16B ...
```

The mapping names the groups; each value is a raster file or store. This example
uses the same pixels twice to show the structure. You can also open a folder
containing one raster file or store per group, or the paths a tree's COG
writer returns. A tree saved with `.gs.to_zarr(folder)` is a folder with one
Zarr store per group. A folder names groups by file/store stem, in sorted name
order; a mapping keeps your names and order. Two folder entries with the same
stem are ambiguous and raise `ValueError`.

Close file-backed objects when you finish:

```python
sample.close()
reopened.close()
zarr_scene.close()
netcdf_scene.close()
```

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

Zarr and GeoParquet use fsspec for remote access. COG, NetCDF, GeoJSON, and
GeoPackage writes use a local temporary file before uploading. NetCDF reads
require local files. GeoJSON and GeoPackage readers can use URLs their GDAL
driver supports, with GDAL credentials.

For S3 TIFFs, `storage_options` can supply `key`, `secret`, `token`, `anon`, and
`client_kwargs` with region or endpoint settings. The reader translates these
to GDAL's process-wide S3 configuration; a returned `RasterFile`'s `path` also
carries its storage options. Alternatively configure GDAL before creating reads:

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
