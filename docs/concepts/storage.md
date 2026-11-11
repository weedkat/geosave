# Storage and file layouts

Details behind [read and save data](../basics/io.md). Examples reuse its `scene`, `files`, `store` and `root`.

## Recognized suffixes

| Reader | Recognized suffixes |
| --- | --- |
| Raster | `.tif`, `.tiff`, `.jp2`, `.png`, `.zarr`, `.nc`, `.nc4`, `.cdf`, `.safe` |
| Vector | `.parquet`, `.geoparquet`, `.geojson`, `.json`, `.gpkg` |

Suffix matching is case-insensitive. `.SAFE` is a vendor product directory,
whose whole-product reader is recognized but currently raises
`NotImplementedError`; open individual `.jp2` files instead. A PNG without georeferencing can be opened,
but spatial operations still need a locatable grid. Unsupported suffixes raise
`ValueError`; use the format's reader directly when suffix dispatch is unsuitable.

## Save a time series as image files

```python
dated = scene.expand_dims(time=[np.datetime64("2025-01-01")])
items = dated.gs.to_cog(root / "dated", format_paths="scene_{time:%Y%m%d}.tif")
[file.name for file in items]
```

Output:

```text
['scene_20250101.tif']
```

`to_cog(path, format_paths=...)` takes an output directory or URL prefix.
The relative `path_format` determines the filenames; without it, the accessor
uses its default filename pattern. Literal braces in `path` stay literal. The template determines how
data splits into files. `{time}` writes a
file per date; `{band}` writes a file per band. Bands sharing a file must have
the same dtype and nodata. For a tree, include `{collection}` to name each
group. Fill any other fields in the template before passing it. A scalar `time` coordinate is
accepted for one image. Several dates need distinct paths, or the writer
raises before writing them.

Existing destinations raise `FileExistsError` by default. Pass `overwrite=True`
to replace them. The returned paths are the files this call wrote, not an
inventory of the destination folder.
For a searchable catalog, write with `to_catalog` instead: it writes the same
files and returns their [STAC Items](../guides/stac.md) as a table.

## Read a tree

```python
optical = gs.open_raster(files[0])
label = gs.open_raster(store)
sample = xr.DataTree.from_dict({
    "/optical": xr.Dataset(attrs=optical.attrs),
    "/optical/r10m": optical,
    "/label": xr.Dataset(attrs=label.attrs),
    "/label/r10m": label,
})
sample
```

Output:

```text
<xarray.DataTree>
Group: /
├── Group: /optical
│   │   Attributes:
│   │       AREA_OR_POINT:  Area
│   └── Group: /optical/r10m
│           Dimensions:      (y: 4, x: 4)
│           Coordinates:
│             * y            (y) float64 32B 5e+06 5e+06 5e+06 5e+06
│             * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
│               spatial_ref  int64 8B 0
│           Data variables:
│               red          (y, x) uint8 16B dask.array<chunksize=(4, 4), meta=np.ndarray>
└── Group: /label
    └── Group: /label/r10m
            Dimensions:      (y: 4, x: 4)
            Coordinates:
              * y            (y) float64 32B 5e+06 5e+06 5e+06 5e+06
              * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
                spatial_ref  int32 4B ...
            Data variables:
                red          (y, x) uint8 16B ...
```

Native xarray composition names the groups explicitly. A tree saved with
`.gs.to_zarr("scene.zarr")` or `.gs.to_netcdf("scene.nc")` preserves its
collection and grid groups; `gs.open_raster(path)` returns a DataTree for such
stores and a Dataset for root-only stores. Directories and mappings are not
assembled by the opener. Use catalog loading for assets split across files.

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

Close file-backed objects when you finish: `sample.close()`.
