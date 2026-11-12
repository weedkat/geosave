# Process large rasters

Crop a raster before reading its pixels:

```python
import dask.array as da
import geopandas as gpd
from shapely.geometry import box

import xarray as xr
from odc.geo.xr import xr_coords

import geosave as gs

from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 502560, 5002560),
    crs='EPSG:32633', resolution=10,
)
pixels = da.random.RandomState(0).randint(
    0, 3000, (256, 256), chunks=128
).astype("uint16")
scene = xr.Dataset({"red": (("y", "x"), pixels)}).assign_coords(xr_coords(grid)).gs.write_nodata(0)
scene
```

Output:

```text
<xarray.Dataset> Size: 135kB
Dimensions:      (y: 256, x: 256)
Coordinates:
  * y            (y) float64 2kB 5.003e+06 5.003e+06 5.003e+06 ... 5e+06 5e+06
  * x            (x) float64 2kB 5e+05 5e+05 5e+05 ... 5.025e+05 5.026e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 131kB dask.array<chunksize=(128, 128), meta=np.ndarray>
```

This example uses generated data so it runs offline. For a file, start with
`scene = gs.open_raster("scene.tif", chunks={"y": 1024, "x": 1024})`.
The `dask.array` shown in the Dataset means the values are described as work to
do later. The 128 × 128 chunks in this example are blocks Dask can process
separately; constructing the raster does not generate their pixels.

| Call | Returns | Pixel execution |
| --- | --- | --- |
| `gs.open_raster(path, chunks=...)` | Dataset with Dask arrays | Reads headers; builds deferred reads. |
| `.isel(...)` or `.gs.crop(...)` | Selected raster | Stays lazy for Dask-backed inputs. |
| `.gs.statistics()` | pandas DataFrame | Computes reductions over selected pixels. |
| `.compute()` | Computed xarray object | Materializes the selected data in memory. |
| `.gs.to_zarr(path)` | One store UPath | Computes and writes output pixels. |
| `.gs.to_tensor()` | PyTorch tensor | Materializes pixels into a writable tensor. |

Selecting first reduces the logical data required by later operations. File
headers and small coordinate arrays are read independently of the pixel graph.

## Select the area you need

```python
bounds = scene.gs.bounds
region = gpd.GeoSeries(
    [box(bounds.left, bounds.bottom, bounds.left + 640, bounds.bottom + 640)],
    crs=bounds.crs,
)
cropped = scene.gs.crop(region)
cropped
```

Output:

```text
<xarray.Dataset> Size: 9kB
Dimensions:      (y: 64, x: 64)
Coordinates:
  * y            (y) float64 512B 5.001e+06 5.001e+06 5.001e+06 ... 5e+06 5e+06
  * x            (x) float64 512B 5e+05 5e+05 5e+05 ... 5.006e+05 5.006e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 8kB dask.array<chunksize=(64, 64), meta=np.ndarray>
```

The region is a 640 metre square. At 10 metres per pixel, the result is
64 × 64 pixels, still backed by Dask. `crop` selects the bounding pixels and
masks those outside the geometry. It needs a nodata value for that mask;
`nodata=0` provides one here. Use `isel` for a rectangular selection by pixel
position when you do not need a geometry mask.

## Compute a result

```python
summary = cropped.gs.statistics().round(2)
summary
```

Output:

```text
     minimum  maximum     mean  stddev  valid_percent
red      1.0   2999.0  1493.72  863.43          99.95
```

`statistics()` computes reductions and excludes nodata and NaN. It returns a
small pandas table; the full raster stays lazy. For ordinary xarray arithmetic,
decode missing values first:

```python
mean = cropped.gs.mask_and_scale()["red"].mean().compute()
mean
```

Output:

```text
<xarray.DataArray 'red' ()> Size: 4B
array(1493.7198, dtype=float32)
Coordinates:
    spatial_ref  int32 4B 32633
```

An xarray mean of the raw integer band would include `0`; it does not interpret
the GeoSave nodata attr by itself. `mask_and_scale()` makes those pixels NaN,
which floating-point reductions skip by default.

The statistics are over all selected axes of each variable, including time
when present. An entirely missing variable raises `ValueError`; it does not
return a row of invented statistics. The result's `valid_percent` is the
fraction of selected pixels that are neither nodata nor NaN.

## Save the prepared subset

```python
import tempfile
from pathlib import Path

root = Path(tempfile.mkdtemp())
store = cropped.gs.to_zarr(root / "crop.zarr")
prepared = gs.open_raster(store, chunks={})
prepared
```

Output:

```text
<xarray.Dataset> Size: 9kB
Dimensions:      (y: 64, x: 64)
Coordinates:
  * y            (y) float64 512B 5.001e+06 5.001e+06 5.001e+06 ... 5e+06 5e+06
  * x            (x) float64 512B 5e+05 5e+05 5e+05 ... 5.006e+05 5.006e+05
    spatial_ref  int32 4B ...
Data variables:
    red          (y, x) uint16 8kB dask.array<chunksize=(64, 64), meta=np.ndarray>
```

```python
prepared.close()
```

Writing computes the selected data and stores its chunks. To schedule a write
yourself, assign `pending = cropped.gs.to_zarr(path, compute=False)`, then call
`paths = pending.compute()`. Nothing reaches the destination until then.
`to_cog` and `to_netcdf` take `compute=False` too. Each call returns one
native Dask Delayed; its computed value has the same shape as an eager result.

## Read one model input at a time

[Catalogs and samples](catalogs-and-cuts.md) shows how to group file metadata
with `items.gs.sample()`, open sample trees with `rows.gs.to_datatree()`, and
cut those trees with `gs.Chips`. Sampling and chip layout read no pixels;
selecting a chip builds lazy raster work for its grid.

The logical window may be smaller than the storage chunks or TIFF blocks the
backend must read. Choose on-disk blocks, Dask chunks, and model chip sizes with
that distinction in mind. A chip at the edge can also require padding.

See [lazy data](../concepts/lazy-data.md) for computation boundaries, and
[raster grids](../concepts/raster-grids.md) before reprojecting a large raster.
