# Process large rasters

Crop a raster before reading its pixels:

```python
import dask.array as da
import geopandas as gpd
from shapely.geometry import box

import geosave as gs

anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=256, resolution=10, crs="EPSG:32633"
)
pixels = da.random.RandomState(0).randint(
    0, 3000, (256, 256), chunks=128
).astype("uint16")
scene = gs.raster({"red": (("y", "x"), pixels)}, anchor.geobox, nodata=0)
scene
```

Output:

```text
<xarray.Dataset> Size: 135kB
Dimensions:      (y: 256, x: 256)
Coordinates:
  * y            (y) float64 2kB 4.989e+06 4.989e+06 ... 4.986e+06 4.986e+06
  * x            (x) float64 2kB 2.623e+05 2.623e+05 ... 2.648e+05 2.648e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 131kB dask.array<chunksize=(128, 128), meta=np.ndarray>
```

This example uses generated data so it runs offline. For a file, start with
`scene = gs.open_raster("scene.tif", chunks={"y": 1024, "x": 1024})`.
The `dask.array` shown in the Dataset means the values are described as work to
do later. The 128 × 128 chunks in this example are blocks Dask can process
separately; constructing the raster does not generate their pixels.

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
  * y            (y) float64 512B 4.987e+06 4.987e+06 ... 4.986e+06 4.986e+06
  * x            (x) float64 512B 2.623e+05 2.623e+05 ... 2.629e+05 2.629e+05
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
  * y            (y) float64 512B 4.987e+06 4.987e+06 ... 4.986e+06 4.986e+06
  * x            (x) float64 512B 2.623e+05 2.623e+05 ... 2.629e+05 2.629e+05
    spatial_ref  int32 4B ...
Data variables:
    red          (y, x) uint16 8kB dask.array<chunksize=(64, 64), meta=np.ndarray>
```

```python
prepared.close()
```

Writing computes the selected data and stores its chunks. To schedule a Zarr
write yourself, assign `write = cropped.gs.to_zarr(path, compute=False)`, then
call `write.compute()`. Creating the delayed write can prepare the destination
and metadata; it delays the pixel write, not every filesystem action.

## Read one model input at a time

[Catalogs and windows](catalogs-and-cuts.md) show how to describe many source
files in a table, cut them into model-sized windows, and open one window with
`row.gs.to_stack()`. Cutting the table reads no pixels. Opening a window builds
lazy rasters for the selected region.

The logical window may be smaller than the storage chunks or TIFF blocks the
backend must read. Choose on-disk blocks, Dask chunks, and model chip sizes with
that distinction in mind. A chip at the edge can also require padding.

See [lazy data](../concepts/lazy-data.md) for computation boundaries, and
[raster grids](../concepts/raster-grids.md) before reprojecting a large raster.
