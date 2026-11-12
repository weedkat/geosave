# Lazy data and computation

Build a calculation without reading every pixel:

```python
import dask.array as da
import xarray as xr
from odc.geo.xr import xr_coords

import geosave as gs

from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 500040, 5000040),
    crs='EPSG:32633', resolution=10,
)
pixels = da.arange(16, chunks=4).reshape(4, 4).astype("uint16")
scene = xr.Dataset({"red": (("y", "x"), pixels)}).assign_coords(xr_coords(grid)).gs.write_nodata(0)
subset = scene.isel(y=slice(0, 2), x=slice(0, 2))
subset
```

Output:

```text
<xarray.Dataset> Size: 44B
Dimensions:      (y: 2, x: 2)
Coordinates:
  * y            (y) float64 16B 5e+06 5e+06
  * x            (x) float64 16B 5e+05 5e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 8B dask.array<chunksize=(1, 2), meta=np.ndarray>
```

Lazy arrays describe work to run later. Their coordinates and metadata can be
available while their pixels remain unread. The Dataset display's `Size` is
the logical size of the values, not a measurement of memory currently used.

## Backend laziness and Dask chunks

`gs.open_raster(path, chunks=None)` generally uses the file backend's lazy array
access. It can defer reads without a Dask graph. Dated TIFFs gain a Dask-backed
time axis because adding that axis to a backend array would otherwise read it.

`chunks={}` requests Dask arrays using the backend's preferred chunks;
`chunks={"y": 1024, "x": 1024}` selects spatial chunk sizes. Chunk shape affects
parallelism and read efficiency. A small logical selection may still read a
whole TIFF block or Zarr chunk that contains it.

STAC sources use Dask chunks by default. Their `set_config(chunks=None)`
explicitly requests eager loading; this differs from a file reader's
backend-lazy `chunks=None`.

## What triggers work

```python
loaded = subset.compute()
loaded
```

Output:

```text
<xarray.Dataset> Size: 44B
Dimensions:      (y: 2, x: 2)
Coordinates:
  * y            (y) float64 16B 5e+06 5e+06
  * x            (x) float64 16B 5e+05 5e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 8B 0 1 4 5
```

`.compute()` returns an in-memory object; `.load()` fills the existing object.
`.gs.to_numpy()` and `.gs.to_tensor()` also materialize the selected pixels.
`.gs.statistics()` computes reductions and returns a small table. Plotting
needs the displayed pixel values, and writes normally compute their output.

Selection, cropping, masking, decoding, and spectral arithmetic can preserve
Dask laziness. Rasterization produces pixels eagerly; vectorization needs the
selected two-dimensional plane in memory. Reprojection can remain lazy, but
its result still needs checking for the selected resampling method.

## Read chips on demand

`gs.Chips([scene], size=256)` plans tile layouts without computing pixels.
`chips[0]` returns one lazy window; computing it reads that window, and
repeated reads can read it again. Keep the source open until all dependent
reads finish. Storage chunks and upstream transforms may require more work than
the requested window.

Prediction merging still allocates full output accumulators. Lazy input reads
do not make whole-scene output memory bounded. See [chip reading and merging](../guides/catalogs-and-cuts.md#cut-raster-chips)
for batching and output rules.

## Schedule a write explicitly

Every `to_cog`, `to_zarr` and `to_netcdf` takes `compute=False` and returns
one native Dask Delayed:
`scene.gs.to_zarr("scene.zarr", compute=False).compute()` writes the store and
returns its UPath, as `to_netcdf` does for its file and `to_cog` for its list of
files; a tree's `to_zarr` and `to_netcdf` write one store with its groups. Nothing reaches the
destination before the delayed job runs.

Keep file-backed rasters and trees open until the work that depends on them
finishes. Close them afterward to release handles. Select the region and
variables before converting to NumPy or tensors so you only materialize the
input you need. See [large rasters](../guides/large-data.md) for a complete
crop, reduce, and save example.
