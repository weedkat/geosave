# Lazy data and computation

Build a calculation without reading every pixel:

```python
import dask.array as da

import geosave as gs

anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=4, resolution=10, crs="EPSG:32633"
)
pixels = da.arange(16, chunks=4).reshape(4, 4).astype("uint16")
scene = gs.raster({"red": (("y", "x"), pixels)}, anchor.geobox, nodata=0)
subset = scene.isel(y=slice(0, 2), x=slice(0, 2))
subset
```

Output:

```text
<xarray.Dataset> Size: 44B
Dimensions:      (y: 2, x: 2)
Coordinates:
  * y            (y) float64 16B 4.987e+06 4.987e+06
  * x            (x) float64 16B 2.635e+05 2.635e+05
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
  * y            (y) float64 16B 4.987e+06 4.987e+06
  * x            (x) float64 16B 2.635e+05 2.635e+05
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

## Schedule a write explicitly

For Zarr, `write = scene.gs.to_zarr(path, compute=False)` returns a delayed
pixel write. Run it with `write.compute()` or your Dask scheduler. The call
can prepare destination metadata before the delayed work runs.

Keep file-backed rasters and stacks open until the work that depends on them
finishes. Close them afterward to release handles. Select the region and
variables before converting to NumPy or tensors so you only materialize the
input you need. See [large rasters](../guides/large-data.md) for a complete
crop, reduce, and save example.
