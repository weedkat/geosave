# Process a region without loading the whole scene

Select the bands and geographic area you need, then write the result in
chunks. Use this pattern for larger source files.

**You need:** the base package and the [sample scene](../basics/first-raster.md#open-the-sample).
The sample keeps the example small; it is not a memory benchmark.

## Select before computing

```python
import geosave as gs

scene = gs.open_raster("sample-scene.tif", chunks={"y": 128, "x": 128})
region = gs.GeoRegion.from_bbox((-121.815, 38.525, -121.800, 38.540))
area = scene[["red", "nir"]].gs.crop(region.extent)
ndvi = gs.features.ndvi(area, nir="nir", red="red").rename("ndvi")
ndvi.sizes["y"], ndvi.sizes["x"]
```

```text
(169, 134)
```

Opening builds deferred reads. Selection, cropping and NDVI remain lazy.
A storage block can cover more pixels than the requested region.

## Write a chunked result

```python
from pathlib import Path

Path("output").mkdir(exist_ok=True)
result = ndvi.to_dataset().chunk({"y": 128, "x": 128})
store = result.gs.to_zarr("output/area-ndvi.zarr")
print(store)
```

```text
output/area-ndvi.zarr
```

The output uses regular 128-pixel chunks; cropping can split the input's
chunks at the boundary. Writing computes the selected output. Reopen it:

```python
saved = gs.open_raster(store)
bands = list(saved.data_vars)
bands
```

```text
['ndvi']
```

## Choose the computation boundary

Plotting, `.compute()` and tensor conversion read pixels into memory.
Select first, and convert model-sized windows with
[RasterChipper](overlapping-chips.md); its `iter_batches` reads one block of
chips at a time, so pixels neighbouring chips share are read once. Dask chunks control work scheduling; they do not
guarantee a fixed memory limit for every operation.

Whole-scene prediction also needs output memory. GeoSave provides region
writes for raster stores. Bounded prediction assembly remains an
[open design question](../contributing/design/release-inference.md#open-prediction-decisions).

See [lazy execution](../concepts/lazy-data.md) for computation rules and the
[I/O reference](../reference/io.md#saved-raster-files) for deferred writes.
