# Lazy data and computation

A lazy raster describes pixel work to run later. Opening a file makes its
coordinates and metadata available without loading every pixel.

With the [sample scene](../basics/first-raster.md#open-the-sample):

```python
import geosave as gs

scene = gs.open_raster("sample-scene.tif", chunks={"y": 128, "x": 128})
ndvi = gs.features.ndvi(scene, nir="nir", red="red").rename("ndvi")
ndvi.chunks
```

```text
((128, 128, 81), (128, 128, 11))
```

NDVI is still split into deferred blocks. This describes the work layout,
not measured memory use. Select bands and an area before computing.

## When pixels are read

| Action | Effect |
| --- | --- |
| Select bands, crop, decode or calculate NDVI | Can preserve lazy arrays. |
| Plot or explore a map | Reads the displayed values. |
| `.compute()` | Returns an in-memory object. |
| `.load()` | Loads values into the existing object. |
| Convert to NumPy or tensors | Materializes the selected input. |
| Save a raster | Normally computes and writes output pixels. |

## File reads and Dask blocks

File readers with `chunks=None` generally defer reads through their backend.
`chunks={}` requests Dask arrays with backend-preferred chunks; an explicit
mapping chooses block sizes.

STAC sources use Dask by default. Their `chunks=None` setting requests eager
loading, unlike a file reader's backend-lazy mode.

A small requested window can still read an entire TIFF block or Zarr chunk.
Lazy inputs do not guarantee bounded output memory. For example, prediction
merging holds whole-raster accumulators unless a chunked output workflow is used.

## Control the boundary

[Large-raster processing](../advanced/large-data.md) selects an area before
writing. [Training chips](../advanced/overlapping-chips.md) materializes model-sized windows.
[Chunked prediction](../advanced/release.md#write-large-predictions-to-zarr)
writes model outputs a region at a time.

The [I/O reference](../reference/io.md#saved-raster-files) describes
`compute=False` and writer-specific behavior.
