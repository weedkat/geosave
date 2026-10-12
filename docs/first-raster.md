# Your first raster

Build a small two-band image, compute NDVI, then save and reopen it. This example
uses synthetic reflectance values and needs no downloads. Install GeoSave first
using the [installation guide](install.md).

## Create an image

```python
import numpy as np
import geosave_engine as gs

anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=4, resolution=10, crs="EPSG:32633"
)
red = np.full((4, 4), 0.25, dtype="float32")
nir = np.full((4, 4), 0.75, dtype="float32")
scene = gs.raster(
    {"red": (("y", "x"), red), "nir": (("y", "x"), nir)}, anchor.geobox
)
scene
```

Output:

```text
<xarray.Dataset> Size: 196B
Dimensions:      (y: 4, x: 4)
Coordinates:
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) float32 64B 0.25 0.25 0.25 0.25 ... 0.25 0.25 0.25 0.25
    nir          (y, x) float32 64B 0.75 0.75 0.75 0.75 ... 0.75 0.75 0.75 0.75
```

`scene` is an xarray Dataset. Each variable is a band, and `y, x` describe its
pixel layout. The anchor supplies a 4 × 4 grid with 10 m pixels around the given
location. `spatial_ref` carries the coordinate reference system (CRS).

## Compute a feature

```python
scene = scene.assign(ndvi=gs.features.ndvi(scene, nir="nir", red="red"))
scene.ndvi
```

Output:

```text
<xarray.DataArray 'ndvi' (y: 4, x: 4)> Size: 64B
array([[0.5, 0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5, 0.5]], dtype=float32)
Coordinates:
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B 32633
```

NDVI is `(nir - red) / (nir + red)`. Here every pixel is `0.5`. Assignment adds
it as a named variable on the existing grid.

## Save and reopen

```python
from pathlib import Path
import tempfile

root = Path(tempfile.mkdtemp())
files = scene.gs.to_cog(root / "scene.tif")
opened = gs.open_raster(files)
opened
```

Output:

```text
<xarray.Dataset> Size: 264B
Dimensions:      (x: 4, y: 4)
Coordinates:
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
    spatial_ref  int64 8B 0
Data variables:
    red          (y, x) float32 64B dask.array<chunksize=(4, 4), meta=np.ndarray>
    nir          (y, x) float32 64B dask.array<chunksize=(4, 4), meta=np.ndarray>
    ndvi         (y, x) float32 64B dask.array<chunksize=(4, 4), meta=np.ndarray>
Attributes:
    AREA_OR_POINT:  Area
```

`to_cog` writes a Cloud Optimized GeoTIFF and returns a tuple of file records.
`open_raster` accepts those records or a file path. The reopened bands are dask
arrays: their pixel data loads when you compute, plot, or convert it to tensors.
Close the dataset with `opened.close()` when finished.

For your own image, use `gs.open_raster("path/to/scene.tif")`. Its band names and
grid come from the file; no anchor is needed to open it.

Continue with [Raster data](basics/rasters.md) for selection and processing, or
[From images to training chips](quickstart.md) for a complete data preparation
example. See the [constructor](reference/rasters.md) and
[I/O reference](reference/io.md) for argument details.
