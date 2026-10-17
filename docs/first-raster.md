# Your first raster

Create a tiny image with two measurements at each location, calculate a useful
feature, then save and reopen it. The example makes its own data, so you can
run it without downloads or accounts. Install GeoSave first using the
[installation guide](install.md).

## Create an image

```python
import numpy as np
import geosave as gs

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

A raster is an image whose pixels correspond to locations on the ground.
`scene` is a normal xarray Dataset with two named arrays: `red` and `nir`.
They represent reflected light in red and near-infrared wavelengths. Real
satellite images supply these values; here we filled them with simple numbers.

The anchor places the 4 × 4 image around the given latitude and longitude.
Each pixel covers 10 × 10 metres. The `y` and `x` coordinates locate the pixel
centres, and `spatial_ref` records the map coordinate system, called the CRS.
GeoSave keeps that location information alongside the values.

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

NDVI, the normalized difference vegetation index, compares near-infrared and
red reflectance: `(nir - red) / (nir + red)`. Here every pixel is `0.5`.
Assignment adds it as a named variable at the same pixel locations. The original
bands remain in the Dataset, so normal xarray selection and arithmetic still work.

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

`to_cog` writes a Cloud Optimized GeoTIFF and returns a list of `Scene`s, one per
date, each mapping an asset key to its `RasterFile`. `open_raster` accepts those
or a single file path.
The reopened bands are Dask
arrays: their pixel data loads when you compute, plot, or convert it to tensors.
Close the dataset with `opened.close()` when finished.

For your own image, use `gs.open_raster("path/to/scene.tif")`. Its band names and
grid come from the file; no anchor is needed to open it.

This example starts with NumPy arrays, so creating and calculating the image
uses pixels already in memory. Opening a file with `chunks={}` instead builds
Dask arrays; selection and arithmetic can then stay lazy until saving or
computing. The [file-ingestion guide](guides/ingest-files.md) follows that path.

Continue with [Raster data](basics/rasters.md) for selection and processing, or
[From images to training chips](quickstart.md) for a complete data preparation
example. See the [constructor](reference/rasters.md) and
[I/O reference](reference/io.md) for argument details.
