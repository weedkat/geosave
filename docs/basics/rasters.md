# Work with rasters

Build a small image with two named bands:

```python
import numpy as np

import geosave as gs

anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=4, resolution=10, crs="EPSG:32633"
)
red = np.arange(1, 17, dtype="uint16").reshape(4, 4)
scene = gs.raster(
    {"red": (("y", "x"), red), "nir": (("y", "x"), red + 4)},
    anchor.geobox,
    nodata=0,
)
scene
```

Output:

```text
<xarray.Dataset> Size: 132B
Dimensions:      (y: 4, x: 4)
Coordinates:
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 32B 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    nir          (y, x) uint16 32B 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
```

A raster stores values in a grid of pixels. Here each pixel covers 10 metres
by 10 metres. `red` and `nir` are two measurements of those same locations.

`scene` is an xarray `Dataset`. Its data variables are the bands, its `y` and
`x` coordinates locate pixel centres, and `spatial_ref` carries the coordinate
reference system (CRS). GeoSave adds geospatial operations through `.gs`.
You can still use normal xarray selection and arithmetic.

| Object | Stores | Use when |
| --- | --- | --- |
| `DataArray` | One named or unnamed array with coordinates and attrs | Working with one band or a model output. |
| `Dataset` | Named arrays on shared coordinates | Working with several bands or derived features. |
| `DataTree` | Named Dataset groups | Keeping inputs with different variables, dates, or grids together. |

The constructors accept NumPy and lazy arrays. They attach coordinates and
metadata; they do not resample or compute the pixels. Use dimensions such as
`("time", "y", "x")` for a time series and supply `time=dates` when the reference
has no time coordinate. Pixel lengths must match the reference's coordinates.

## Select bands and pixels

```python
scene["red"]
```

Output:

```text
<xarray.DataArray 'red' (y: 4, x: 4)> Size: 32B
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16]], dtype=uint16)
Coordinates:
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B 32633
Attributes:
    _FillValue:  0
```

One band is an xarray `DataArray`. Select several bands with
`scene[["red", "nir"]]` to keep a `Dataset`. Use `isel` for pixel positions:

```python
patch = scene.isel(y=slice(0, 2), x=slice(0, 2))
patch
```

Output:

```text
<xarray.Dataset> Size: 52B
Dimensions:      (y: 2, x: 2)
Coordinates:
  * y            (y) float64 16B 4.987e+06 4.987e+06
  * x            (x) float64 16B 2.635e+05 2.635e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 8B 1 2 5 6
    nir          (y, x) uint16 8B 5 6 9 10
```

The patch keeps the coordinates of the selected pixels. Selecting fewer pixels
changes the extent; it does not resample them.

For a file you already have, start with `gs.open_raster("scene.tif")`.
The [read and write guide](io.md) gives a complete local round trip.

## Calculate an index

```python
ndvi = gs.features.ndvi(scene, nir="nir", red="red")
scene = scene.assign(ndvi=ndvi)
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
    red          (y, x) uint16 32B 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    nir          (y, x) uint16 32B 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    ndvi         (y, x) float32 64B 0.6667 0.5 0.4 ... 0.125 0.1176 0.1111
```

NDVI compares near-infrared and red reflectance. This synthetic example only
shows the calculation. The feature function returns an unnamed `DataArray`;
assignment gives it the name `ndvi`.

Spectral indices decode the selected bands' nodata and scale/offset metadata
before calculating. For your own arithmetic, decode explicitly with
`scene.gs.mask_and_scale()` when the file stores packed values.
See [describe raster values](attrs.md).

## Plot a band

```python
scene.gs.plot("ndvi", cmap="YlGn", xlabel="")
```

![NDVI values on the synthetic raster grid](../assets/raster-ndvi.png)

The color scale shows the band's values. `xlabel=""` leaves out a location
caption and avoids its reverse-geocoding lookup. Name three variables to draw a
composite: `scene.gs.plot("B08", "B04", "B03")`.

## Explore on a map

```python
scene.gs.explore("ndvi", basemap="Imagery")
```

`explore` returns a native Panel Material UI app with a light/dark theme toggle.
The compact sidebar groups catalogs, rasters and vectors separately, with
visibility and zoom/remove actions on each layer. Style, Info and Identify tabs
contain the selected layer’s controls, legend and metadata.
The map sits beside it; a collapsible catalog or attribute table sits below
the map. Checking catalog rows opens their rasters automatically; unchecking
removes them. Column filters and date ranges retain original row identity.

A raster opens with its declared RGB channels, otherwise its first spatial
band. Explicit `band` or `rgb` arguments take priority. The **Display** dropdown lists
each spatial band and **RGB** when a declared or explicit composite exists. Each raster
keeps its own choices, opacity, stretch and date. **Color by** selects a vector
attribute. The optional comparison card switches between two open rasters.

The map uses Panel, HoloViews, GeoViews and Bokeh. Dense point, line and polygon layers
use Datashader for numeric, categorical and single-color rendering. Imagery, light, dark and OpenStreetMap basemaps include their
providers' attribution. Basemaps need network access; the explorer needs a
live notebook kernel or `panel serve`.

COG display reads windows and overviews before reprojection. Computed xarray
rasters select and sample the viewport before Dask execution. A zoomed-out
computed overview can still execute every source chunk; the explorer does
not create a pyramid. Opacity and visibility changes reuse displayed raster
pixels; hidden rasters do no viewport work. Percentile sliders apply on release.
Display choices do not mutate the source raster or its RGB metadata.

Layers compose with `scene.gs.explore() * prediction * fields`, or with
`scene.gs.explore(overlay={"fields": fields})`. Two explorers also compose:
`raster1.gs.explore() * raster2.gs.explore()` lists both under Rasters, with
independent display settings and no catalog table. Adding a catalog shows its
checkbox table; adding a vector shows its attribute table. For local serving, put
`scene.gs.explore().servable()` in a script and run `panel serve explore.py`.
See the [Material UI and performance validation](../contributing/handoffs/2026-10-16-holoviz-explorer/UI-VALIDATION.md)
for the tested notebook setup and remaining environment limitations.

## Put new pixels on an existing grid

```python
prediction = np.ones((4, 4), dtype="uint8")
labels = gs.raster({"label": (("y", "x"), prediction)}, scene, nodata=255)
labels
```

Output:

```text
<xarray.Dataset> Size: 84B
Dimensions:      (y: 4, x: 4)
Coordinates:
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B 32633
Data variables:
    label        (y, x) uint8 16B 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```

`gs.raster` needs explicit dimension names ending in `y, x`. Passing `scene`
as the reference places the prediction on the same grid. Coordinates on other
dimensions, such as `time`, pass on when the new variables use those dimensions.
Same-name variables also inherit band attrs; supply a new name or explicit attrs
when the new values have different meaning.

To build one `DataArray` directly, use
`gs.array(prediction, scene, dims=("y", "x"), nodata=255)`.

## Prepare an image tensor

```python
image = scene[["red", "nir"]].gs.to_tensor(dtype="float32")
image.shape, image.dtype
```

Output:

```text
(torch.Size([2, 4, 4]), torch.float32)
```

The tensor has channels first, in the selected variable order. Converting to a
tensor reads the pixels. It does not mask nodata or scale stored values for you.
Keep the georeferenced raster alongside the tensor to place model outputs later.
For a time series, the shape is `[time, band, y, x]`; other shared nonspatial
axes also precede `band`. All variables must span the same axes. If their
dtypes differ, pass `dtype=` explicitly; GeoSave does not choose a common dtype
for you.

## Put rasters together

Use named groups when inputs have different variables or time coordinates:

```python
sample = gs.tree({"optical": scene, "label": labels})
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
│       spatial_ref  int32 4B 32633
├── Group: /optical
│       Dimensions:      (y: 4, x: 4)
│       Coordinates:
│           spatial_ref  int32 4B 32633
│       Data variables:
│           red          (y, x) uint16 32B 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
│           nir          (y, x) uint16 32B 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
│           ndvi         (y, x) float32 64B 0.6667 0.5 0.4 ... 0.125 0.1176 0.1111
└── Group: /label
        Dimensions:      (y: 4, x: 4)
        Coordinates:
            spatial_ref  int32 4B 32633
        Data variables:
            label        (y, x) uint8 16B 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```

A tree is an xarray `DataTree`, with one `Dataset` per group. These groups share
one spatial grid, so their spatial coordinates appear at the root. Other trees
can keep each group's native grid until you align them explicitly.
`sample["optical"].to_dataset()` retrieves a raster, and
`sample.gs.to_tensor()` returns a dictionary of tensors keyed by group name.

Continue with [raster grids](../concepts/raster-grids.md) for CRS, alignment,
and resampling, or [large rasters](../guides/large-data.md) for lazy processing.
For a task starting with files, follow [file ingestion](../guides/ingest-files.md).
The full API is in [rasters](../reference/rasters.md) and
[features](../reference/features.md).
