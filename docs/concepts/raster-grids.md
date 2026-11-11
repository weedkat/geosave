# Raster grids

Place new pixel values on an existing raster's grid with `write_grid`:

```python
import numpy as np
import xarray as xr

import geosave as gs

from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 500040, 5000040),
    crs='EPSG:32633', resolution=10,
)
scene = (
    xr.Dataset({"red": (("y", "x"), np.arange(16, dtype="uint16").reshape(4, 4))})
    .gs.write_grid(grid)
    .gs.write_nodata(65535)
)
prediction = (
    xr.Dataset({"label": (("y", "x"), np.ones((4, 4), dtype="uint8"))})
    .gs.write_grid(scene.gs.geobox)
    .gs.write_nodata(255)
)
prediction
```

Output:

```text
<xarray.Dataset> Size: 84B
Dimensions:      (y: 4, x: 4)
Coordinates:
  * y            (y) float64 32B 5e+06 5e+06 5e+06 5e+06
  * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
    spatial_ref  int32 4B 32633
Data variables:
    label        (y, x) uint8 16B 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```

A grid says where pixels sit. Its **CRS** defines how coordinates refer to
Earth, its **transform** maps pixel positions to those coordinates, and its
**shape** states the number of rows and columns. Together these form a
`GeoBox`, available as `scene.gs.geobox`. The `y` and `x` coordinates label
pixel centres; the bounds describe the outer edges.

## Same shape is not the same grid

Two 256 × 256 arrays can cover different locations or resolutions.
To combine their pixels spatially, they need the same CRS, transform, and
shape. Even rasters in the same CRS can have offset pixel centres.

`write_grid(geobox)` places supplied values on a grid's coordinates. It does not
reproject those values: pixels must be as large as the grid, and dimensions must
end in `y, x`, or it refuses. Coordinates on other dimensions, such as `time`,
are set with xarray's own constructor and kept.

Nothing is copied from the raster the grid came from, neither its band attrs nor
its Dataset attrs, since they would not describe how the new raster was
produced. Write what the new values mean with `write_nodata`, `write_legend` and
the other `write_*` methods.

## Change a grid explicitly

```python
latlon = scene.gs.reproject("EPSG:4326", resampling="nearest")
latlon
```

Output:

```text
<xarray.Dataset> Size: 212B
Dimensions:      (y: 6, x: 8)
Coordinates:
  * y            (y) float64 48B 45.15 45.15 45.15 45.15 45.15 45.15
  * x            (x) float64 64B 15.0 15.0 15.0 15.0 15.0 15.0 15.0 15.0
    spatial_ref  int32 4B 4326
Data variables:
    red          (y, x) uint16 96B 65535 65535 65535 65535 ... 65535 65535 65535
```

Reprojection creates values on a new grid. Nearest-neighbour resampling chooses
an existing source value, which is appropriate for class labels. Bilinear and
other interpolating methods mix values and can be useful for continuous data.
Choose both the grid and the resampling method for your use case.

Projected CRSs such as this UTM CRS commonly use metres; geographic CRSs such
as EPSG:4326 use degrees. A numeric resolution means units of the target CRS,
so `resolution=10` is not always ten metres.

## Keep or align a tree's native grids

An xarray `Dataset` uses one spatial grid for its variables. A `DataTree`
tree can keep different grids in separate groups. When all groups share an
exact grid, the root publishes common spatial coordinates. Otherwise each
group retains its own coordinates, and the tree has no common `geobox`.

Use `tree.gs.to_dataset(grid=...)` before pixelwise combination. `grid` is a
node path relative to the node you call on, or a GeoBox; every node is
reprojected onto it, then merged. Kernels are keyed by variable name, since the
merged Dataset needs distinct names anyway:

```python
scene = tree.gs.to_dataset(
    grid="imagery/r10m",
    resampling={"*": "bilinear", "quality": "nearest", "mask": "nearest"},
)
```

Categorical validation checks every variable after resolving its kernel.
Constructing a tree does not resample it.

## Cropping and windows

`isel(y=..., x=...)` selects pixel positions. `.gs.crop(region)` also masks
pixels outside a geometry, using nodata. Neither operation changes resolution.
A catalog window's rows open with `.gs.to_datatree()` as one node per STAC
asset, each on its own grid (`/s2/B04`, `/s2/B11`). `tree.gs.group_grid()`
merges assets sharing one exact grid into a node named by its pixel size
(`/s2/r10m`, `/s2/r20m`); `tree["s2"].gs.to_dataset()` merges only nodes on
one exact grid. For mixed grids, select a reference explicitly before merging:

```python
scene = tree.gs.group_grid()["s2"].gs.to_dataset(grid="r10m")
```

Conversion does not select a grid or resample pixels.

Reprojection can be lazy for Dask inputs, but interpolation and nodata behavior
depend on the resampling backend. Check a small representative area before
running a large job. See [transforms](../reference/transforms.md),
[rasters](../reference/rasters.md), and [large data](../guides/large-data.md).

## Measured lazy-warp limitation

The current lazy cross-CRS warp can differ numerically from the eager path on
the same target grid. A synthetic nearest/bilinear comparison, timings, and
reproduction scripts are in the [performance review](../contributing/benchmarks/2026-11-04/index.md#cross-crs-performance-and-numerical-comparison).
Choose and validate the resampling behavior your data needs; a correct output
shape and CRS do not by themselves prove equivalent pixels. The review does
not recommend silently computing a whole raster to hide this limitation.

## Reprojection destinations

`raster.gs.reproject(how, ...)` accepts an ODC `GeoBox` or a CRS
(string, EPSG integer, or supported CRS object). A GeoBox specifies the exact
shape, affine transform, and CRS. Resolution is derived from its transform;
do not also supply a resolution override. A CRS destination derives an output
grid from the source, optionally using `resolution=`.

```python
projected = raster.gs.reproject("EPSG:4326", resolution=0.0001)
matched = raster.gs.reproject(reference.gs.geobox)
scene = tree.gs.to_dataset(grid=reference.gs.geobox)
```

Raster objects are not destinations. Select their grid explicitly; for a tree
with different grids, name a node. A tree is not reprojected as a whole; it
reaches one grid only on its way to a Dataset.
