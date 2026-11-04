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

Use `tree.gs.align(target=reference)` to put the groups on the reference
grid before pixelwise combination. The target is required; the default extent
is `"match"`. Request `extent="union"` or `extent="intersection"` explicitly
to change coverage. A reference tree must have a shared grid; otherwise select
one of its groups. Supply resampling per variable, using names such as
`"labels/mask"`, when continuous imagery and categorical labels need different
treatment. Constructing a tree itself does
not perform this alignment.

## Cropping and windows

`isel(y=..., x=...)` selects pixel positions. `.gs.crop(region)` also masks
pixels outside a geometry, using nodata. Neither operation changes resolution.
A catalog window's rows open with `.gs.to_datatree()` on their native grids,
one grid node per pixel size (`/s2/r10m`, `/s2/r20m`); `tree["s2"].gs.to_dataset()`
brings a collection's nodes onto one of its grids, and `.gs.align(target=...)`
reprojects onto another.

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
