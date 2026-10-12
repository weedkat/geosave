# Raster grids

Use an existing raster as the reference when placing new pixel values:

```python
import numpy as np

import geosave as gs

anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=4, resolution=10, crs="EPSG:32633"
)
scene = gs.raster(
    {"red": (("y", "x"), np.arange(16, dtype="uint16").reshape(4, 4))},
    anchor.geobox,
    nodata=65535,
)
prediction = gs.raster(
    {"label": (("y", "x"), np.ones((4, 4), dtype="uint8"))},
    scene,
    nodata=255,
)
prediction
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

A grid says where pixels sit. Its **CRS** defines how coordinates refer to
Earth, its **transform** maps pixel positions to those coordinates, and its
**shape** states the number of rows and columns. Together these form a
`GeoBox`, available as `scene.gs.geobox`. The `y` and `x` coordinates label
pixel centres; the bounds describe the outer edges.

## Same shape is not the same grid

Two 256 × 256 arrays can cover different locations or resolutions.
To combine their pixels spatially, they need the same CRS, transform, and
shape. Even rasters in the same CRS can have offset pixel centres.

`gs.raster(variables, reference)` places supplied values on the reference
coordinates. It does not reproject those values. For an xarray reference,
coordinates on the dimensions used by the new variables pass on unchanged.
Existing coordinate labels that disagree are rejected rather than silently
realigned. Always declare dimensions ending in `y, x`.

Same-name variables inherit the reference's band attrs; each supplied variable's
own attrs override them. Dataset attrs are not copied as a record of how the
new raster was produced. Use a different variable name or explicit attrs when
the values have different semantics.

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
  * y            (y) float64 48B 45.0 45.0 45.0 45.0 45.0 45.0
  * x            (x) float64 64B 12.0 12.0 12.0 12.0 12.0 12.0 12.0 12.0
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

## Keep or align a stack's native grids

An xarray `Dataset` uses one spatial grid for its variables. A `DataTree`
stack can keep different grids in separate groups. When all groups share an
exact grid, the root publishes common spatial coordinates. Otherwise each
group retains its own coordinates, and the stack has no common `geobox`.

Use `stack.gs.align(...)` to put the groups on one grid before pixelwise
combination. Supply resampling per variable, using names such as
`"labels/mask"`, when continuous imagery and categorical labels need different
treatment. Constructing a stack itself does
not perform this alignment.

## Cropping and windows

`isel(y=..., x=...)` selects pixel positions. `.gs.crop(region)` also masks
pixels outside a geometry, using nodata. Neither operation changes resolution.
A catalog window has its own grid; `.gs.to_stack()` places each source on that
grid, slicing an aligned source and reprojecting a different one.

Reprojection can be lazy for Dask inputs, but interpolation and nodata behavior
depend on the resampling backend. Check a small representative area before
running a large job. See [transforms](../reference/transforms.md),
[rasters](../reference/rasters.md), and [large data](../guides/large-data.md).
