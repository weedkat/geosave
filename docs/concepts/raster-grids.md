# Raster grids

Place new pixel values on an existing raster's grid with ODC's `xr_coords`:

```python
import numpy as np
import xarray as xr
from odc.geo.xr import xr_coords

import geosave as gs

from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 500040, 5000040),
    crs='EPSG:32633', resolution=10,
)
scene = (
    xr.Dataset({"red": (("y", "x"), np.arange(16, dtype="uint16").reshape(4, 4))})
    .assign_coords(xr_coords(grid))
    .gs.write_nodata(65535)
)
prediction = (
    xr.Dataset({"label": (("y", "x"), np.ones((4, 4), dtype="uint8"))})
    .assign_coords(xr_coords(scene.gs.geobox))
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

`assign_coords(xr_coords(geobox))` places supplied values on a grid's coordinates. It does not
reproject those values: pixels must be as large as the grid, and dimensions must
end in `y, x`, or it refuses. Coordinates on other dimensions, such as `time`,
are set with xarray's own constructor and kept.

Nothing is copied from the raster the grid came from, neither its band attrs nor
its Dataset attrs, since they would not describe how the new raster was
produced. Write what the new values mean with `write_nodata`, `write_flags` and
the other `write_*` methods.

## Change a grid explicitly

```python
latlon = scene.gs.reproject("EPSG:4326", resampling="nearest")
latlon.gs.crs.epsg
```

```text
4326
```

Reprojection creates values on a new grid. Nearest-neighbour resampling chooses
an existing source value, which is appropriate for class labels. Bilinear and
other interpolating methods mix values and can be useful for continuous data.
Choose both the grid and the resampling method for your use case.

Projected CRSs such as this UTM CRS commonly use metres; geographic CRSs such
as EPSG:4326 use degrees. A numeric resolution means units of the target CRS,
so `resolution=10` is not always ten metres.

## Keep or align a tree's native grids

A Dataset uses shared spatial coordinates. A DataTree can retain different
asset grids, such as Sentinel-2's 10 m and 20 m bands, in separate leaf nodes.
Each asset has its own `gs.geobox`; do not assume the tree has one common grid.

An exact GeoBox places every asset on the same grid:

```python title="Adaptation example — use your tree and reference raster"
aligned = tree.gs.reproject(reference.gs.geobox, resampling="nearest")
scene = aligned["optical"].to_dataset()
```

A CRS destination instead sizes each asset independently, even with an explicit
resolution. It does not ensure identical placement or extent. Select the
Dataset at a caller-defined node after alignment; native xarray combines Datasets.
Categorical variables refuse interpolating kernels. Reproject imagery and labels
separately when they need different methods; see [align labels with imagery](../guides/align-rasters.md).

## Cropping and windows

`isel(y=..., x=...)` selects pixel positions. `.gs.crop(region)` also masks
pixels outside a geometry, using nodata. Neither operation changes resolution.
Load selected catalog rows onto an exact grid directly:

```python title="Adaptation example — use selected catalog rows and a reference raster"
scene = rows.gs.to_dataset(geobox=reference.gs.geobox, chunks={})
```

Use `stac.load_native_datasets(rows.gs.to_items())` when separate native-grid
Datasets are preferable to resampling.

Reprojection can be lazy for Dask inputs, but interpolation and nodata behavior
depend on the resampling backend. Check a small representative area before
running a large job. See [transforms](../reference/transforms.md),
[rasters](../reference/rasters.md), and [large data](../guides/large-data.md).

## Reprojection destinations

`raster.gs.reproject(how, ...)` accepts an ODC `GeoBox` or a CRS
(string, EPSG integer, or supported CRS object). A GeoBox specifies the exact
shape, affine transform, and CRS. Its transform supplies resolution; a separate `resolution` argument is ignored. A CRS destination derives an output
grid from the source, optionally using `resolution=`.

```python
projected = raster.gs.reproject("EPSG:4326", resolution=0.0001)
matched = raster.gs.reproject(reference.gs.geobox)
aligned = tree.gs.reproject(reference.gs.geobox)
```

Raster objects are not destinations; pass their `gs.geobox`. A tree with a CRS destination reprojects each asset independently. Pass a GeoBox
when all assets must land on one grid.
