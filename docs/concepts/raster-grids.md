# Raster grids

A grid says where each pixel sits on Earth.

| Part | Describes |
| --- | --- |
| CRS | How coordinates refer to Earth. |
| Transform | Pixel size, orientation and placement. |
| Shape | Number of rows and columns. |

Together these form a GeoBox, available as `.gs.geobox`. Raster coordinates
locate pixel centres; the bounds describe the outer edges.

For the training pair, the CRS is UTM zone 14N and each pixel spans 10 metres.
A 256 × 256 chip therefore covers 2.56 km on each side. The grid connects that
array shape to a specific place, so an image pixel and its label can describe
the same ground location.

## Alignment means the same grid

Matching array sizes is not enough: two 256 × 256 images can cover different
places. Imagery and labels need matching CRS, transform and shape.

The [training pair](../basics/training-chips.md) is already aligned:

```python
import geosave as gs

image = gs.open_raster("training-image.tif")
labels = gs.open_raster("training-labels.tif")
aligned = image.gs.geobox == labels.gs.geobox
aligned
```

```text
True
```

Use [explicit alignment](../advanced/align-rasters.md) when grids differ.
Assigning coordinates to pixels does not reproject them.
It states where supplied values already belong. Reprojection instead samples
values from their original locations onto a destination grid.

## Cropping and reprojection

Cropping selects an area on the existing grid. Reprojection creates values on
a destination grid and needs an explicit resampling choice.

Use nearest-neighbor for categorical labels. Interpolating methods mix values
and can suit continuous imagery. Resolution uses the target CRS's units:
ten units in a geographic CRS means degrees, not metres.

For example, an average of class codes for water and trees is not another
meaningful class. Nearest-neighbor chooses an existing code. Continuous
measurements can use interpolation when that matches the analysis.

A CRS alone does not fully specify a grid. An exact reference GeoBox fixes
placement, extent and pixel size together.

## Multiple grids

A Dataset shares coordinates between its variables. A DataTree can retain
separate native grids in different nodes, such as a sensor's 10 m and 20 m bands.

Keep those grids separate until the workflow needs alignment. RasterChipper
requires its prepared image/label nodes to share one grid.

See [xarray structure](xarray-structure.md) for object layout,
[STAC loading](../advanced/stac.md) for selecting an output grid, and the
[transform reference](../reference/transforms.md) for exact options.
