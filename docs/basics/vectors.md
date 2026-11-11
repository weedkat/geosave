# Work with vector features

Create a table of field boundaries:

```python
import geopandas as gpd
from shapely.geometry import box

import geosave as gs

fields = gpd.GeoDataFrame(
    {"name": ["north"], "class": [1]},
    geometry=[box(12.0, 45.0, 12.001, 45.001)],
    crs="EPSG:4326",
)
fields
```

Output:

```text
    name  class                                           geometry
0  north      1  POLYGON ((12.001 45, 12.001 45.001, 12 45.001,...
```

Vector data represents locations as points, lines, or polygons. Each row is one
feature: ordinary columns describe it and the geometry column locates it.
`fields` is a normal GeoPandas `GeoDataFrame`; GeoSave adds `.gs` operations.

If you already have a file, use `gs.open_vector("fields.parquet")`,
`gs.open_vector("fields.geojson")`, or `gs.open_vector("fields.gpkg")`.
Reading preserves the file's columns; it does not add a training split or ID.

## Check spatial records

```python
fields.gs.validate(crs=True)
fields.iloc[0].gs.validate()
```

Validation returns `None` and leaves the data unchanged. By default it checks
that every geometry is present, nonempty, and valid. `crs=True` also requires a
declared CRS; `geometry=False` skips geometry checks. Empty tables pass these
checks, although operations such as reading a footprint require rows.

For a GeoDataFrame, `area=True` requires valid geometries with positive area.
`temporal=True` checks complete, ordered start/end bounds and requires a stated
`datetime` to fall within them. Undated rows remain valid. Sampling requests
area validation for selected footprints and temporal validation when using dates;
neither check opens raster assets.

For Item tables, `stac_item=True` checks the columns needed for STAC conversion;
it does not validate the full STAC schema or open assets. Ordinary features do
not need those columns. Failures raise `geosave.geodata.VectorValidationError`
with geometry errors identifying the affected row. A standalone Series does
not inherit its parent frame's CRS: use `fields.iloc[[0]].gs.validate(crs=True)`
to check one row while retaining that CRS.

Use `validate(stac_item=False)` to require ordinary features: it rejects STAC
columns or metadata. The default `stac_item=None` skips this check.

## Create a grid for the features

```python
from odc.geo.geobox import GeoBox

region = fields.gs.region
grid = GeoBox.from_geopolygon(region.extent, crs="EPSG:32633", resolution=10)
```

The region retains the feature union and any recognized STAC time coverage.
The GeoBox separately specifies output pixels. Its CRS determines resolution units.

## Turn features into pixels

```python
mask = fields.gs.rasterize(grid)
mask
```

Output:

```text
<xarray.DataArray 'mask' (y: 13, x: 9)> Size: 117B
array([[False, False, False, False, False, False, False, False, False],
       [False,  True,  True,  True,  True,  True,  True,  True,  True],
       [False,  True,  True,  True,  True,  True,  True,  True,  True],
       [False,  True,  True,  True,  True,  True,  True,  True,  True],
       [False,  True,  True,  True,  True,  True,  True,  True,  True],
       [False,  True,  True,  True,  True,  True,  True,  True,  True],
       [False,  True,  True,  True,  True,  True,  True,  True, False],
       [False,  True,  True,  True,  True,  True,  True,  True, False],
       [False,  True,  True,  True,  True,  True,  True,  True, False],
       [ True,  True,  True,  True,  True,  True,  True,  True, False],
       [ True,  True,  True,  True,  True,  True,  True,  True, False],
       [ True,  True,  True,  True,  True,  True,  True,  True, False],
       [False, False, False, False, False, False, False, False, False]])
Coordinates:
  * y            (y) float64 104B 4.987e+06 4.987e+06 ... 4.987e+06 4.987e+06
  * x            (x) float64 72B 2.636e+05 2.636e+05 ... 2.636e+05 2.636e+05
    spatial_ref  int32 4B 32633
```

With no column selected, the result is a boolean `DataArray`: `True` where a
geometry covers the pixel centre and `False` elsewhere. Rasterization computes
these pixels immediately. `all_touched=True` includes every pixel touched by a
geometry instead.

To burn a column's values, use
`fields.gs.rasterize(grid, column="class", dtype="uint8", fill=0)`.
The fill value is used outside the features; choose it separately from the
valid class codes.

The target can be a GeoRegion or a georeferenced xarray object; the result is
an eager `DataArray` on that exact grid.

## Turn pixels into polygons

```python
regions = mask.gs.vectorize()
regions
```

Output:

```text
   value                                           geometry
0      0  POLYGON ((263550 4987450, 263550 4987360, 2635...
1      1  POLYGON ((263560 4987440, 263560 4987360, 2635...
2      0  POLYGON ((263630 4987390, 263630 4987330, 2635...
```

Vectorization groups connected pixels with the same value into polygons.
For this boolean mask, the `value` column contains `0` and `1`.
The output uses the raster's CRS. Select a two-dimensional band or time slice
before vectorizing a raster with extra dimensions.

## Select features near a raster

```python
shape = region.extent.to_crs(fields.crs).geom
selected = fields.loc[fields.geometry.intersects(shape)]
selected
```

Output:

```text
    name  class                                           geometry
0  north      1  POLYGON ((12.001 45, 12.001 45.001, 12 45.001,...
```

Reproject the target geometry onto the frame’s CRS before selecting rows.
Use `fields.geometry.within(shape)` to keep features contained by the target.

Selection keeps matching rows and does not clip geometries; use GeoPandas
`.clip(...)` for that.

## Save and reopen the table

```python
import tempfile
from pathlib import Path

root = Path(tempfile.mkdtemp())
path = fields.gs.to_geoparquet(root / "fields.parquet")
reopened = gs.open_vector(path)
reopened
```

Output:

```text
    name  class                                           geometry
0  north      1  POLYGON ((12.001 45, 12.001 45.001, 12 45.001,...
```

GeoParquet preserves geometry, CRS, and ordinary table columns.
`.gs.to_geojson()` and `.gs.to_geopackage()` write the other supported vector
formats. Edit a table with normal GeoPandas operations.

Writers return a destination `UPath` and refuse an existing file unless
`overwrite=True`. GeoJSON is written in WGS84.

[Catalogs and windows](../guides/catalogs-and-cuts.md) use the same native table
type to describe raster files and selections. See the
[vector API](../reference/vectors.md) for all options.
