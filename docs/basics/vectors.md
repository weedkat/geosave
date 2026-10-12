# Work with vector features

Create a table of field boundaries:

```python
import geopandas as gpd
from shapely.geometry import box

import geosave_engine as gs

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

## Create a grid for the features

```python
anchor = gs.GeoAnchor.from_geometry(fields.gs.footprint, resolution=10)
anchor.geobox
```

Output:

```text
GeoBox((13, 9), Affine(10.0, 0.0, 263550.0,
       0.0, -10.0, 4987450.0), CRS('EPSG:32633'))
```

An anchor describes a raster grid and an optional time range. Here GeoSave
chooses a local projected CRS so `resolution=10` is in metres. The grid covers
the field footprint. For another raster's exact grid, pass that raster to
`rasterize` directly.

## Turn features into pixels

```python
mask = fields.gs.rasterize(anchor)
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
`fields.gs.rasterize(anchor, column="class", dtype="uint8", fill=0)`.
The fill value is used outside the features; choose it separately from the
valid class codes.

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
selected = fields.gs.query(anchor)
selected
```

Output:

```text
    name  class                                           geometry
0  north      1  POLYGON ((12.001 45, 12.001 45.001, 12 45.001,...
```

`query` selects intersecting features by default and handles CRS conversion.
Pass `predicate="within"` to keep rows whose geometry lies within the target.
If the target has a time range and the table has `datetime`, `start_datetime`,
or `end_datetime` columns, it also filters by time. Missing dates leave those
rows selectable.

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
formats. Use normal GeoPandas operations to edit a table; use
`gs.GeoVector.concat([first, second])` to concatenate frames. The frames must
already share a CRS; convert explicitly with `.to_crs(...)` when needed.

[Catalogs and windows](../guides/catalogs-and-cuts.md) use the same native table
type to describe raster files and selections. See the
[vector API](../reference/vectors.md) for all options.
