# Work with vector features

Create a table of field boundaries, select a feature, and save it as GeoParquet.
This example uses the base package and runs offline.

```python
import tempfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

import geosave as gs
from geosave.geodata.core.vector.catalog import valid_geometry

fields = gpd.GeoDataFrame(
    {"name": ["north", "south"], "class": [1, 2]},
    geometry=[box(12.0, 45.0, 12.001, 45.001), box(12.0, 44.999, 12.001, 45.0)],
    crs="EPSG:4326",
)
fields[["name", "class"]]
```

```text
    name  class
0  north      1
1  south      2
```

Vector data represents points, lines, or polygons. Each row is one feature:
ordinary columns describe it and the geometry column locates it. `fields` is
a normal GeoPandas GeoDataFrame; GeoSave adds `.gs` operations.

If you already have a file, use `gs.open_vector("fields.parquet")`,
`gs.open_vector("fields.geojson")`, or `gs.open_vector("fields.gpkg")`.

## Check and select features

```python
fields.gs.validate(valid_geometry)
selected = fields.loc[fields["name"] == "north"]
selected["name"].tolist()
```

```text
['north']
```

Validation applies only the rules you pass. `valid_geometry` checks for
present, nonempty, valid geometries and returns None on success, without
changing the table. Failures raise `gs.VectorValidationError`.
Use GeoPandas for filtering, clipping, and reprojection:

```python
area = box(12.0, 45.0002, 12.001, 45.0008)
nearby = fields.loc[fields.geometry.intersects(area)]
projected = nearby.to_crs("EPSG:32633")
```

The selection geometry uses the table's CRS, here longitude/latitude degrees.
`intersects` selects whole features; `fields.clip(area)` clips their geometry.
Assigning a CRS with `.set_crs(...)` declares known coordinates; `.to_crs(...)`
changes coordinates to another CRS.

`fields.gs.region` returns their combined geographic coverage. A
`GeoBox` separately defines pixels when a later operation needs a raster grid.

## Save and reopen

```python
root = Path(tempfile.mkdtemp())
path = selected.gs.to_geoparquet(root / "fields.parquet")
reopened = gs.open_vector(path)
reopened["name"].tolist(), reopened.crs == selected.crs
```

```text
(['north'], True)
```

GeoParquet preserves geometry, CRS, and table columns. Writers return a UPath
and refuse existing files unless `overwrite=True`. GeoJSON writes WGS84 coordinates.

## Raster labels and catalogs

Vector `.gs.rasterize(...)` is currently unavailable while catalog accessors
are rebuilt. Existing raster labels can still be opened, aligned, and used in
[training chips](training-chips.md). Raster `.gs.vectorize()` converts a selected
two-dimensional band to polygons and computes its pixels.

A STAC Item table uses the same GeoDataFrame type, with columns identifying
observations and their assets. See [catalog creation and loading](catalog.md)
for that workflow and [vector reference](../reference/vectors.md) for signatures.
