# Create and reopen a raster catalog

Write three dated observations as TIFF assets, save their Item table, and
reopen them as a raster time series. Use this when you need files indexed by
observation and collection. For windows of one prepared image, use
[chip reading and prediction merging](chips.md).

**You need:** the base package. This synthetic example runs offline and writes
to a temporary directory.

## Make three observations

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

import geosave as gs

root = Path(tempfile.mkdtemp())
grid = GeoBox.from_bbox(
    (500000, 5000000, 500320, 5000320), crs="EPSG:32633", resolution=10,
)
dates = pd.date_range("2025-01-01", periods=3, freq="MS")
optical = xr.Dataset(
    {"red": (("time", "y", "x"), np.ones((3, 32, 32), dtype="uint16"))},
    coords={"time": dates},
).assign_coords(xr_coords(grid)).gs.write_nodata(0)
tree = optical.gs.expand(id_format="{start:%Y%m%d}")
[leaf.path for leaf in tree.leaves]
```

```text
['/default/20250101/red', '/default/20250201/red', '/default/20250301/red']
```

`expand` makes a STAC-shaped DataTree: collection → Item → asset. Each time
step becomes an Item and each variable an asset with its own grid and metadata.
This inspects metadata without computing pixels. Variables without the Item
dimension are omitted. An explicit `id_format` chooses stable, unique Item names.

The collection is `default` unless the Dataset carries `StacCollection` attrs,
as a STAC-loaded Dataset does. For a derived product, supply a Collection
whose identity, description and license describe that product before expanding.
The automatically generated Collection is a minimal teaching default.

## Write assets and save the table

```python
items = tree.gs.to_catalog(root / "assets")
path = items.gs.to_geoparquet(root / "catalog.parquet")
items["id"].tolist(), items["collection"].unique().tolist()
```

```text
(['20250101', '20250201', '20250301'], ['default'])
```

`tree.gs.to_catalog` writes the pixels as COGs and returns a GeoDataFrame with
one row per Item. Files follow the node paths, such as
`assets/default/20250101/red.tif`. `to_geoparquet` saves the table separately.
Pass `overwrite=True` to intentionally replace existing destinations.

An Item describes an observation's footprint, date, properties and files.
An asset's `href` points to its file; the table does not contain raster pixels.
The GeoParquet writer stores Collection definitions and relative asset paths.
Move the table and its asset folder together to retain those relationships.

Dataset and DataArray `.gs.to_catalog()` are unavailable. Use the tree path
shown above. `tree.gs.to_cog(folder)` writes the assets without returning Items.

## Select and load observations

```python
rows = gs.open_vector(path)
selected = rows.loc[rows["id"].isin(["20250101", "20250301"])]
with selected.gs.to_datatree() as loaded:
    observations = loaded["default"].gs.combine()
    means = observations.gs.mask_and_scale()["red"].mean(("y", "x")).compute()
means.values.tolist()
```

```text
[1.0, 1.0]
```

`to_datatree` opens assets on their native grids. `combine` works on one
**collection node**, merges its assets, and stacks Items along `time`.
It requires matching CRS, resolution and pixel alignment; extents may differ,
in which case the result spans their union. Different collection timelines
remain separate. Consume lazy pixels before closing their source tree.

For mixed grids, pass an exact destination before combining:
`loaded.gs.reproject(grid)["default"].gs.combine()`.
For several source tiles that need mosaicking into one observation, use
[`stac.load_dataset(items, grid)`](stac.md#search-first-then-load-onto-a-chosen-grid).

## Combine tables and inspect metadata

Use `geosave.geodata.stac.catalog.concat_catalogs` to combine compatible
Item tables while preserving Collection definitions. Conflicting definitions
raise. Tables read from different files currently cannot be concatenated
this way, because relative asset hrefs have different bases.

`rows.gs.collections` returns native PySTAC Collection copies.
`rows.gs.to_items()` converts the rows to native Items, resolving asset hrefs.
See [STAC search and loading](stac.md) for remote providers and native PySTAC inputs.

## Sampling status

`rows.gs.sample()`, `rows.gs.to_dataset()`, `row.gs.read_pixels()`, and
`GeoSamples` currently raise `NotImplementedError` while their design is rebuilt.
Select rows with GeoPandas and load them with `.gs.to_datatree()` as above.
For prepared rasters, [Chips](chips.md) remains available independently.
