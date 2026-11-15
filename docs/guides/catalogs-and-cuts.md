# Create and reopen a raster catalog

Write three dated observations as COG assets, save their Item table, and
reopen them as a raster time series. Use this when you need files indexed by
observation. For windows of one prepared image, use
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
```

The Dataset has three time positions on one grid. Each position becomes an
Item and each variable an asset. The `time` dimension must exist; a scalar
time coordinate alone does not provide an Item dimension.

## Write assets and save the table

```python
items = optical.gs.to_catalog(root / "assets", id_format="{start:%Y%m%d}")
path = items.gs.to_geoparquet(root / "catalog.parquet")
items["id"].tolist()
```

```text
['20250101', '20250201', '20250301']
```

`Dataset.gs.to_catalog` writes COGs and returns one row per Item. Files land
at `assets/20250101/red.tif`, for example. `id_format` must produce unique
Item names. `to_geoparquet` saves the table separately. Pass `overwrite=True`
to intentionally replace existing destinations.

An Item describes an observation's footprint, date, properties and files.
Its asset hrefs point to the pixels. The GeoParquet writer stores relative
asset paths; move the table and asset folder together. If the source has
Collection metadata, that definition travels with the Items and table.
For a derived product, replace the source Collection with one describing
that product before publication. This minimal example has no Collection.

For several asset grids, use `stac.RasterStac.from_dataset` on each Dataset,
merge the staged publications by Item id, then call `to_cog` and
`stac.create_catalog`. See [STAC publication](../reference/stac.md#publication).
Static variables are included in every Item and written repeatedly.

## Select and load observations

```python
rows = gs.open_vector(path)
selected = rows.loc[rows["id"].isin(["20250101", "20250301"])]
observations = selected.gs.to_dataset(geobox=grid, chunks={})
means = observations.gs.mask_and_scale()["red"].mean(("y", "x")).compute()
means.values.tolist()
```

```text
[1.0, 1.0]
```

`to_dataset` validates the selected Item rows and loads one collection through
ODC. The supplied GeoBox chooses the exact output grid; `chunks={}` builds
lazy pixel reads. Native `stac.load_dataset(selected.gs.to_items(), ...)`
is the equivalent route for callers already working with PySTAC Items.

For several native grids, use `stac.load_native_datasets` to obtain
`(crs, resolution, dataset)` groups. Loading does not construct a STAC tree.
Choose resampling explicitly for assets that must move to a different grid.

## Combine tables and inspect metadata

Use `geosave.geodata.stac.catalog.concat_catalogs` to combine compatible
Item tables while preserving Collection definitions. Conflicting definitions
raise. Tables read from different files currently cannot be concatenated
this way because relative asset hrefs have different bases.

`rows.gs.collections` returns native PySTAC Collection copies.
`rows.gs.to_items()` converts rows to native Items, resolving asset hrefs.
Request Item properties when loading, for example
`with_properties=["eo:cloud_cover"]`; they become marked coordinates.
See [STAC search and loading](stac.md) for remote providers.

## Sampling status

Catalog sampling and row-level `read_pixels` remain unavailable. Select rows
with GeoPandas and load them with `.gs.to_dataset()` as above. For prepared
rasters, use temporal `.gs.sample` and spatial [RasterChipper](chips.md) independently.
