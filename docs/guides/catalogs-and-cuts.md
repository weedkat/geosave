# Build catalogs and read samples

Raster publication returns a native GeoDataFrame. Its rows describe the
assets written by the path format:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pystac
import xarray as xr

import geosave as gs

root = Path(tempfile.mkdtemp())
from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 500320, 5000320),
    crs='EPSG:32633', resolution=10,
)
dates = pd.date_range("2025-01-01", periods=3, freq="MS")
optical = (
    xr.Dataset(
        {"red": (("time", "y", "x"), np.ones((3, 32, 32), dtype="uint16"))},
        coords={"time": dates},
    )
    .gs.write_grid(grid)
    .gs.write_nodata(0)
)
# Any extent will do: to_catalog sets it to cover the Items it writes.
everywhere = pystac.Extent(
    pystac.SpatialExtent([[-180, -90, 180, 90]]), pystac.TemporalExtent([[None, None]])
)
collection = pystac.Collection("optical", "Monthly imagery", everywhere)
items = optical.gs.to_catalog(
    root, path_format="{collection}/{time:%Y%m%d}/{band}.tif", collection=collection
)
items[["id", "collection", "datetime"]]
```

`{collection}` uses the supplied collection id. `{time}` splits dates and
`{band}` splits variables; omitting `{band}` keeps compatible bands together.
The same path collision checks as `to_cog` reject filename collisions before writing.
A DataTree uses its collection names for `{collection}` and can take a mapping
of collection names to path formats when dated and undated collections need
different paths. A collection on several grids, such as Sentinel-2's 10 m and
20 m bands, publishes one Item per date whose assets come from every grid, each
stating its own grid in `proj:*`; `{band}` or `{res}` keeps the grids' files apart.

Catalog assets are COGs, so path formats end in `.tif`; a Zarr or netCDF store is
written whole with `to_zarr` or `to_netcdf`. A catalog Item represents one
observation or composite, so multi-date publication needs `{time}`.

## Persist the Item table

```python
path = items.gs.to_geoparquet(root / "catalog.parquet")
items = gs.open_vector(path)
```

The writer retains Collection definitions, updates their extents, and stores
asset hrefs relative to the catalog so a folder can move with its assets.
Ordinary features and Item catalogs both use `gs.to_geoparquet(...)`. The
frame's `stac-geoparquet` metadata selects the STAC encoder. Invalid catalog
metadata or missing Item fields raise instead of silently losing information.

Every catalog row comes from a raster `to_catalog` wrote, or from a saved
catalog opened with `gs.open_vector`. A raster loaded from STAC carries its
Collection and Item properties, so its catalog keeps them; see
[STAC](stac.md#preserve-properties-and-collection-definitions).

## Group samples

```python
samples = items.gs.sample(pair=["optical"], window=2, freq="MS")
trees = []
for rows in samples:
    trees.append(rows.gs.to_datatree())
```

Sampling groups Items without opening pixels. Each distinct footprint gathers
the Items sharing at least `overlap` of its area, so a swath joins every tile
it covers. `pair` names the collections forming a time step; `tolerance`
controls nearest-date matching. `window` and `step` select sequences of those
steps. With `freq`, each time slot is a resampling bucket. Timeless Items, such
as a DEM, join each window of their footprint.

The result is a `GeoSamples`: the item table and, in `samples.indices`, the
row positions of each sample. `samples[i]` is that sample's rows of the item
table, so `samples[i].gs.to_datatree()` opens a lazy DataTree, `/{collection}/{res}`:
each collection holds one grid node per native pixel size (`r10m`, `r20m`),
retaining native grids and dates. Reading one Item row with
`items.iloc[0].gs.to_datatree()` opens its scene's grid nodes; `.gs.to_dataset()`
converts a scene, or one collection `tree["s2"]`, to one Dataset on its finest
grid, or on `res="r20m"` with `resampling="average"`.
`samples.footprints` holds each sample's footprint and Item count;
`samples.footprints.gs.explore()` maps them, and `samples[i].gs.explore()` maps
one sample's rows, whose raster column opens each row's raster. Persist the
Item catalog and sample it again after reading.

## Choose the reading grid

`rows.gs.to_datatree()` preserves original asset grids. Successive Items of a
collection must agree on their grid nodes, coordinates and variables so they
can concatenate exactly. Geographic overlap alone does not imply that their
pixels align. An accepted temporal sample can therefore fail native opening
when one acquisition is shifted.

For a selected **single collection** with differing grids, choose a native
GeoBox and use the existing explicit-grid loader:

```python
# Reuse this guide's optical rows and grid; choose another grid deliberately
# when aligning your own acquisitions.
selected = samples[0]
aligned = gs.io.load_dataset(
    selected.gs.to_items(),
    geobox=grid,
    resampling="nearest",
    groupby="time",
    chunks={},
)
```

This operation explicitly aligns observations; ODC owns the reads and
resampling. `groupby="time"` keeps distinct acquisition timestamps rather than
using the loader's default solar-day grouping. Select each collection
separately for a multi-collection sample, and choose an appropriate kernel for
its variables. Categorical labels require a categorical-safe choice.
Close `aligned` and the opened trees when finished.

A shared swath's full extent can exceed the sample's geographic Region.
`Chips` currently uses the union of the opened tree's grids, so two Regions
sharing one swath can receive identical oversized layouts. Pass an explicitly
selected/cropped tree to Chips; do not assume sampling cropped the assets.
The [review](../contributing/review-2026-11-04.md#24-preserve-region-support-across-sampling-loading-and-chip-layout)
records the missing Region handoff. Interval-only composite Items also have a
[current sampling limitation](../contributing/review-2026-11-04.md#25-interpret-composite-intervals-consistently-during-sampling):
they may be mistaken for timeless Items. Do not rely on their temporal pairing
until that implementation is corrected.

## Cut raster chips

```python
chips = gs.Chips(trees, size=16, resolution=10, overlap=4)
tree_index, chip = chips[0]
```

`Chips` lays out raster trees and opens requested chips lazily. `tree_index`
identifies the input sample. Supply per-chip model outputs with
`chips.add(index, output)` and reconstruct output rasters with `chips.merge()`.
Chipping and merging belong to raster trees, independent of the GDF.

See the [catalog API](../reference/catalog.md) for the public interface and
[model input guide](model-spec.md) for preprocessing contracts.

```python
aligned.close()
for tree in trees:
    tree.close()
```
