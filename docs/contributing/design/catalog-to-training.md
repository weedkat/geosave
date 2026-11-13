# Catalog → training: design

Status: open design, 2026-11-13. Replaces the A/B/C proposals of 2026-11-12.
The structures and their order are settled; the mechanisms marked open are not.

## The chain (settled)

```text
STAC JSON          documents: Collection, Item, Asset
   ⇅
stac-geoparquet    the same documents, columnar on disk
   ⇅
GeoDataFrame       Item rows; geometry and time index them; choose, group, pair; no pixels
   ⇅  load ↓  publish ↑
DataTree           the STAC hierarchy with pixels: collection / item / asset
   ⇅  combine ↓  expand ↑
Dataset            ODC raster: one 2-D grid (y, x), labeled extra dims, coords and attrs
   ⇅
DataArray          one variable
   ⇅  strip labels ↓  wrap_xr ↑
numpy / tensor     bare array, what the model sees
```

Each conversion moves one step. Going down narrows or decides (which rows,
which grid, which labels to drop); going up restores what was dropped.

## The STAC tree (settled 2026-11-13)

```text
/                          catalog      nothing of its own
├── /s2                    collection   attrs: StacCollection
│   ├── /20250601T...      Item         attrs: the Dataset's other attrs
│   │   │                               coords: time, start/end_datetime, stac_properties
│   │   ├── /B04           asset        B04 + its attrs + its grid (y, x, spatial_ref)
│   │   └── /SCL           asset        SCL on another grid
└── /labels                collection   another layer, on its own timeline
```

| Level | From a Dataset (`ds.gs.expand(item_dim)`) | Back (`tree["s2"].gs.combine(assets)`) |
| --- | --- | --- |
| collection | `StacCollection` attrs; named by its id, else `default` | Dataset attrs |
| Item | one per `item_dim` label, named by its id (`ITEM_ID_FORMAT`: start, MGRS square, centroid); the coords along `item_dim`; the other Dataset attrs | `merge` its assets; stack the Items as one dask array per variable (`utils.stack`) |
| asset | one per variable: its attrs and its grid | merged per Item; variables without `item_dim` are left out by `expand` |

| Rule (`core.raster.datatree`) | Requires | Used by |
| --- | --- | --- |
| `stac_catalog` | every child a STAC collection | tests of `expand` |
| `stac_collection` | data only on leaves, each at Item/asset | `combine` |
| `same_grid` | one CRS, resolution and pixel alignment; extents may differ | `combine` (union extent, NaN outside an Item) |
| `same_geobox` | one GeoBox: grid and extent | the tree's `geobox` |
| `same_pixel_meaning` | each asset states one `CFPacking`, `CFFlags`, units across Items | `combine` |

Settled behaviour: the Item node name is the Item id, chosen by whoever
creates the node (`expand` formats it from the Item's region,
`load_datatree` keeps the Item's own), so node paths are file paths
(`tree.gs.to_cog`); a missing asset or uncovered extent is NaN; assets on several grids are
resampled explicitly (`tree.gs.reproject`) or chosen with `combine(assets=...)`;
collections at level 1 are stacking layers, each combined on its own.

Load is the mirror of publish: rows → tree → Dataset, and Dataset → tree → rows.

## Tree ↔ catalog (2026-11-13)

```python
rows = scene.gs.expand().gs.to_catalog("out")          # COGs at node paths + one row per Item
path = rows.gs.to_geoparquet("catalog.parquet")        # hrefs stored relative to the file
tree = open_vector(path).gs.to_datatree()              # attrs["source"] resolves them
scene_back = tree["default"].gs.combine()
```

| Piece | Owner |
| --- | --- |
| Item node → Item | `stac.publish.StacItem.from_datatree(item, id, collection, properties, geometry, stac_extensions)`: time from the node's scalar coords, footprint from asset GeoBoxes (4 points a side) |
| asset → Item asset | `StacItem.add_raster_asset(key, raster, href, roles, media_type, title, with_proj, with_raster, with_eo)`: added first, so extensions declare their schema on the Item |
| collection node → Collection | `stac.publish.create_stac_collection`: the stated `StacCollection`, else one spanning its Items |
| table rules | `core.vector.catalog`: `valid_geometry`, `stac_items`, `stac_collections`, `one_collection`, `dated` |
| relative hrefs | `open_geoparquet` sets `attrs["source"]`; `stac.catalog.to_item(s)` resolves through pystac |

Collections live in the stac-geoparquet file footer (`collections` map). The
spec allows several per file but advises one schema per file; one file per
collection is the intended layout, not yet enforced.

## The loop

```text
raw catalog ──recipe (Prefect flow)──▶ LitData store + sample catalog ──▶ Lightning
                                                                             │
new catalog ◀── expand · to_catalog ◀── merge by grid ◀── predictions ◀─┘
```

Each flow produces a catalog. Preparation is paid once, so training speed
does not depend on how fast catalog → tree → Dataset → tensor is.

## A sample

A sample is one chip of a tree with its labels removed. Only what varies
between samples travels with it. Everything constant is in the STAC
Collection kept next to the store.

| Per sample, in the LitData record | Per store, in the Collection |
| --- | --- |
| pixels per node | variable names, dims order |
| `crs`, `transform` (shape is the array's) | dtype, nodata, scale, offset |
| `time` per node | flags and classes, units |
| source Item ids | |

```python
grid = GeoBox(pixels.shape[-2:], Affine(*sample["transform"]), sample["crs"])
raster = wrap_xr(pixels, grid)          # back up the chain
row = grid.extent.to_crs(4326).geom     # a sample catalog row
```

## Open

| Mechanism | Direction | Not yet known |
| --- | --- | --- |
| Rows → trees | group rows by overlapping geometry, one tree per region (`iter_rasters`) | the grouping rule |
| Load speed | the conversions are quick; reading through a large dask graph is the cost | graph build vs compute, per region vs the whole selection |
| Chips | input a tree on one grid; indexing returns a chip tree; merge takes the index and the raw array | deferred |
| `tree.gs.to_sample()` | returns the dict LitData stores | nested dicts in LitData; collating `crs` and `time` |
| Store index | the sample catalog is built from the sample metadata while writing | whether `optimize` keeps input order |
| Recipe | Prefect flow, one task per region | one store from many tasks |
| Pairing sensors, time windows | pandas over rows, before loading | |

## Set aside

| Idea (2026-11-12) | Why |
| --- | --- |
| Virtual cube: the whole catalog as one Dataset per collection on one grid, read by a custom `BackendArray` | skips the tree; a second laziness mechanism beside dask and ODC |
| `select_times`, `select_items` | patches for a time axis holding every acquisition |
| Compositing onto a calendar in the library | project policy; at most a recipe |

## Findings kept

| Finding | Consequence |
| --- | --- |
| `Dataset.load()` computes in place | docs and templates use `.compute()` |
| Collections mixed in one parquet union every asset key | one file per collection; read the directory |
| DataLoader (forkserver) and LitData (spawn) re-import the script | Datasets live in modules; nothing runs at import |
| A 200k-row catalog pickles to 302 MB | workers open the catalog by path |
