# Cuts as self-contained window rows

> Historical proposal, partially implemented. Window tables, `row.gs.to_tree`,
> cuts, and prediction merging are available. The proposed `ChipWriter` and
> task-owned prediction flow are not implemented. Current STAC creation uses
> singular `create_item`, writers return `RasterFile` records, and source
> Datasets are supplied by callers. The module paths and planned removals
> below record an earlier design. See [Catalogs and cuts](../../guides/catalogs-and-cuts.md)
> and [Training contracts](../../concepts/training.md) for current behavior.

## Goal

`cuts` turns an item table into a table of windows. Each row reads into an
aligned, lazy stack with `row.gs.to_tree()`, and traces back to the items it
came from. Stacks are found by footprint, not by a stored `geosave:stack`
property. Every chip a model predicts is written back as a georeferenced COG,
merged per frame, and catalogued as a new collection.

Users see GeoDataFrames, xarray objects and the `gs` accessor. pystac Items stay
internal.

```python
import geosave as gs

items   = gs.open_vector("data/catalog.parquet")                  # any collections, any dates
stacks  = gs.cuts.stacks(items, on="label")                        # one row per label item
frames  = gs.cuts.frames(stacks, 4, tolerance="10D")               # each stack split in time
windows = gs.cuts.chips(frames, 224, overlap=32)                   # each frame split in space

chip = windows.iloc[0].gs.to_tree(resampling={"s2": "bilinear"})  # aligned lazy DataTree
```

## Decisions

| Topic | Decision |
|---|---|
| Stack instance | One row per item of the anchor collection `on`. |
| Membership | An item joins a window when its footprint **covers** the anchor footprint. A member never covers only part of the window, so no mosaic is needed. |
| Same instant twice | If two items of one group cover the window at the same instant, `stacks` keeps the first in table order and emits `DuplicateInstantsWarning`. |
| Item instant | `datetime`, or `start_datetime` when the Item carries a range. |
| Grid | The anchor item's grid (CRS, transform, shape). Other groups are reprojected onto it at read time. |
| Alignment | `to_tree` always returns every group on the window grid. Groups keeping native resolution are out of scope. |
| Loading | Existing transforms per group: `open_raster(href)` → `reproject(chip grid)` → `merge_bands` for an Item's band assets → `concat_time` across Items. Not odc-stac, which drops band names and attrs. |
| Items | One Item = one scene of COG/GeoTIFF files. Zarr and NetCDF are datacube storage and never become Items. |
| Band layout | By default, one multi-band COG per scene. `split_bands=True` gives one asset per band, for mixed resolutions or selective fetch. Both are common STAC conventions. |
| Band metadata | Every asset writes `eo:bands` through pystac's `EOExtension`, naming every variable in file order. Common names and wavelengths are added where the raster is spectral. The STAC 1.1 core `bands` is not written: pystac 1.14 has no API for it, and the smoke test showed odc-stac 0.5.2 ignores it. |
| Index | Plain `RangeIndex`. `parent` holds the input row's index position. |
| Trace back | Each member Item dict in `sources` keeps its `id` and `collection`. |
| Chip boundary | `chips` lays tiles with `tiler.Tiler` via `layout()`. A chip row's grid columns are that tile's bbox placed on the ground. |
| Halo | `to_tree` loads the part of the chip inside its parent, then pads the rest lazily with `mode` (`reflect` by default). |
| Anchor properties | A window row carries the anchor item's own properties (e.g. annotations) as columns, so model context reads them from the row as it does today. STAC structural fields (`type`, `stac_version`, `stac_extensions`, `assets`, `links`, `bbox`, `id`, `collection`, time fields) are excluded. |
| Georeferenced only | Windows come from item tables, so every window has a CRS. In-memory and unreferenced parents are dropped from `cuts`; tests write their rasters as COGs first. |
| Static group | A group contributing exactly one item to a window joins every frame whole (label, DEM). |
| Merge | `cuts.merge` undoes `chips` for logits only, with `tiler.Merger` overlap-add. Class maps are `argmax` after merging. It is not a general mosaic. |

## Window row

Tables build stacks and rows read them. `cuts.stacks`, `frames` and `chips`
map a table to a table and open nothing. `row.gs.to_tree()` opens one row.
`GeoVector.to_tree`, which read many item rows as one sample, is removed.

A window row uses STAC projection names for its grid, but it is not a STAC
Item. An Item is one scene with fixed asset keys, while a window points at many
scenes per group, a set that differs per frame and chip. stac-geoparquet stores
`assets` as one struct whose fields are the union of every row's keys. Scene-named
keys across 100k chips would give thousands of mostly-null fields, so the
members live in `sources` as JSON.

```text
anchor           str               id of the anchor Item the window was built on
parent           int | None        index position of the row it was cut from
sources          str (JSON)        {group: [STAC Item dict, ...]}, trimmed to this window
proj:code        str               anchor grid CRS, as the STAC projection extension names it
proj:transform   list[float]       6 affine terms of this row's grid
proj:shape       [int, int]        this row's pixels, height and width
start_datetime   Timestamp | None  span of the dated members
end_datetime     Timestamp | None
geometry         Polygon           window footprint, EPSG:4326

chips add:
chip             int               tile id in the layout; `Merger.add` key
halo             [[int, int], [int, int]]   padding the layout put around the parent
overlap          int               layout setting; with the parent shape and chip size it rebuilds the Tiler
mode             str               how pixels past the parent are filled
pad              [[int, int], [int, int]]   this chip's pixels past the parent edge, top/bottom and left/right
```

`sources` is JSON text (orjson). The smoke test showed that a nested dict column
comes back from parquet as numpy arrays inside structs whose keys are the union
across rows, which `pystac.Item.from_dict` rejects. JSON text round-trips exactly.

Each Item dict is stripped to what loading needs: `id`, `collection`, `geometry`,
`bbox`, `properties` time fields, and `assets` with their extension fields.
`links` and other properties are dropped. Hrefs are made absolute in `stacks`, so
a saved windows table reads anywhere. Chips of one frame share one string object,
which parquet dictionary-encodes once.

## Cuts API

```python
def stacks(items: gpd.GeoDataFrame, *, on: str) -> GeoDataFrame:
    """One window per item of collection `on`, joined by every item covering it."""

def frames(windows, length, *, tolerance, stride=None, mode="strict") -> GeoDataFrame:
    """Split each window in time; each frame keeps only the member items of its instants."""

def chips(windows, size, *, overlap=0, mode="reflect") -> GeoDataFrame:
    """Split each window in space; rows copy `sources`, grid columns change."""

def merge(parents, chips, predictions, *, taper=None) -> dict[int, Dataset]:
    """Overlap-add saved chip logits onto the grid of each parent they were cut from."""
```

`frames` keeps today's joint-axis algorithm (`_joint_axis`, `_spans`). Each
group's time labels are now the instants of its member items, and a frame keeps
the items whose instants it uses.

### `GeoSample`: reading a row

`row.gs.to_tree(*, resampling=None, chunks=None)` is a pandas **Series**
accessor, `GeoSample`, in `cuts/sample.py`. It raises `AttributeError` for a
Series without `sources`, the same way the DataFrame accessor rejects
non-GeoDataFrames.

```python
grid = GeoBox((height, width), Affine(*transform), crs)
inner = grid shrunk by the chip's part of `halo` past the parent edge
groups = {}
for group, members in orjson.loads(sources).items():
    kernel = resampling.get(group, "nearest")
    scenes = []
    for item in members:                                     # one Item = one scene
        bands = [reproject(open_raster(asset["href"], chunks={}), inner, resampling=kernel)
                 for asset in data_assets(item)]
        scenes.append(merge_bands(bands) if len(bands) > 1 else bands[0])   # split-band assets
    groups[group] = concat_time(scenes) if len(scenes) > 1 else scenes[0]   # instants of the group
return stack(groups).pad(past_edge, mode=mode)
```

Every step is an existing, tested GeoSave function, so variable names and attrs
come from the files under the metadata contract. `reproject` refuses to blend
class codes, and it reduces to an exact lazy slice on a sub-grid of the source
grid. Remote hrefs go through the same `open_raster`. odc-stac stays only inside
`StacSource`, for remote anchor-based loads.

A static group carries no `time` axis.

## Accessors: `GeoVector` and `GeoCatalog`

An item table is a vector with STAC rules. The two get separate classes but share
one `gs` name. pandas allows a factory as the accessor, and the smoke test
confirmed that it dispatches by content.

```python
# geodata/core/vector.py: pure vector, no STAC knowledge
class GeoVector:
    def query(...): ...
    def rasterize(...): ...
    def to_geoparquet(self, path, **options): ...            # plain GeoParquet
    def to_geopackage(...): ...
    def to_geojson(...): ...

# geodata/stac/catalog.py
class GeoCatalog(GeoVector):
    def to_geoparquet(self, path, *, collections=None, overwrite=False): ...   # stac-geoparquet
    def to_geopackage(self, *args, **kwargs):
        raise ValueError("an item table is STAC; GeoPackage can't hold its assets, write GeoParquet")
    to_geojson = to_geopackage

@pd.api.extensions.register_dataframe_accessor("gs")
def _gs(data: gpd.GeoDataFrame) -> GeoVector:
    return GeoCatalog(data) if _table.is_item_table(data) else GeoVector(data)
```

- **Item-table check.** `is_item_table` is structural: it needs `type`, `stac_version`, `id`, `assets`, `geometry`, and `datetime` or `start_datetime`/`end_datetime`. Schema validation (`pystac.Item.validate()`) stays in tests.
- **Writing.** `_table.write` parses every row with `pystac.Item.from_dict`, so a malformed row fails on write.
- **Collections on write.** `to_geoparquet` builds a minimal Collection for any `collection` id that `collections=` doesn't describe, with the extent taken from its rows.
- **What `GeoCatalog` folds in.** `to_stac_parquet` folds into `to_geoparquet`. A suffix-less path writes one file per collection (`write_catalog`).
- **Dependency direction.** `stac/catalog.py` imports `core/vector.py`, never the reverse. The factory is registered in `stac/catalog.py` and imported at package init.
- **Inherited `rasterize`.** On a `GeoCatalog` it burns item footprints, which gives a coverage map.

## Public API

```python
# root: open_* for every kind of data
gs.open_raster("scene.tif")
gs.open_vector("data/catalog.parquet")        # renamed from read_vector
gs.open_tree("cache/s0.zarr")

# gs.stac: build, inspect, search
forest = gs.stac.create_collection("forest", description="Forest samples")
files  = scene.gs.to_cog("data/forest_{time:%Y%m%d}.tif")                         # tuple[CogFile, ...]
rows   = gs.stac.create_items(files, collection=forest)                              # item table
files  = sample.gs.to_cog("data/s0/{collection}_{time:%Y%m%d}.tif")                 # each CogFile names its layer
rows   = gs.stac.create_items(files, id="s0")                                        # layer = collection
rows.gs.to_geoparquet("data/catalog.parquet", collections=[forest])
gs.stac.get_collections("data/catalog.parquet")                                    # {"forest": pystac.Collection}

client = gs.stac.StacClient.open("element84")
s2     = client.search(gs.stac.StacQuery(collections=["sentinel-2-l2a"], ...))     # item table

windows = gs.cuts.stacks(gs.GeoVector.concat([items, s2]), on="label")             # local + remote
```

| `gs.stac` public | Change |
|---|---|
| `create_collection` | unchanged |
| `create_item(assets, *, id, ...)` | Kept for explicitly named assets, e.g. `{"label": path}` in `workflow/tasks/labels.py`. Returns a one-row item table. |
| `create_items(files, *, id=None, collection=None, datetime=None)` | Takes the `CogFile`s `to_cog` returned and returns an item table. Files sharing `(collection, datetime, grid)` form one Item, one Asset per file: the band name when a file holds one band, `"image"` when it holds several. `eo:bands` follows `CogFile.bands` order. `collection=` applies to files naming none and is a `ValueError` otherwise. `id` is a `GeoAnchor.format` template, or a prefix for a stack's files. Undated files take `datetime=`, or the span the dated files cover, widened by TimeSpec. |
| `get_collections(path)` | New. Reads the `collections` key of a stac-geoparquet file's metadata. |
| `StacClient.open(...)` | `.search()` returns an item table. `StacSource` reads pystac Items through an internal `items(query)` method on both clients. |
| `StacSource` | New `search(anchor)` returns the item table `load(anchor)` reads; new `collection` property returns the cached `pystac.Collection`. `load` keeps `ACDD` and per-variable band attrs, and no longer writes `StacMetadata`. |
| `StacSourceConfig` | Loses `item_properties` and `asset_fields`. Per-scene values load through `with_properties` as `time` coordinates, and Item fields live in `search(anchor)`. |
| `StacQuery` | unchanged |

**Made private:**
- `stac.item` becomes `_item.py`;
- `stac.table` becomes `_table.py`;
- `StacTableClient` stays reachable through `StacClient.open("x.parquet")` but is no longer exported.

Filtering a local catalog uses the existing `open_vector(path).gs.query(anchor)`.

### `RasterFile`: what `to_cog` returns

```python
# geodata/io/raster/file.py
@dataclass(frozen=True)
class RasterFile:
    """One saved raster file and the header it states."""

    path: UPath         # carries storage_options
    header: xr.Dataset  # as the file states it, pixels replaced by empty lazy arrays
```

The writer reads the header back from the staged file before it is moved or
uploaded, because GDAL infers some of what a file states (colour
interpretation, area-or-point). `create_item` takes the headers from the records
instead of reopening each file, and opens only bare paths, through
`RasterFile.open`. The layer is the header's ACDD `id`, so no separate field
names it. The stack template field `{group}` is renamed `{collection}`, so one
word runs from layer to Item. `RasterFile` is path-like for a local file, so it
opens wherever a path does. `to_zarr` and `to_netcdf` keep returning a path,
because datacube stores are never catalogued.

`CogFile` remains only on the read side: it names a catalog asset's bands and
instant for `cogs.open_raster`, where there is no file to open.

Records read back through one reader, `io.raster.cogs.open_raster(files, geobox=None)`.
Files of one instant merge band by band, and instants join along `time`.
`open_raster(files)` and `open_tree(files)` take the records `to_cog` returned,
and `GeoSample.to_tree` reads a row's member Items by turning their assets into
the same records (`stac.item.cog_files`), onto the chip grid. `open_raster`
otherwise reads one path, and folders of COGs are no longer read.

A metadata record of how a raster was loaded is not kept on the raster. The load
recipe, including `groupby`, is declared in the model spec's
`rasters.<name>.stac.load`.

## Inference

```python
trainer.predict(module, loader, callbacks=[ChipWriter("runs/pred")])  # logits COG per chip, halo included
chip_preds = gs.open_vector("runs/pred/chips.parquet")                 # Items with a `chip` property

merged = gs.cuts.merge(frames, windows, chip_preds, taper="hann")      # {frame index: Dataset}
files  = [f for i, ds in merged.items() for f in ds["logits"].argmax("band").gs.to_cog(f"runs/pred/{i}.tif")]
preds  = gs.stac.create_items(files, collection=gs.stac.create_collection("prediction"))
preds.gs.to_geoparquet("runs/pred/prediction.parquet")
merged[0]["logits"].argmax("band").gs.vectorize().gs.to_geoparquet("runs/pred/0.parquet")  # polygons for GIS
```

- **`ChipWriter`.** It writes each logits tensor through `raster(...).gs.to_cog` on the chip row's grid, plus one item table per rank.
- **`merge`.** It rebuilds the layout with `layout()` from the parent shape and the chip's size, `overlap` and `mode`. It reads each chip's COG, calls `Merger.add(chip, values)`, and finishes with `Merger.merge(extra_padding=halo)`.
- **The prediction table** is an ordinary item table. Concatenated with the original catalog, `cuts.stacks(..., on="label")` puts prediction and label in one window for evaluation.

## Module layout

```text
geodata/cuts/
├── stacks.py   frames.py   chips.py   merge.py     one public function each
└── sample.py                                       GeoSample Series accessor
geodata/stac/
├── _item.py        Items from saved files
├── _table.py       Items <-> rows, write, collections, is_item_table
├── catalog.py      GeoCatalog + accessor factory
├── client.py       StacClient (search returns rows via _table), _StacTableClient
├── source.py       unchanged
└── extensions/     minus geosave, zarr
```

Dependencies: `cuts` → `stac._table` → pystac. `cuts` never imports `client`.

## Removed

- `geosave:stack`: `stac/extensions/geosave.py`, its schema, and the stamping in `create_stack_items`.
- `create_stack_items`, `stac.table.load`, and `create_items`' path, sequence and mapping input forms, together with `_items`' timespan grouping.
- `StacMetadata` and `StacItem` attrs models (root `stac_items`, `stac_groupby`), and `StacSourceConfig.item_properties` and `asset_fields`.
- The datacube STAC extension (`stac/extensions/datacube.py`): one instant per COG makes it repeat the Item's `datetime`.
- `cuts.select`, `cuts.select_times`, `cuts.select_pixels`.
- `GeoVector.to_tree` (item rows to stack); `row.gs.to_tree()` replaces it.
- `GeoVector.to_stac_parquet`; `GeoCatalog.to_geoparquet` replaces it.
- Multi-path and folder `open_raster`, and `_combine_rasters`. `open_raster` reads one path.
- `open_tree` on COG path lists; it reads Zarr/NetCDF stacks only.
- `.zarr` and `.nc` in `_MEDIA_TYPES`, and `stac/extensions/zarr.py`.
- The `times` column and string ids (`s0/frame-0/chip-3`).
- `read_vector`, renamed to `open_vector` file by file, never by regex.


## Consumers updated

- `ModelSpec.prepare(raw, output, *, on, keep)`: iterate `cuts.stacks(raw, on=on)`, read each row with `to_tree`, preprocess, write COGs, and catalogue them with `create_items`.
- `ml/segmentation/supervised/data.py` `Dataset`: built from a windows table; `__getitem__` is `windows.iloc[i].gs.to_tree()`.
- `ml/segmentation/callbacks.py` `ChipWriter`: a logits COG per chip plus an item table.
- `model/spec/cuts.py`: forwards to the new `frames` and `chips` signatures.
- Docs, README, AGENTS.md, examples and workspace templates using any removed or renamed name.

## Smoke tests (2026-10-10)

The setup: a 10 m label with 512 px, and S2 on a shifted 20 m grid with 3 dates. S2 was saved as split-band COGs, giving one Item with 2 assets per date.

- `odc.stac.load` on a 224 px chip geobox is lazy and reprojects correctly. On a 3-band COG written by `to_cog`, however, it names the variables `image.1`, `image.2` and `image.3`, and keeps only `nodata` in attrs. That rules it out for `to_tree`.
- `open_raster(href, chunks={})` → `reproject(chip)` keeps `red`, `nir` and `swir` and their attrs. It stays lazy, gives 0 differing pixels against the source slice on a same-grid chip, and stays lazy for 10 m → 20 m.
- `.pad(mode="reflect")` stays lazy.
- A nested-dict `sources` column fails the parquet round trip. JSON text round-trips, and the reloaded Items load again.
- A factory function registered as a DataFrame accessor returns the subclass for an item table, and the base class for a plain frame or a column subset.
- odc-stac 0.5.2 names a multi-band asset's variables from `eo:bands` names (`red`, `nir`, `swir`). With only the STAC 1.1 core `bands` it falls back to `image.1..3`.

## Tests

Tests mirror the source tree, under `tests/geodata/cuts/`, `tests/geodata/stac/`, `tests/geodata/core/` and `tests/ml/`.

**`stacks`**
- One row per anchor item.
- An item covering only part of the anchor does not join.
- A duplicate instant keeps the first and warns.
- Hrefs are absolute.
- `sources` Items are stripped.

**`frames`**
- Members are trimmed per frame.
- A single-item group joins every frame.
- A ranged Item uses `start_datetime`.
- Existing joint-axis tests pass on item instants.

**`chips`**
- Chip grids equal the Tiler bboxes on the ground.
- `sources` is unchanged.
- `parent` indexes the frame row.

**`to_tree`**
- Every group is on the chip grid.
- The result stays lazy.
- A reflect halo appears only past the parent edge.
- Split-band items load as one Dataset.
- Variable names and attrs match `open_raster` of the source file.
- A non-window Series raises `AttributeError`.

**`merge`**
- Merging saved chip COGs equals merging the same outputs in memory.
- Chips may arrive in any order.

**Accessors**
- `gs` is a `GeoCatalog` on an item table and a `GeoVector` otherwise.
- `GeoCatalog.to_geopackage` raises.
- `to_geoparquet` on an item table writes stac-geoparquet, and the rows validate with `pystac.Item.validate()`.

**`stac`**
- `create_items` returns an item table, maps `{group: paths}` to collections, and rejects `.zarr` and `.nc`.
- `get_collections` returns the stored Collections.
- A multi-band asset lists its band names in `eo:bands`, spectral or not.
- `odc.stac.load` of a written catalog names its variables after them, not `image.1..N`.
- `StacClient.search` returns an item table.

**Persistence**
- A windows table round-trips through GeoParquet, and a reloaded row reads the same pixels.
- A windows table of 1k chips stays small on disk.
- A prediction chip COG round-trips onto its chip grid.

## Breaking changes

`read_vector`, `geosave:stack` catalogs, `create_item` returning a pystac Item, `create_stack_items`,
`stac.table`, `stac.item`, `StacTableClient` export, `to_stac_parquet`,
`open_raster([...])`, folder `open_raster`, `open_tree` on COG lists,
`GeoVector.to_tree`, `table.load`, `cuts.select*`, string window ids,
`StacClient.search` returning pystac Items, `ModelSpec.prepare` without `on`, and
Zarr-backed prepared catalogs.

## Remaining risks

- Each chip opens its COGs anew, so a training epoch opens each COG many times. The GDAL block cache and the dataloader workers need checking in the first real fit.
- The `covers` rule drops an item that misses the anchor by a sliver, for example when a scene footprint is simplified. The drop happens with no message unless `stacks` reports items that intersect without covering.
- A misspelled `on=` collection gives no windows. `stacks` raises when `on` names no collection in the table.
