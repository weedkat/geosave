# Open decisions

Proposals not yet settled. Settled design lives in [architecture](architecture.md).

| Topic | Proposal |
| --- | --- |
| Per-Item band metadata | Deferred (YAGNI): band attrs cannot vary along `time`; a per-variable coordinate would be needed. Differing scale/offset is refused at load. |
| ModelSpec | Disabled pending redesign; old ingestion/processing helpers removed. Stage construction and chain/weight releases remain active. |
| Prediction decoding | `softmax_argmax` raises `NotImplementedError` pending redesign; calibration depending on it is unavailable. |
| Publication gaps | `tree.gs.to_catalog(path)` publishes and `rows.gs.to_datatree()` loads back (see design). Open: Item node attrs (`title`) are published nowhere — Item properties, or collection-level in `expand`?; a stated upstream Collection is reused verbatim for a derived product; the minimal Collection's `license`/`description` are placeholders; decoded variables (packing only in `encoding`) publish no `raster:bands` scale; the GeoTIFF header spells float32 scales longer than the asset does; `CFFlags` → `classification:classes` and `with_eo` are untested; Dataset/DataArray `to_item`/`to_catalog` and Dataset `to_cog` still raise; an explicit band array writes one TIFF. |
| Tree overview figure | `tree.gs.plot` was removed (one panel per asset does not scale to a catalog). Plot `tree["<collection>"].gs.combine()`; decide later whether a tree needs its own figure or explorer view. |
| Catalog accessors | Still stubbed: `rows.gs.to_dataset`, `sample`, `rasterize`; `row.gs.read_pixels`; `GeoSamples` (to take a STAC DataTree). The explorer's pixel view of catalog rows raises meanwhile (7 tests skipped). `concat_catalogs` refuses tables read from different files (each resolves hrefs against its own `attrs["source"]`); alternative: make hrefs absolute on concat. One stac-geoparquet file per collection (spec advice) is not enforced yet. |
| GCP rasters | `gs.geobox` raises `NotImplementedError`; a Sentinel-1 extension may add support. |
| Explorer snapshots | Export a static view via HoloViews `save` reusing `Map._draw`, without a live kernel. |
| Segmentation templates | `src/templates/tasks/segmentation/{supervised,fixmatch}` still import the disabled `ModelSpec`; `geosave add-task` generates code that fails on import. Rebuild from the Dynamic World workspace once its Dataset and Sample settle. |
| Example notebooks | `examples/00_basics.ipynb` and `01_model.ipynb` still call `gs.write_grid`; `tests/examples` fails on them until they are migrated. |

## Next: catalog and Items

The path from catalog to training is designed in
[catalog → training](design/catalog-to-training.md).

The raster core is settled (see [architecture](architecture.md#raster-structure)):
ODC structure in memory, time cells as `start_datetime`/`end_datetime`,
STAC metadata on the object (`stac_properties` coordinate, `StacCollection`
attrs), band attr groups, CF only in files. The catalog work builds on it:

| Area | State | Question |
| --- | --- | --- |
| Item table (`core/vector/catalog.py`, `item.py`) | One stac-geoparquet row per Item; `validate(*rules)` with `valid_geometry`, `stac_items`, `stac_collections`, `one_collection`, `dated`; `stac.catalog.concat_catalogs`; `to_item(s)`, `to_datatree` load | `to_dataset`, `read_pixels`, sampling to rebuild. |
| Loading (`stac/load.py`) | `load_dataset` (one grid, ODC), `load_datatree` (collection → Item → asset, native grids) | Keep both? `load_dataset` needs a GeoBox, tree reprojection takes a CRS + resolution; harmonize. |
| Tree ↔ Dataset | Settled: `ds.gs.expand(item_dim, id_format)` ↔ `tree["<collection>"].gs.combine(assets)`, see [design](design/catalog-to-training.md#the-stac-tree-settled-2026-11-13) | — |
| Tree ↔ catalog | `tree.gs.to_catalog(path)` (COGs at node paths, `StacItem`, `create_stac_collection`) ↔ `rows.gs.to_datatree()`; round trip tested | See "Publication gaps" above. |
| Band metadata on publish | `CFPacking` → `raster:bands`, one-variable `long_name` → asset `title`; `EOBand` → `eo:bands` with `with_eo` | `CFFlags` → `classification:classes` not yet wired in `add_raster_asset`. |
