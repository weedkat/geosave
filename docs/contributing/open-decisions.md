# Open decisions

Proposals not yet settled. Settled design lives in [architecture](architecture.md).

| Topic | Proposal |
| --- | --- |
| Per-Item band metadata | Deferred (YAGNI): band attrs cannot vary along `time`; a per-variable coordinate would be needed. Differing scale/offset is refused at load. |
| STAC publication | `to_item` / `to_catalog` raise `NotImplementedError` on every raster accessor; rebuild on `expand_stac` and `tree.gs.to_cog`. |
| GCP rasters | `gs.geobox` raises `NotImplementedError`; a Sentinel-1 extension may add support. |
| Explorer snapshots | Export a static view via HoloViews `save` reusing `Map._draw`, without a live kernel. |
| Example notebooks | `examples/00_basics.ipynb` and `01_model.ipynb` still call `gs.write_grid`; `tests/examples` fails on them until they are migrated. |

## Next: catalog and Items

The raster core is settled (see [architecture](architecture.md#raster-structure)):
ODC structure in memory, time cells as `start_datetime`/`end_datetime`,
STAC metadata on the object (`stac_properties` coordinate, `StacCollection`
attrs), band attr groups, CF only in files. The catalog work builds on it:

| Area | State | Question |
| --- | --- | --- |
| Item table (`core/vector/catalog.py`, `item.py`, `samples.py`) | One stac-geoparquet row per Item; `row.gs.to_item()`, `rows.gs.to_items()`, `rows.gs.to_datatree()`, `row.gs.read_pixels(view)` | Review against the 2-D `.gs` scope, as the raster accessors were. |
| Loading (`stac/load.py`) | `load_dataset` (one grid, ODC), `load_datatree` (collection → Item → asset, native grids) | Keep both? `load_dataset` needs a GeoBox, tree reprojection takes a CRS + resolution; harmonize. |
| Tree ↔ Dataset | `ds.gs.expand_stac(format_id=, asset=)`, `tree.gs.combine_stac()` | Settled shape; publication should read this tree. |
| Publication (`stac/publish.py`, raster `to_item`/`to_catalog`) | `create_stac_item` describes saved files; raster publication paused | Rebuild as tree → `tree.gs.to_cog(path)` → Items from the written files and the tree's STAC metadata. |
| Band metadata on publish | `CFPacking` → `raster:scale/offset/nodata`, `CFFlags` + `Display.color_map` → `classification:classes`, `EOBand` → `eo` | Regenerate from groups; no per-Item variation (see above). |
