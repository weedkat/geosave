# Nested Raster Trees Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GeoSave's flat raster trees with `/{collection}/{res}` trees, read catalog rows into them, convert one collection to a Dataset, and lay trees out as files through one chain of template axes.

**Architecture:** A tree's leaves are grid nodes keyed by path (`"s2/r10m"`); `create_datatree` and `gs.rasters` round-trip through that path-keyed mapping, so every accessor method that already maps over `gs.rasters` keeps working. Reading opens a scene as grid nodes, takes a missing grid from the asset's `proj:*` fields, and concatenates rows per collection. Writing expands a raster through one splitter per template field.

**Tech Stack:** xarray DataTree, odc-geo, rioxarray, pystac, pytest.

**Spec:** `docs/contributing/specs/2026-11-03-nested-raster-trees.md`

## Global Constraints

- Grid node names: `r` + pixel size (`:g`) + CRS unit: `r10m`, `r20m`, `r0.0001deg`.
- A tree's leaves sit at depth 2 under collections (`/{collection}/{res}`); a scene tree's leaves at depth 1 (`/{res}`).
- `to_dataset` is the only tree → Dataset conversion; it converts one collection and refuses a node whose children are collections.
- Checks are the operations' own (`combine.merge`, `concat`, writers); methods add only: every row names a collection; a raster in a tree has a grid; `to_dataset` converts one collection.
- No `to_xarray`, no `open_scene(resolution=)`, no `resolve_group_paths`, no compatibility aliases.
- `ModelSpec` is out of scope; its call sites are renamed mechanically only.
- Edits stay uncommitted in the working tree, as for the previous plan.
- Names: verb + object for functions; variables name the value held; loops written out.
- Every changed file passes `uv run ruff check` and `uv run basedpyright`.

## Progress (2026-11-03)

- Tasks 1–7 done, uncommitted on main.
- Rulings beyond the plan: `expand_paths` keeps `collection=` for a raster that is one collection; a collision names the fields whose values differ. `writers.locate_files` replaces `resolve_paths`; `writers.write_store` writes one Zarr/netCDF store (Dataset or tree). Tree `to_cog` returns a flat `list[UPath]`; `to_zarr`/`to_netcdf` return one `UPath`. Tree stores write inherited coords per group and Zarr records group order. `StacSource.load_datatree` zooms the anchor by the ratio to the finest band family.
- Open: `ModelSpec` reads `gs.rasters` by collection name (5 tests in `tests/model/spec/test_model.py`), plus the earlier `ModelSpec.chips` failures (31) and the workspace walkthrough's `ModelSpec` YAML validation.
- Open: a per-grid template (`{collection}/{res}/{time}.tif`) with multi-variable nodes is refused (asset key `image` twice); the message points to `{band}`.

## Review Focus

- A row whose file has a grid and whose asset states another: the file wins and a warning names both.
- A timeless collection beside a dated one in one tree: the timeless leaf has no `time`; `times` and `timespan` ignore it.
- A tree of several collections: `tree.gs.to_dataset()` refuses listing them; `tree["s2"].gs.to_dataset()` works.
- A Zarr template without `{collection}`: one store with groups, readable back by `open_tree`.
- A GeoTIFF template without `{res}` or `{band}` on a mixed-grid tree: refused, naming `{res}`.

---

### Task 1: Grids for every asset, and their names

**Files:**

- Modify: `src/geosave/geodata/io/raster/stac.py` (`_open_asset` assigns a stated grid; warns on a disagreeing one)
- Modify: `src/geosave/geodata/core/ctor/datatree.py` (add `name_grid`)
- Test: `tests/geodata/io/raster/test_stac.py`, `tests/geodata/core/ctor/test_datatree.py` (create if absent)

**Interfaces:**

- Produces: `name_grid(geobox: GeoBox) -> str` → `"r10m"`, `"r0.0001deg"`
- Produces: `_open_asset` returns a raster with a grid whenever the file or the asset states one

- [ ] **Step 1: Failing tests**

```python
def test_name_grid_follows_eopf():                       # 10 m UTM -> "r10m"; 0.0001° -> "r0.0001deg"
def test_an_asset_states_the_grid_a_png_lacks(tmp_path): # PNG + proj:* -> open_scene(...).odc.geobox == stated
def test_a_stated_shape_unlike_the_file_is_ignored(...)  # proj:shape differs -> no grid assigned
def test_a_file_grid_wins_over_its_asset(...)            # GeoTIFF grid A, asset states grid B -> grid A, UserWarning naming both
```

- [ ] **Step 2:** `uv run pytest tests/geodata/io/raster/test_stac.py tests/geodata/core/ctor -k "grid" -v` → FAIL
- [ ] **Step 3: Implement.** In `_open_asset`, after opening: `stated = projection.to_geobox(asset)`; file without a CRS and `stated.shape` equal to the file's → `raster.assign_coords(xr_coords(stated))`; file with a CRS and a stated grid unequal → `warnings.warn`. `name_grid` formats `abs(resolution.x)` with `:g` and the CRS's first axis unit (`metre` → `m`, `degree` → `deg`).
- [ ] **Step 4:** same command → PASS

---

### Task 2: Path-keyed trees: `create_datatree`, `gs.rasters`, `gs.to_dataset`

**Files:**

- Modify: `src/geosave/geodata/core/ctor/datatree.py` (`create_datatree` takes `"collection"` or `"collection/res"` keys)
- Modify: `src/geosave/geodata/core/raster/datatree.py` (`groups`, `rasters`, `variables`, `to_dataset`; docstrings drop "flat")
- Test: `tests/geodata/core/raster/test_datatree.py`, `tests/geodata/core/ctor/test_datatree.py`

**Interfaces:**

- Consumes: Task 1 `name_grid`
- Produces: `create_datatree(rasters: Mapping[str, xr.Dataset | xr.DataArray]) -> DataTree` — a key without `/` becomes `key/name_grid(raster.odc.geobox)`; a raster without a grid raises naming it
- Produces: `GeoDataTree.groups -> tuple[str, ...]` (collections), `rasters -> dict[str, Dataset]` (leaf paths), `variables` (`"s2/r10m/red"`), `to_dataset(res: str | None = None, resampling: Resampling | Mapping[str, Resampling] = "nearest") -> Dataset`
- Invariant: `create_datatree(tree.gs.rasters)` equals `tree`

- [ ] **Step 1: Failing tests**

```python
def test_a_bare_dataset_becomes_its_collections_grid_node(scene):    # {"s2": ds} -> rasters keys == ["s2/r10m"]
def test_rasters_round_trip_through_create_datatree(tree)
def test_groups_are_collections(tree)                                 # ("s2", "dem")
def test_a_collection_converts_on_its_finest_grid(mixed_tree)         # tree["s2"].gs.to_dataset(): 10 m, each swir pixel repeated 2x2
def test_a_coarser_member_averages_blocks_when_asked(mixed_tree)      # res="r20m", resampling="average": red is the mean of each 2x2 block
def test_nodes_that_do_not_nest_point_to_align(shifted_tree)          # half-pixel origin shift -> ValueError naming align
def test_conversion_stays_lazy(mixed_tree)                            # dask-backed nodes -> no tasks run
def test_several_collections_refuse_listing_them(tree)                # tree.gs.to_dataset() -> ValueError naming s2, dem
def test_a_collections_metadata_reaches_its_dataset(tree)             # StacMetadata on /s2 -> to_dataset().attrs
def test_a_raster_without_a_grid_is_refused(png_dataset)
```

- [ ] **Step 2:** `uv run pytest tests/geodata/core -k "tree or datatree" -v` → FAIL
- [ ] **Step 3: Implement.** `create_datatree` builds `xr.DataTree.from_dict` from path keys and copies each collection's `StacMetadata` attrs to its node. `rasters` walks `tree.leaves` and keys by `leaf.path.lstrip("/")`. `to_dataset`: refuse when any child has children; pick the member (named by `res`, else the finest); check each node nests in it (same CRS, and one grid is the other `zoom_out(k)` for a whole k); `warp.reproject` each node not on it with `resampling`; `combine.merge` the nodes, each with its ancestors' attrs up to this node.
- [ ] **Step 4:** run every caller of `gs.rasters` and `create_datatree`: `uv run pytest tests/geodata/transform tests/geodata/core -q` → PASS (fix `warp`, `temporal`, `extent`, `chips` only where a test shows they assumed depth 1)

---

### Task 3: Reading rows as trees

**Files:**

- Modify: `src/geosave/geodata/io/raster/stac.py` (`open_scene` returns a scene tree; delete `resolution`, `_warp_to_resolution`)
- Modify: `src/geosave/geodata/core/vector/item.py` (`to_datatree`; delete `to_xarray`)
- Modify: `src/geosave/geodata/core/vector/catalog.py` (`to_datatree`; delete `to_xarray`)
- Modify callers: `src/geosave/geodata/viz/explore/layers/vector.py` (`row.gs.to_datatree().gs.to_dataset()`), `src/geosave/model/spec/model.py` and `src/geosave/model/README.md` (rename only), docs naming `to_xarray`
- Test: `tests/geodata/io/raster/test_stac.py`, `tests/geodata/core/vector/test_item.py`, `tests/geodata/core/vector/test_catalog.py`, every test calling `to_xarray`

**Interfaces:**

- Consumes: Tasks 1–2
- Produces: `open_scene(item, **options) -> xr.DataTree` (`/{res}` nodes, root holds the Item's date, properties)
- Produces: `GeoCatalogItem.to_datatree(**options) -> DataTree` (root stamped with `read_stac_metadata`)
- Produces: `GeoCatalog.to_datatree(**options) -> DataTree` (`/{collection}/{res}`)

- [ ] **Step 1: Failing tests**

```python
def test_a_mixed_resolution_row_opens_on_its_native_grids(tmp_path)   # write_mixed_scene -> rasters keys ["r10m", "r20m"], no resampling
def test_a_single_grid_row_converts_to_its_scene(...)                  # row.gs.to_datatree().gs.to_dataset() equals the written Dataset
def test_rows_stack_along_time_per_collection_and_grid(tiles)          # rows.gs.to_datatree()["s2/r10m"].sizes["time"] == 5
def test_every_row_needs_a_collection(catalog)                         # ValueError "collection"
def test_closing_the_tree_closes_every_file_once(...)
```

- [ ] **Step 2:** run them → FAIL
- [ ] **Step 3: Implement.** `open_scene`: open each asset (Task 1), group the dated files by `name_grid(raster.odc.geobox)`, `combine.merge` each group, build the scene tree with `xr.DataTree.from_dict({name: dataset})` (not `create_datatree`, whose bare keys are collections), keep the `ExitStack` closer. On a scene tree `gs.groups` and `gs.rasters` read the grid nodes, its direct children. `GeoCatalog.to_datatree`: refuse rows without a collection; per collection, sort rows by `read_instant`, open each row's tree, `concat` them (trees concatenate node by node), stamp `read_stac_metadata` on the collection node; join collections with `create_datatree`.
- [ ] **Step 4:** replace every `to_xarray` call; `uv run pytest tests/geodata tests/flows/test_ingest.py -q` → PASS

---

### Task 4: COG layout and writing; stores take no template

**Files:**

- Modify: `src/geosave/geodata/io/raster/paths.py` (`expand_paths(raster, template: str | Mapping[str, str]) -> dict[str, xr.Dataset]`; add `split_by_collection`; one collision check)
- Modify: `src/geosave/geodata/io/raster/writers.py` (keep `write_raster_files(files: Mapping[UPath, xr.Dataset], writer, ...)` as the only writer; delete `write_group_files`, `resolve_group_paths`, `resolve_paths`)
- Modify: `src/geosave/geodata/io/raster/zarr.py`, `netcdf.py` (write a DataTree as one store with groups, applying `fill_encoding` and the recorded variable order per group path)
- Modify: `src/geosave/geodata/core/raster/{dataset,dataarray,datatree}.py` (`to_cog` lays out with `expand_paths`; `to_zarr`, `to_netcdf` drop `template`, `path` is the store)
- Modify: `src/geosave/geodata/io/openers.py` (`open_tree` reads one store's groups; docstring example)
- Modify docs: `docs/concepts/storage.md:85`, `docs/concepts/lazy-data.md:85`, `docs/basics/io.md`
- Test: `tests/geodata/io/raster/test_paths.py`, `test_writers.py`, `test_zarr.py`, `test_netcdf.py`, `tests/geodata/io/test_openers.py`, `tests/geodata/core/raster/test_datatree.py`

**Interfaces:**

- Produces: `expand_paths(raster: xr.Dataset | xr.DataTree, template: str | Mapping[str, str]) -> dict[str, xr.Dataset]`
- Produces: `split_by_collection(raster) -> list[tuple[str | None, xr.Dataset | xr.DataTree]]`, `split_by_res`, `split_by_time`, `split_by_band`
- Produces: `tree.gs.to_zarr(path, *, compute=True, overwrite=False, ...)`, `tree.gs.to_netcdf(path, ...)` -> `UPath`

- [ ] **Step 1: Failing tests**

```python
def test_each_field_splits_its_axis(tree)                  # "{collection}/{res}/{time:%Y%m%d}/{band}.tif" -> one Dataset per collection, grid, date, band
def test_a_collection_on_several_grids_needs_res(tree)     # "{collection}/{time:%Y%m%d}.tif" -> ValueError naming {res}
def test_several_collections_need_collection(tree)         # "{time:%Y%m%d}.tif" -> ValueError naming a field
def test_a_template_per_collection(tree)                   # {"s2": "...{time}.tif", "dem": "dem.tif"}
def test_a_tree_round_trips_through_one_zarr_store(tree, tmp_path)    # to_zarr(path) -> open_tree(path) keeps /s2/r10m ... and time
def test_a_tree_round_trips_through_one_netcdf_file(tree, tmp_path)
```

- [ ] **Step 2:** run them → FAIL
- [ ] **Step 3: Implement** as the spec's Writing section.
- [ ] **Step 4:** `uv run pytest tests/geodata/io tests/geodata/core/raster -q` → PASS

---

### Task 5: `to_catalog` writes COGs and one Item per collection and date

**Files:**

- Modify: `src/geosave/geodata/io/raster/publish.py` (`publish_rasters(raster, path, *, template, collections, id_format, datetime, overwrite, ...)` loops `split_by_collection`, lays out, writes COGs, builds rows)
- Modify: `to_catalog` on Dataset, DataArray, DataTree
- Test: `tests/geodata/io/raster/test_publish.py`, `tests/geodata/core/raster/test_base.py` (`.nc`/`.zarr` catalog tests become refusals), `tests/geodata/io/raster/test_netcdf.py`

**Interfaces:**

- Consumes: Task 4 `expand_paths`, `split_by_collection`, `write_raster_files`
- Produces: `tree.gs.to_catalog(path, *, template, ...) -> GeoDataFrame`; refuses a template not ending in `.tif`/`.tiff`

- [ ] **Step 1: Failing tests**

```python
def test_a_mixed_grid_tree_publishes_one_item_per_date(mixed_tree, tmp_path)  # assets red (10 m) and swir (20 m) in one Item, each its own proj:*
def test_publication_round_trips_through_to_datatree(mixed_tree, tmp_path)    # rows.gs.to_datatree() equals mixed_tree
def test_a_store_template_is_refused_for_catalogs(tree, tmp_path)             # ".zarr" -> ValueError pointing to to_zarr
```

- [ ] **Step 2:** run them → FAIL
- [ ] **Step 3: Implement**; each Item's grouping comes from the split, not from reading files back.
- [ ] **Step 4:** `uv run pytest tests/geodata tests/flows/test_ingest.py -q` → PASS

---

### Task 6: Docs and full checks

- [ ] **Step 1:** Update `docs/basics/io.md`, `docs/guides/catalogs-and-cuts.md`, `docs/concepts/catalog-windows.md`, `docs/reference/*` for `to_datatree`, `to_dataset` and nested trees; `uv run python scripts/api_reference.py`.
- [ ] **Step 2:** `uv run python scripts/check_docstrings.py` → clean.
- [ ] **Step 3:** `uv run pytest`; `uv run ruff check .`; `uv run basedpyright src/geosave/geodata` → report results, separating failures this plan did not touch (`ModelSpec.chips`).

---

### Task 7: `StacSource.load_datatree`

**Files:** `src/geosave/geodata/stac/source.py`; test `tests/geodata/stac/test_source.py`

**Interfaces:** `StacSource.load_datatree(anchor: GeoAnchor) -> DataTree` (`/{collection}/{res}`)

- [ ] **Step 1: Failing tests:** a two-resolution local collection loads as `r10m` on `anchor.geobox` and `r20m` on `anchor.geobox.zoom_out(2)`; band metadata and `StacMetadata` as `load_dataset` stamps them.
- [ ] **Step 2:** implement: group the bands by their stated native pixel size (`proj:*` per asset), `odc.stac.load(items, bands=group, geobox=anchor.geobox.zoom_to(resolution=size))` per group, then the same stamping as `load_dataset`, built into a tree with `create_datatree`.
- [ ] **Step 3:** `uv run pytest tests/geodata/stac -q` → PASS
