# Samples by Region Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-row-per-sample table, and its frame, chip and merge machinery, with the item table plus one `region` geometry column, grouped by pandas.

**Architecture:** `GeoCatalog.sample(regions)` adds a normalized WGS84 `region` column. An Item repeats once per region it shares area with. `GeoSamples` (a `GeoCatalog` subclass, dispatched when `region` is present) counts, indexes and reads samples, and saves them as stac-geoparquet with `region` as a second GeoParquet geometry column. Old sampling code is deleted first, not ported.

**Tech Stack:** GeoPandas 1.1, shapely 2.1, pandas, PySTAC, stac-geoparquet (`stac_geoparquet.arrow`), PyArrow, xarray/Dask, odc-geo, pytest, Ruff, basedpyright.

**Spec:** [Samples are catalog rows grouped by region](../specs/2026-10-31-samples-by-region.md)

## Global Constraints

- Build from scratch. Copy no code from stash `40624b45`; read its rules in the spec's "Retired mechanisms" only.
- Rows stay STAC Items; Item ids are never rewritten.
- `region` is a real geometry column, never JSON or a string encoding.
- Grouping is spatial only; the group key is the `region` geometry itself (no id column).
- An Item joins a region only when their intersection area is above 0.
- Item tables and `region` are WGS84 (`EPSG:4326`).
- `to_xarray` stays lazy; no pixels computed by `sample()` or `to_xarray()`.
- Use existing dependencies only. No compatibility aliases. Ignore notebooks.
- Preserve unrelated working-tree changes. Commit only when the user asks.
- A function name is a verb for what the process does; a variable names the value it holds. No vague words ("named", "stated", "info", "data" as a suffix); clear at a glance.
- No multi-step comprehensions or clever one-liners; write loops out. Split a function when that makes it easier to read; a long function needs a reason.
- When code is tangled, fix the cause; no workaround layered on top.
- After each task: `uv run ruff check .`, `uv run basedpyright src/geosave/geodata/core` and the task's tests.

## Review Focus

- A `regions` frame without a CRS: refuse with ValueError instead of assuming WGS84 (Task 3).
- A region touching an Item only along an edge or a corner: no row for that pair (Task 3).
- Region polygons drawn with opposite ring order or another start vertex but the same shape: one sample (Task 3).
- A sample table filtered with a boolean mask, then concatenated with another: still `GeoSamples` with Collection metadata (Task 3).
- `to_xarray` on rows of two regions, or on zero rows: clear ValueError, no files opened (Task 4).

---

### Task 1: Restore the import

Modules import `geosave.geodata.core.constructors`; the folder is `core/ctor/`, which stays (user's choice).

**Files:**

- Modify: every `src`, `tests` and `docs` reference to `core.constructors` / `..constructors` -> `core.ctor` / `..ctor`
- Modify: `CLAUDE.md` structure line `constructors` -> `ctor`

**Interfaces:**

- Produces: `geosave.geodata.core.ctor.{catalog,dataarray,dataset,datatree}` unchanged.

- [ ] **Step 1:** Replace the import paths; `grep -rn "constructors" src tests docs CLAUDE.md` shows no import path left.
- [ ] **Step 2:** Run `uv run python -c "import geosave"`. Expected: no output, exit 0.
- [ ] **Step 3:** Run `uv run pytest tests/geodata/core -q -x`. Record the pass/fail count as the baseline; failures here are not in scope.

### Task 2: Delete the old sample machinery

**Files:**

- Delete: `src/geosave/geodata/sampling/` (whole package), `src/geosave/geodata/core/vector/schema.py`, `src/geosave/geodata/io/vector/sample_geoparquet.py`, `src/geosave/model/spec/windows.py`
- Delete tests: `tests/geodata/sampling/`, `tests/geodata/core/vector/test_samples.py`, `tests/geodata/core/vector/test_schema.py`, `tests/model/spec/test_windows.py`
- Delete docs: `docs/contributing/specs/2026-10-30-{frame-input-sequences,frame-records-and-derived-padding,sample-read-contract,sampling-windows}.md` and their rows in `docs/contributing/specs/index.md`; `docs/contributing/plans/2026-10-30-{sample-read-contract,sample-row-ownership,sampling-windows,slot-parser-and-merge}.md`
- Empty: `src/geosave/geodata/core/vector/samples.py` (delete the file; Task 3 creates it new)
- Modify: `core/__init__.py`, `core/register.py`, `geodata/__init__.py`, `geosave/__init__.py` (drop `GeoSample`, `GeoSamples` exports)
- Modify: `core/vector/catalog.py` (drop `sample()`), `io/vector/geoparquet.py` (drop the `GeoSamples` branch), `io/vector/__init__.py` docstring, `viz/explorer/vector.py:61` (`isinstance(frame.gs, GeoCatalog)`)
- Modify: `model/spec/model.py` (drop `frames`, `chips` fields and `FramesSpec`/`ChipsSpec` imports; drop `ModelSpec.sample` until Task 3), `model/spec/__init__.py`
- Modify: the `frames:`/`chips:` blocks out of every `src/templates/tasks/*/*/configs/*/model_spec.yaml` and `workspace/configs/dynamic-world/model_spec.yaml`
- Modify: `tests/model/spec/test_model.py` (delete tests of `frames`, `chips` and `ModelSpec.sample`; Task 3 adds the new one)
- Modify: `tests/geodata/core/vector/conftest.py` (move `grid`, `wide_grid`, `sample`, `radar`, `items` fixtures from the deleted sampling conftest into it, written fresh; keep `composites` only if a remaining test uses it)

**Interfaces:**

- Produces: `RowAccessor` returns `GeoCatalogRow` for item-table rows; `FrameAccessor` chooses `GeoCatalog` or `GeoFrame`.

- [ ] **Step 1:** Delete and edit as listed.
- [ ] **Step 2:** Run `grep -rn "sampling\|GeoSample\|schema import\|FramesSpec\|ChipsSpec\|sample_geoparquet\|frames:\|chips:" src tests workspace --include=*.py --include=*.yaml`. Expected: no hit outside the spec and unrelated words.
- [ ] **Step 3:** Run `uv run python -c "import geosave"` and `uv run pytest tests -q -x -m "not integration and not slow"`. Expected: the Task 1 baseline minus the deleted tests; any new failure points at a missed reference.

### Task 3: `GeoCatalog.sample` and the `GeoSamples` table

**Files:**

- Create: `src/geosave/geodata/core/vector/samples.py`
- Modify: `src/geosave/geodata/core/vector/catalog.py`, `core/register.py`, `core/__init__.py`, `geodata/__init__.py`, `geosave/__init__.py`, `model/spec/model.py`
- Test: `tests/geodata/core/vector/test_samples.py` (new), `tests/geodata/core/test_register.py`, `tests/model/spec/test_model.py`

**Interfaces:**

- Produces:
  - `GeoCatalog.sample(self, regions: gpd.GeoDataFrame | None = None) -> GeoDataFrame[GeoSamples]`
  - `class GeoSamples(GeoCatalog)` with `COLUMNS = (*GeoCatalog.COLUMNS, "region")`, `__len__(self) -> int`, `__getitem__(self, position: int) -> GeoDataFrame[GeoSamples]`
  - `ModelSpec.sample(self, catalog: GeoDataFrame[GeoCatalog]) -> GeoDataFrame[GeoSamples]` returning `catalog.gs.sample()`

Test data: build Items in memory with `pystac.Item` and `create_catalog`; no files needed. Use the spec's layout in WGS84 degrees: `s2_A_jan`, `s2_A_feb` on `box(0,0,1,1)`; `s2_B_jan` on `box(1,0,2,1)`; `s1_swath` on `box(0.5,0.2,1.5,0.8)`; `label` on `box(0.1,0.1,0.4,0.4)`; regions `box(0,0,1,1)`, `box(1,0,2,1)`.

- [ ] **Step 1: Write the failing tests** in `test_samples.py`:
  - `test_sample_groups_equal_footprints`: `items.gs.sample()`; `len(samples.gs) == 4`; the rows of `s2_A_jan` and `s2_A_feb` share one `region`; `len(samples) == len(items)`.
  - `test_sample_repeats_item_per_region`: `items.gs.sample(regions)`; `samples.groupby("region", sort=False).size().tolist() == [4, 2]` (A: `s2_A_jan`, `s2_A_feb`, `s1_swath`, `label`; B: `s2_B_jan`, `s1_swath`); `s1_swath` appears twice; ids unchanged.
  - `test_sample_ignores_shared_edge`: `s2_A_jan` (box(0,0,1,1)) is not in region `box(1,0,2,1)`.
  - `test_sample_normalizes_regions`: regions given as `box(0,0,1,1)` and `box(0,0,1,1).reverse()` in two separate calls give equal `region` values.
  - `test_sample_reprojects_regions`: regions in `EPSG:3857` give the same groups as WGS84; `samples["region"].crs == "EPSG:4326"`.
  - `test_sample_refuses_regions_without_crs`: `pytest.raises(ValueError)`.
  - `test_sample_keeps_collections`: `samples.attrs["stac:collections"] == items.attrs["stac:collections"]`; and after `samples[samples["collection"] == "s2"]`, `isinstance(subset.gs, GeoSamples)`.
  - `test_samples_getitem`: `samples.gs[0]["id"].tolist()` equals the first group's ids; `samples.gs[len(samples.gs)]` raises IndexError.
  - `test_sample_empty_catalog`: `items.iloc[:0].gs.sample()` has `len(...gs) == 0` and a `region` column.
  - In `test_register.py`: a frame with the catalog columns plus a `region` GeoSeries reads through `GeoSamples`; a sample-table row reads through `GeoCatalogRow`.
- [ ] **Step 2:** Run `uv run pytest tests/geodata/core/vector/test_samples.py -q`. Expected: FAIL, `GeoSamples`/`sample` undefined.
- [ ] **Step 3: Implement** `GeoCatalog.sample` in `catalog.py`. Without regions, `region = geometry.normalize()`. With regions: refuse `regions.crs is None`; `regions.to_crs(items.crs)`; normalize the region geometries once; `self._data.sjoin(regions[[regions.geometry.name]], predicate="intersects")`; take each pair's region by `index_right`; drop pairs whose `intersection(...).area` is 0; drop `index_right`; `reset_index(drop=True)`; set `region` as a GeoSeries with the table's CRS; keep `attrs`. Implement `GeoSamples` in `samples.py`; `__getitem__`/`__len__` use `groupby("region", sort=False).ngroup()`; out-of-range position raises IndexError. `register.py` checks `GeoSamples.COLUMNS` before `GeoCatalog.COLUMNS`; the row accessor keeps `GeoCatalogRow` only. Restore exports and `ModelSpec.sample`.
- [ ] **Step 4:** Run `uv run pytest tests/geodata/core tests/model/spec -q`. Expected: PASS.

### Task 4: `GeoSamples.to_xarray`

**Files:**

- Modify: `src/geosave/geodata/core/vector/samples.py`
- Test: `tests/geodata/core/vector/test_samples.py`

**Interfaces:**

- Consumes: `io.raster.stac.open_raster(items, *, chunks, collection_fields) -> xr.Dataset`, `transform.extent.crop(data, region, *, mask)`, `core.constructors.datatree.create_datatree(rasters)`.
- Produces: `GeoSamples.to_xarray(self, *, chunks: ChunkSpec = {}) -> DataTree`

Test data: the `items` fixture from Task 2's conftest (s2 dated series, dem, label on one 10 m grid; s1 at 20 m), sampled with `items.gs.sample()` and with one region covering half the grid.

- [ ] **Step 1: Write the failing tests:**
  - `test_to_xarray_groups_collections`: `tree.gs.groups` lists the sample's collections in first-appearance order.
  - `test_to_xarray_stacks_time`: the `s2` group has `time` of the sample's s2 Item count; `label` and `dem` have no `time` dimension.
  - `test_to_xarray_cuts_to_region`: with a half-grid region, each group's `gs.bounds` lies within the region's bounds in that group's CRS, within one pixel.
  - `test_to_xarray_is_lazy`: every data variable is Dask-backed (`variable.chunks is not None`).
  - `test_to_xarray_refuses_several_regions`: `samples.gs.to_xarray()` on the whole table raises ValueError naming `gs[position]`; on zero rows raises ValueError.
  - `test_to_xarray_carries_collection_attrs`: a group's attrs include the Collection's ACDD title when the table holds its definition.
- [ ] **Step 2:** Run them. Expected: FAIL, `to_xarray` undefined.
- [ ] **Step 3: Implement:** require exactly one distinct `region`; for each collection in order of first appearance, `open_raster` its rows' Items (`self._data[...].gs.to_items()`) with that collection's stored fields, then `crop(raster, Geometry(region, "EPSG:4326"), mask=False)`; build with `create_datatree`. If `open_raster` returns a length-1 `time` for timeless Items, drop it there and say why in one comment.
- [ ] **Step 4:** Run `uv run pytest tests/geodata/core/vector/test_samples.py -q`. Expected: PASS.

### Task 5: Save and open sample tables

**Files:**

- Modify: `src/geosave/geodata/io/vector/stac_geoparquet.py`, `src/geosave/geodata/core/vector/samples.py`
- Test: `tests/geodata/io/vector/test_stac_geoparquet.py`

**Interfaces:**

- Consumes: `stac_geoparquet.arrow.parse_stac_items_to_arrow`, `stac_geoparquet.arrow._to_parquet.create_parquet_metadata(schema, *, schema_version, collections)`.
- Produces: `GeoSamples.to_geoparquet(self, path, *, collections=None, overwrite=False, storage_options=None) -> UPath`, inherited from `GeoCatalog` unchanged in signature.

- [ ] **Step 1: Write the failing tests:**
  - `test_samples_round_trip`: `open_vector(samples.gs.to_geoparquet(path))` reads through `GeoSamples`; `region` is a GeoSeries; `geometry` is active; groups, ids, asset hrefs and `attrs["stac:collections"]` equal the original.
  - `test_samples_repeat_ids_across_regions`: a table where `s1_swath` sits in two regions writes and reads back both rows.
  - `test_samples_refuse_repeated_row`: two rows with the same collection, id and region raise ValueError.
  - `test_catalog_still_refuses_repeated_ids`: an item table with a repeated `(collection, id)` still raises.
- [ ] **Step 2:** Run them. Expected: FAIL at the repeated-id check.
- [ ] **Step 3: Implement** in `stac_geoparquet.write`: for a `GeoSamples` table, check uniqueness on `(collection, id, region WKB)` instead of `validate_ids`; build the Arrow table as now; append `region` as a WKB binary column; build the schema metadata with `create_parquet_metadata`, add a `"region"` entry (`"encoding": "WKB"`, same `crs` as `geometry`) to its `geo` columns, and write with `pq.ParquetWriter` through the existing `storage.write`, so the file is written once.
- [ ] **Step 4:** Run `uv run pytest tests/geodata/io/vector tests/geodata/core -q`. Expected: PASS.

### Task 5b: Vector accessors from the stac-geoparquet spec

Tasks 1-5 are done under the old names. This task redesigns the vector
accessors as the spec's "Vector accessors" section states.

**Files:**

- Modify: `core/vector/vector.py` (`GeoVector` base: no writers)
- Create: `core/vector/frame.py` (`GeoFrame`: the GeoJSON, GeoPackage, GeoParquet writers)
- Create: `core/vector/catalog.py` (`GeoCatalog` with `StacMetadata` folded in, `GeoCatalogRow`); delete `catalog.py`, `stac.py`
- Create: `core/vector/samples.py` (`GeoSamples`); delete `samples.py`
- Modify: `core/register.py`, exports, `io/vector/geoparquet.py` (drop the `type`/`stac_version` injection), `stac/translate/table.py` (empty-table columns)
- Rename across `src`, `tests`, `docs`, `README.md`, `CLAUDE.md`: `GeoCatalog` -> `GeoCatalog`, `GeoItem` -> `GeoCatalogRow`, `GeoSamples` -> `GeoSamples`, public `GeoVector` -> `GeoFrame`
- Test: `tests/geodata/core/vector/test_catalog.py`, `test_samples.py`, `test_frame.py` (moved from `test_catalog.py`, `test_samples.py`, `test_vector.py`, `test_stac.py`)

- [ ] **Step 1:** Write tests: a frame with `id`, `links`, `assets`, `stac_extensions` and no `type`/`stac_version` reads through `GeoVector` and `to_items()` returns Items; `GeoCatalog` has no `to_geojson`/`to_geopackage`; `to_geoparquet` has no `**options`; a plain frame reads through `GeoFrame`.
- [ ] **Step 2:** Run them; expected FAIL.
- [ ] **Step 3:** Restructure and rename as listed.
- [ ] **Step 4:** Run `uv run pytest tests/geodata -q -m "not integration and not slow"`, ruff, basedpyright on `core`, `io/vector`, `stac/translate`. Expected: PASS.

### Task 6: Smoke test and docs

**Files:**

- Modify: `docs/reference/catalog.md`, `docs/guides/catalogs-and-cuts.md` (replace sample/frame/chip sections with the new caller code from the spec), `docs/contributing/open-decisions.md` (note frame, chip and merge as open, pointing to the spec)
- Modify: `docs/contributing/specs/2026-10-31-samples-by-region.md` (status: implemented)

- [ ] **Step 1:** Write a throwaway smoke script in the scratchpad: COGs for two overlapping scenes and one label -> `create_catalog` -> `sample(regions)` -> `to_geoparquet` -> `open_vector` -> `gs[0].gs.to_xarray().compute()`. Expected: values equal a direct `open_raster(...).gs.crop(region, mask=False)` read.
- [ ] **Step 2:** Edit the docs.
- [ ] **Step 3:** Run `uv run pytest -q -m "not integration and not slow"`, `uv run ruff check .`, `uv run pre-commit run --all-files`. Report failures with their output.
