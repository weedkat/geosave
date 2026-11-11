# Core API Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make native-object factories and the vector accessor hierarchy consistent without changing sample or raster semantics.

**Architecture:** Native values retain .gs accessors. GeoCatalog and GeoSamples share an internal STAC-metadata parent below GeoVector; constructors, registration, translation and persistence have explicit owners.

**Tech Stack:** xarray, GeoPandas/pandas, PySTAC, stac-geoparquet 0.8.2, Arrow, rustac, Dask, pytest, Ruff, Hatchling.

**Spec:** [Approved design](../specs/2026-10-31-core-api-ownership.md). Implementation authorized on 2026-10-31.

## Global Constraints

- Values remain native xarray and GeoPandas objects.
- Preserve lazy raster behavior, metadata, grid semantics, sample recipes, storage options and current formats.
- Use existing dependencies; do not change version floors.
- Do not change ModelSpec or sampling algorithms. Ignore notebooks in this work.
- Preserve unrelated working-tree changes. No compatibility aliases are required.
- Keep the current first-definition-wins Collection concat policy.
- Execute in an isolated checkout when implementation begins; do not commit unrelated edits.

## Review Focus

- Upstream files omit optional JSON-only columns: recognize from format evidence without misclassifying ordinary id/assets business columns (Task 1).
- Filtered/concatenated Arrow chunks and heterogeneous assets: preserve Item identities, timestamps and assets (Task 1).
- Empty/renamed-geometry/reprojected tables: preserve accessor selection and Collection metadata (Task 2).
- GCP references, conflicting coordinates and lazy pixels: preserve grid metadata and refuse misalignment without computing (Task 3).
- Root registration/import order: accessors work on native objects without eager torch/viz imports (Task 4).

## Task 1: Establish the upstream conversion and persistence boundary

**Files:**
- Modify: src/geosave/geodata/stac/translate/table.py
- Modify: src/geosave/geodata/io/vector/geoparquet.py
- Modify: src/geosave/geodata/io/vector/stac_geoparquet.py
- Tests: tests/geodata/stac/translate/test_table.py
- Tests: tests/geodata/io/vector/test_stac_geoparquet.py
- Tests: tests/geodata/io/vector/test_geoparquet.py

**Interfaces:** Preserve create_catalog(items, *, collections=()) -> GeoDataFrame[GeoCatalog], to_items(rows) -> list[pystac.Item], and all reader/writer signatures. Readers restore canonical in-memory Item columns and Collection metadata when upstream file metadata identifies a STAC table.

- [x] Add an interoperability test that builds Arrow through parse_stac_items_to_arrow, writes with upstream to_parquet(collections=...), then asserts gs.open_vector returns a native GeoDataFrame with GeoCatalog and the same Collection id/definition. Use one actual asset reference; opening pixels is unnecessary.
- [x] Repeat with omitted type/stac_version and pin the reader's normalization to type="Feature" and stac_version="1.1.0" only when absent on an upstream STAC-metadata-bearing file. Preserve existing values. Assert ordinary id/assets tables without STAC evidence remain GeoVector. Keep existing reader sample recognition.
- [x] Run those tests before implementation and capture actual failures. Distinguish protocol compatibility from runtime schema validation.
- [x] Smoke stac_table_to_items using filtered tables, concatenated sliced chunks, empty tables, differing asset keys and scalar/null properties. Compare Item id, collection, geometry, datetime, asset keys/hrefs and extra properties with current conversion.
- [x] If all cases preserve semantics, replace the local rustac conversion with the upstream public converter, removing only bbox/chunk code made unnecessary. If not, retain current conversion and document the precise incompatibility in the design's evidence section.
- [x] Normalize omitted JSON-only fields at the format reader boundary, using file STAC metadata as evidence before schema-based href restoration. Reuse the existing upstream/table translator rather than introducing another serializer.
- [x] Run `uv run pytest tests/geodata/stac/translate/test_table.py tests/geodata/io/vector/test_stac_geoparquet.py tests/geodata/io/vector/test_geoparquet.py -q`. All local checks must pass; report schema-fetch failures separately.
- [x] Review Task 1 files. Commit deferred because Git metadata is read-only.

## Task 2: Make vector accessor ownership explicit

**Files:**
- Rename: src/geosave/geodata/core/vector/table.py -> stac.py
- Modify: src/geosave/geodata/core/vector/catalog.py, samples.py, __init__.py
- Merge/remove: src/geosave/geodata/core/vector/catalog.py, sample.py
- Create: src/geosave/geodata/core/vector/schema.py
- Modify: src/geosave/geodata/core/__init__.py
- Modify: src/geosave/geodata/sampling/sample.py, frame.py, chip.py
- Tests: tests/geodata/core/vector/test_stac.py, test_catalog.py, test_samples.py
- Move row tests from test_catalog.py into test_catalog.py and test_samples.py into test_samples.py; rename test_table.py to test_stac.py.
- Create: tests/geodata/core/vector/test_schema.py only for moved schema decoding behavior.

**Interfaces:** GeoVector -> internal StacMetadata -> GeoCatalog/GeoSamples. GeoItem and GeoSample retain current signatures and native read returns. schema.py owns SAMPLE_FIELDS, SAMPLE_COLUMNS, _TimeBin and _FrameRecipe; readers and planners consume those definitions. Move _decode_frame with its recipe types, retaining its exact signature and validation.

- [x] Pin metadata-copy independence, concat order/first-definition precedence, empty outputs, renamed geometry, reprojected samples and lossy-export rejection using existing tests; add assertions only for uncovered behavior.
- [x] Rename CollectionTable to StacMetadata and update imports, without changing inheritance behavior or adding a second shared base. Set its module __all__ to an empty list.
- [x] Co-locate GeoItem in catalog.py and GeoSample in samples.py. Move the persisted sample contract/decoder to schema.py; replace imports from reader-private types with schema imports. Replace GeoSamples.FIELDS/COLUMNS consumers with the shared constants and remove the duplicate class constants.
- [x] Trace imports through sampling, model types, I/O and exports. Preserve Window/WindowFunction import locations and ModelSpec behavior. No compatibility re-export modules.
- [x] Move tests to match their owning source modules, retaining markers and fixtures. Add no duplicate copies of moved tests.
- [x] Run `uv run pytest tests/geodata/core/vector tests/geodata/sampling tests/geodata/io/vector tests/model/spec -q`. Compare any failures with the unchanged baseline before modifying behavior.
- [x] Review Task 2 files. Commit deferred because Git metadata is read-only.

## Task 3: Split and rename native-object factories

**Files:**
- Replace: src/geosave/geodata/core/raster/create.py with core/constructors/{__init__,dataarray,dataset,datatree,catalog}.py
- Create: src/geosave/geodata/core/raster/grid.py
- Modify: src/geosave/geodata/core/raster/dataset.py, datatree.py
- Modify: src/geosave/geodata/stac/create.py, __init__.py
- Modify: src/geosave/geodata/{__init__,core/__init__}.py and src/geosave/__init__.py
- Modify: all non-notebook factory callers in src/, tests/, scripts/, examples/ and docs/ identified with rg.
- Tests: tests/geodata/core/constructors/test_dataarray.py, test_dataset.py, test_datatree.py, test_catalog.py
- Tests: tests/geodata/core/raster/test_grid.py

**Interfaces:** create_dataarray has array's current signature/return; create_dataset has raster's; create_datatree has tree's; create_catalog has stac.create_catalog's. No create_vector. write_cf_axes retains its signature; grid helper behavior is unchanged.

- [x] Pin native result types and lazy pixel identity through the new factory names, plus current empty/mismatched-grid/GCP behavior. Run to observe missing imports before moving code.
- [x] Move grid helpers to raster/grid.py; move constructor bodies to their type-named modules. Keep create_dataarray delegation to create_dataset and band-attribute inheritance beside create_dataset.
- [x] Move create_catalog to constructors/catalog.py, retaining duplicate-id validation and translator delegation. Keep create_items/create_collection in stac/create.py; remove stac.create_catalog export.
- [x] Export all four factories at gs and geodata; remove array/raster/tree public functions. Retain native gs.DataArray/Dataset/DataTree/GeoDataFrame runtime aliases and typing shims.
- [x] Remove GeoVector.from_geometry. Replace its callers with native GeoDataFrame construction; use utils.geometry.to_shapely and explicit CRS resolution where parsing/ODC geometry requires it. Preserve explicit-CRS mismatch validation in callers that still require that domain constraint; do not add a replacement factory.
- [x] Update all non-notebook imports/calls in one coherent change, including flows and generated workspace recipes. Use dependency-module imports internally, not the package root. Inspect ambiguous raster/array/tree text matches before editing.
- [x] Redistribute constructor tests from test_create.py and tree-construction cases from test_datatree.py to mirror new ownership; keep accessor behavior tests in raster/.
- [x] Run `uv run pytest tests/geodata/core tests/geodata/transform tests/geodata/io tests/geodata/sampling tests/geodata/stac tests/templates tests/flows -q`; record environmental/integration failures honestly. Verify generated segmentation preparation code uses the public factory.
- [x] Review Task 3 files. Commit deferred because Git metadata is read-only.

## Task 4: Centralize registration and complete the public migration

**Files:**
- Create: src/geosave/geodata/core/register.py
- Remove: src/geosave/geodata/core/vector/accessors.py
- Modify: src/geosave/geodata/core/raster/{dataarray,dataset,datatree}.py
- Modify: src/geosave/geodata/{core/__init__,__init__}.py, src/geosave/__init__.py
- Move: tests/geodata/core/vector/test_accessors.py -> tests/geodata/core/test_register.py
- Modify: tests/test_init.py
- Modify: docs/reference/public-api.md, docs/reference/generated/data.md and affected text documentation
- Modify: source docstrings and generated-workspace examples affected by removed names

**Interfaces:** One root import registers native xarray and pandas accessors; pandas dispatch stays schema-aware, sample-first. Root exports include GeoItem and GeoSample as well as the existing public accessor classes. Internal StacMetadata is not exported.

- [x] In fresh-process tests, construct native DataArray/Dataset/DataTree/ordinary GeoDataFrame/Item table/sample table and both row types after importing geosave. Assert expected accessor classes and native containers, including empty tables. Assert importing geosave does not import torch, panel or holoviews.
- [x] Move the existing pandas dispatch classes to register.py; move xarray registration decorators there as explicit registrations. Import registration once after core class imports; keep constructors independent of register.py.
- [x] Add GeoItem/GeoSample to root public imports, __all__ and typing exposure. Move registration typing tests with their owner and retain static reader typing checks.
- [x] Update text docs and docstrings to the approved public API, including the distinction between table operations and row to_xarray. List obsolete notebook calls as remaining work without editing notebooks.
- [x] Run `uv run python scripts/api_reference.py` and `uv run python scripts/api_reference.py --check`. Inspect generated changes before retaining them.
- [x] Run `uv run ruff check` and `uv run basedpyright` with the explicit list of touched Python source/test files, then `uv run pytest tests/geodata/core tests/geodata/stac tests/geodata/io/vector tests/geodata/sampling tests/model/spec tests/templates -q`.
- [x] Run a fresh-process end-to-end smoke: lazy factory -> raster write -> STAC Items -> create_catalog -> sample/frame/chip -> row read -> sample GeoParquet round trip. Assert native return types, preserved grids/Collections and lazy pixels. Keep all outputs in a temporary directory.
- [x] Search non-notebook callers for removed imports/names and inspect each match. Run `git diff --check` and review only this migration's changes.
- [x] Record the final affected-suite result and remaining risks. Commit deferred because Git metadata is read-only.

## Baseline evidence

On 2026-10-31, focused tests produced 251 passed and one external-schema DNS
failure in test_a_catalog_writes_stac_geoparquet_that_validates. Existing shared
working-tree modifications predate this plan and must be preserved.

Direct upstream Arrow conversion and Parquet write/read passed with a real
asset reference. The current reader recognized upstream output and restored
Collection metadata. Removing type/stac_version reproduced GeoVector fallback
and lost Collection restoration; Task 1 pins and resolves that behavior.

## Implementation decisions and verification

The user approved implementation after reviewing the design and plan.

- Worktree creation was blocked by read-only Git metadata. Work proceeded in the
  existing checkout with a complete snapshot of the original files in /tmp.
  No commits were created and unrelated changes were preserved.
- The existing rustac converter was retained: direct upstream conversion still
  requires the bbox representation our native tables omit. No duplicate converter
  or dependency was added.
- CollectionTable became internal StacMetadata. Row readers moved beside their
  table accessors; shared sample fields and recipe decoding moved to schema.py.
- Co-locating the catalog tests exposed a grid fixture scope conflict. Catalog
  persistence cases now build their own rows; Item readers retain the shared
  sampling fixtures. All moved parameterization/markers were preserved.
- A direct grid helper test now uses the required y/x coordinate convention and
  checks existing CF behavior, without introducing new arbitrary-attribute
  preservation semantics. The STAC namespace test was updated for create_catalog
  moving to the root API.
- Root/registration tests: 22 passed. Final constructor/grid/root checks: 95 passed.
- Independent review: no blocking runtime findings; 127 independent checks passed.
  Two NumPy output representations accidentally renamed in docstrings were fixed.
- End-to-end smoke passed: native/lazy factories, COG writing, catalog creation,
  sample/frame/chip planning, lazy row read and sample GeoParquet round trip.
- Source type checks: zero errors. A baseline comparison found 147 existing test
  typing diagnostics; the migrated files have 146, with no new diagnostics after
  accounting for renamed APIs. Those unrelated test-typing issues are unchanged.
- Ruff passed for touched Python files. Zensical built successfully; generated
  API freshness and rendered coverage passed (1188/1188 contracts).
- The broad test run was rerun outside the sandbox after a Zarr write hung in
  both the original and migrated code inside the sandbox. The isolated Zarr
  round-trip passed outside the sandbox.
- Three sampling failures were reproduced using the original snapshot: a missing
  public-scenario probe, an obsolete thumbnail grid assertion, and an obsolete
  missing-grid warning assertion. They remain outside this migration.
- Notebooks were not changed. They can retain old factory calls until a separate
  notebook update. Factory/import renames and removal of from_geometry are
  intentional public breaks; no persistence format changed.

Final affected-suite verification: **1608 passed, 26 deselected, 3 xfailed** in
183.22 seconds. The deselections comprise the normal 23 slow/integration cases
and the three original sampling failures named above. The final command covered
core, transforms, I/O, sampling, STAC, generated training recipes, flows and
ModelSpec tests. Root/fresh-process registration checks and the end-to-end smoke
also passed separately. No migration regressions remain in these checks.


