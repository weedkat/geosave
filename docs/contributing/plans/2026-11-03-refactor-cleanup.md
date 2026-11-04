# I/O and STAC cleanup implementation plan

> Execute task-by-task with the executing-plans skill. This plan follows the
> approved I/O/STAC design; it does not introduce another architecture.

**Goal:** Remove refactor residue and make maintained code, tests, documentation,
and all four example notebooks agree with the current public API.

**Architecture:** GeoAnchor describes geometry/time; native GeoBox describes a
pixel grid. Clients discover Items, io loaders read them, io.formats owns file
formats, and stac owns metadata. Native xarray and GeoPandas remain public values.

**Spec:** ../specs/2026-11-03-io-stac-ownership.md

**Stack:** Python, xarray, GeoPandas, ODC, PySTAC, pytest, nbformat/nbclient, Zensical.

## Constraints

- Preserve unrelated shared-checkout changes; no blanket git clean or reset.
- Remove obsolete code rather than adding compatibility aliases.
- A missing caller alone does not prove a public API or plugin entry point dead.
- Preserve lazy reads, exact requested grids, metadata, and persistence round trips.
- Historical plans/specs may name old APIs; mark superseded documents where needed
  rather than rewriting history as current instructions.
- Existing model/template failures are migration issues, not automatically stale
  tests. Do not delete or xfail them to obtain a green suite.
- Notebooks are explicitly in scope. Preserve their teaching order and exercises.

## Audit evidence and work already completed

- Removed cache-only legacy directories under src/geosave/geodata/io/{raster,vector},
  src/geosave/geodata/stac/translate, and their three mirrored test directories.
  Each was inspected first; no source files were removed by this operation.
- Import/signature smoke confirmed StacClient.source and GeoCatalog.to_xarray are
  absent; GeoAnchor.from_geometry accepts no resolution. GeoCatalog.concat still
  exists and must not be removed from examples merely because it looks old.
- examples/02_sources.ipynb uses source/set_config/set_query/load, anchor {res},
  GeoAnchor.from_geometry(..., resolution=10), and old catalog write arguments.
- examples/03_inputs.ipynb uses to_xarray and assumes pixels directly on collection
  parents; current saved catalogs load /collection/grid leaves.
- Both notebook 2 and 3 use template= for to_catalog; the current name is path_format.
- AGENTS.md and CLAUDE.md describe translate/, source.py and StacSource. The xarray
  backend still links to geosave-engine. These are current documentation residue.
- tests/examples/test_notebooks.py already executes notebooks and folded exercise
  solutions under slow/integration markers; extend it rather than add a second runner.

## Task 1 — Finish the source and module audit

Files: src/geosave/geodata/{io,stac,core}, affected model/flow callers,
pyproject.toml entry points; mirrored tests.

- [x] Inventory remaining empty/cache-only obsolete directories; remove only
  verified residue, including orphan bytecode for removed modules.
- [x] Trace candidate unused helpers through imports, lazy exports, entry points,
  scripts, tests, notebook cells and docstrings before deleting them.
- [x] Review io/stac.py versus stac/{catalog,item,asset,band}. Keep pixel/file
  operations in io and native STAC metadata conversion in stac; remove duplicate
  implementations only after comparing behavior and callers.
- [x] Clean migration artifacts: scattered imports, redundant dict construction,
  stale helper names/comments and inaccurate parameter descriptions. Keep useful
  format helpers and the xarray backend; they are not empty legacy packages.
- [x] Run the relevant existing round-trip, loader and import-laziness checks.

Deliverable: a navigable module tree with a short deletion ledger naming what
was removed and why. No speculative new public API.

## Task 2 — Consolidate tests around observable contracts

Files: tests/geodata/io/test_odc*.py, test_stac*.py, formats/;
tests/geodata/stac/, tests/geodata/core/test_anchor.py, tests/examples/test_notebooks.py.

- [x] Map overlapping tests to the failure each prevents. Merge repeated setup
  and remove duplicate assertions only when a surviving test protects the same
  values, metadata, laziness or failure boundary.
- [x] Remove tests solely asserting obsolete wrapper plumbing or internal layout.
  Keep signing, fused properties, common attrs, grid equality, relative-resolution
  trees, timeless anchors, preflight failures and persistence round trips.
- [x] Keep test modules aligned with owners; multiple files for meaningful
  workflow coverage are acceptable and do not require artificial source modules.
- [x] Preserve real external notebook/service/browser execution under explicit
  slow/integration checks; replace brittle notebook source-word bans with useful
  artifact/behavior checks where they add no independent protection.
- [x] Record removed tests and their surviving coverage. Run affected tests once
  after consolidation; investigate failures rather than deleting their assertions.

## Task 3 — Migrate the example series together

Files: examples/{00_basics,01_model,02_sources,03_inputs}.ipynb,
examples/README.md, tests/examples/test_notebooks.py.

- [x] Notebook 2: build StacQuery and call set_anchor; select Items explicitly;
  call io.load_dataset with collection metadata, patch_url and typed load options.
  Use label.gs.geobox for image/label alignment. Format resolution from the grid.
- [x] Change fetch_image to accept the anchor and explicit grid; reuse one region
  grid across all monthly loads. Build that grid with native GeoBox and an explicit
  projected CRS/resolution. Keep discovery geometry separate from grid bounds.
- [x] Migrate all to_catalog calls to path_format= and all sample reads to
  to_datatree(). Update markdown, exercises and folded solutions, not just code cells.
- [x] Notebook 3: select raster leaves through existing public DataTree accessors;
  verify singleton time handling and tensor shapes for image/label pairs and
  imagery-only inference. Retain original grids through chip merging/export.
- [x] Review notebooks 0/1 for current model builder, constructor, training and
  export APIs. Change only demonstrated mismatches, preserving the lesson scope.
- [x] Reconcile README instructions, data dependencies and generated filenames.
  Clear invalidated outputs/counts; retain or regenerate outputs only from a real
  execution. Do not fabricate successful plots, metrics or execution counts.
- [ ] Smoke-test notebook 2's local search/load/write path and notebook 3's
  catalog-to-tensor/chip/merge path with small real local rasters before accepting
  the migration. Use the existing notebook test harness for full execution,
  including solutions, in lesson order and isolated working directories.

Full notebook runs require the teaching dataset, a live STAC service, kernels,
and training time. Report exactly which notebooks/cells executed; syntax checks
or injected local fixtures do not prove the live lesson passed.

## Task 4 — Reconcile maintained documentation

Files: README.md, AGENTS.md, CLAUDE.md, docs/examples.md, docs/basics/,
docs/concepts/raster-grids.md, docs/guides/{stac,catalogs-and-cuts,workflows}.md,
docs/reference/, docs/contributing/architecture.md, src/geosave/geodata/io/backend.py.

- [x] Correct the module map and ownership paragraphs in repository guidance,
  preserving every unrelated instruction and local edit.
- [x] Explain geometry/time versus exact grid, set_anchor replacement semantics,
  selected Items, signing, typed options, per-time properties and DataTree leaves.
- [x] Remove unused anchor variables introduced into grid-only examples and update
  printed outputs whenever shape, coordinates or object representation changed.
- [x] Correct stale links and references, including the backend repository URL.
- [x] Regenerate API pages via scripts/api_reference.py rather than editing them.
- [x] Execute local guide examples and run the Zensical documentation build;
  investigate unresolved API references and broken internal links.

## Task 5 — Final validation and handoff

- [x] Run Ruff on changed Python and notebook code using repository configuration.
- [ ] Validate notebook JSON, execute local smoke workflows, then run eligible
  notebook integration tests; explicitly report skipped external checks.
- [x] Run scripts/api_reference.py --check and the documentation build.
- [x] Run affected tests, then the default suite; compare failures with the
  pre-cleanup baseline by node and cause. Prior refactor result: 1394 passed,
  37 failed, 41 deselected, 3 xfailed; all failures reproduced before that refactor.
- [x] Recheck obsolete paths/references excluding explicitly historical documents.
- [x] Report removed modules/tests, updated lessons, actual checks, and remaining
  blockers. Do not claim the suite or notebook series is green unless it is.

## Review focus

1. One selected row and a frame retain geometry/time without inventing a grid.
2. Images align exactly with labels; repeated monthly loads use the same grid.
3. Catalog reads expose grid leaves, with tensor axes and time handled explicitly.
4. Signed asset reads retain lazy behavior and persisted catalogs remain portable.
5. Examples can rerun from a fresh kernel without stale outputs hiding failures.

Tasks 1–3 retain or extend existing workflow checks for these conditions; task 5
records their actual execution. ModelSpec redesign and training-template repair
remain separate unless the example migration exposes a required shared fix.

## Execution record

- Baseline snapshot for this cleanup: `/tmp/geosave-cleanup-before`. The shared
  checkout and unrelated edits are preserved; no commits or resets were made.
- Source audit: every top-level I/O/STAC definition has a caller/reference except
  GeoSaveBackendEntrypoint, which is registered in pyproject.toml. Retained that
  integration rather than misclassifying an entry point as dead code.
- Removed cache-only tests/geodata/catalog and pipeline directories, plus orphan
  bytecode for removed source modules. The six original legacy directories remain
  absent. Normal caches for existing source files are not blanket-deleted.
- Cleaned migration imports, stale source docstrings, backend repository link and
  nested options construction; kept format and storage ownership unchanged.
- Removed one duplicate one-Item load test and retained its time-axis assertions
  in the existing alias/packing workflow. Removed notebook source-word bans;
  retained execution, artifact and training-metric checks. No model/template
  failures were deleted or marked expected failures.
- Added one local example workflow check that executes the actual notebook
  functions against real saved assets: query/load/write, catalog pairing, native
  grid leaves, tensor shapes, chip merging and imagery-only inference.
- Ruling: notebook 3 consumes notebook 2's paired catalog, because the cached
  teaching dataset has no catalog.parquet. Original download still supplies
  classes; the ten-tile notebook 2 dataset supplies training/validation/test.
- Migrated all four notebooks to native xarray construction/copy, selected-Item
  loading and current catalog writers. Updated exercises/solutions and cleared
  stale outputs and execution counts. Full runs save evidence under
  `/tmp/geosave-notebook-run`, never overwrite the user's cached examples/data.
- Corrected scalar-time handling after a local failing smoke check. Notebook
  readers squeeze only an existing singleton time axis, preserving spatial axes.
- Updated repository guidance and docs to geometry/time versus explicit grids.
  Local doc execution also exposed stale writer return-value and TIFF-list reader
  assumptions; corrected those and regenerated displayed outputs from execution.
- Current checks: 369 focused tests passed before duplicate-test consolidation;
  all four notebooks and folded solutions validate/parse. 49 local documentation
  blocks pass, credential-dependent storage setup is excluded. Zensical builds.
- Independent read-only review against the cleanup snapshot found no concrete
  correctness issues; full notebook/live-service verification was still pending.
- Final targeted verification after import alias cleanup and test consolidation:
  69 passed, 2 deselected, 11 warnings. Full default suite: 1394 passed, 37 failed,
  41 deselected, 3 xfailed, 118 warnings (192 seconds). The exact 37 failing nodes
  match the pre-cleanup run; no new failing nodes. Logs:
  `/tmp/geosave-cleanup-final-targeted.log`, `/tmp/geosave-cleanup-full.log`.
- Ruff check and format verification passed for all 49 changed Python/notebook
  files. Generated API freshness passed. Final Zensical build passed (18 seconds).
- Notebook 0 and 1 completed with exercise solutions in fresh kernels using
  copied cached teaching data. Browser launches were disabled for verification.
- Live notebook 2 reached the multi-label acquisition cell and saved several
  real image/label pairs, but exceeded this verification run's 180-second cell
  limit. The normal notebook integration harness allows 600 seconds. This is
  incomplete live validation, not a passing full notebook run; no examples were
  weakened to conceal the timeout. Notebook 3 continues against copied cached
  catalogs rather than claiming fresh notebook-2 outputs from this run.
- The two notebook-execution checkboxes remain open solely for complete live
  acquisition verification; JSON/solution validation and the local end-to-end
  workflow portions of those steps passed. Implementation and documentation
  cleanup are complete; the external execution limitation is retained visibly.
- Notebook 3 completed end-to-end: training, held-out evaluation, forest chip
  inference/merge, and prediction catalog export. Final notebook outcome:
  00_basics, 01_model and 03_inputs passed; 02_sources timed out during its live
  acquisition batch. Execution log: `/tmp/geosave-notebooks.log`; executed
  notebooks are in `/tmp/geosave-notebook-run/`.
