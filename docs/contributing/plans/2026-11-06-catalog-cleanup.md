# Catalog and raster time cleanup

User-approved scope: remove GeoCatalog.query in favor of native GeoPandas
selection, consolidate RasterTime in base.py, and resolve sampling and catalog
ownership without compatibility aliases or pixel preparation in GeoSamples.

Implementation uses the writing-plans and executing-plans workflows inline.
The current dirty checkout is the required input; retain all unrelated changes.

- [x] Add regression cases for auxiliary time dimensions, plain-feature date
  selection, excluded invalid geometry, positional slot membership, direct
  writer protection and loader failure cleanup. Observe the new failures.
- [x] Move RasterTime beside GeoRaster; separate bounds cardinality cases and
  bind per-label bounds to the actual time coordinate dimension. Preserve lazy
  pixels and DataTree ownership.
- [x] Delete query and its private helpers/types. Remove query-only tests and
  replace active documentation examples with native selection.
- [x] Centralize vector geometry area and temporal guarantees in gs.validate.
  Share temporal decoding between anchor and sampling without collection or
  spatial dependencies. Keep the explorer's open-ended filter semantics intact.
- [x] Refactor sampling into selection, footprint grouping, coverage clipping,
  calendar construction, slot assignment and window selection. Use original
  positional identities through slot assignment; keep static-only window rules
  with window selection. Retain all existing calendar policies.
- [x] Move collection loading into io/stac.py with explicit resource ownership;
  move metadata-loss protection into the ordinary vector format writers.
  Validate concat CRS through the existing accessor and refresh docstrings.
- [x] Run focused workflow tests, lint, generated reference checks and review
  the change against the task-start snapshot. Record exact results below.

Review focus: duplicate source labels, empty selections, all-static groups,
interval endpoint inclusion, close-on-failure and untouched lazy pixel buffers.

Decisions: no changes to ModelSpec, notebook files, or GeoSamples loading API.
Temporal validation remains opt-in for generic tables; anchor and calendar
sampling request complete ordered coverage. General selection uses GeoPandas.

## Verification

- Regression-first run reproduced auxiliary time dimensions (DataArray failure,
  Dataset extra dimension), plain-feature date filtering, excluded invalid
  geometry and unprotected direct writers. A precision regression introduced
  during decoding was reproduced and fixed before final verification.
- Final scoped run: 239 passed, 8 deselected, 16 warnings in 15.76s. Covers
  vector accessors/sampling, raster base, anchors, STAC I/O, GeoJSON/GeoPackage/
  GeoParquet, STAC catalog conversion and temporal transforms. Deselections are
  the configured integration marker exclusions; the full suite was not run.
- Ruff check and format check passed for all 20 changed Python files.
- Generated API reference freshness and git diff --check passed.
- BasedPyright passed for changed raster/vector/format/STAC catalog code.
  Including io/stac.py reports one pre-existing Dataset | DataTree mismatch
  in _open_asset. Reproduced on the task-start snapshot at line 220; unchanged.
- Independent review found no actionable introduced issues. Existing scene
  close-once coverage and new open/concatenate failure cleanup checks passed.

## Compatibility

- GeoCatalog.query and its predicate types were removed; use native GeoPandas
  masks with explicitly reprojected target geometry. No aliases remain.
- RasterTime now lives in core/raster/base.py. Its tests were consolidated into
  test_base.py. Shared bounds are scalar even for a single labelled observation.
- Catalog anchors now request the same complete, ordered coverage guarantee as
  temporal sampling, including rejection of nominal time outside coverage.
- Direct GeoJSON/GeoPackage writes now reject STAC metadata loss just as the
  accessor writers already did. GeoParquet retains the catalog round trip.
- No notebook, ModelSpec, or unrelated pre-existing edits were changed.
