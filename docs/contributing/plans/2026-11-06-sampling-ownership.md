# Sampling ownership correction

User-approved scope: GeoSamples stays a row grouper; indexing returns the native
GeoDataFrame. No GeoSample wrapper, preparation settings, hidden frame attrs, or
loader on the grouper. Keep temporal membership as inspectable grouping metadata.

- [x] Remove reduce/interpolate/max_gap from sample and remove GeoSamples loading.
  Keep interval assignment, exact/overlap grouping, windows and missing=drop/keep.
- [x] Delete vector/preparation.py; existing I/O and temporal transforms retain
  their responsibilities and semantics. Update meaningful workflow tests.
- [x] Move raster support into core/raster/utils, geometry and sampling support
  into core/vector/utils, and mirror tests. Rename summary to statistics.
- [x] Update active docs and generated references, run targeted tests, typing,
  lint and an explicit lazy read/resample/reduce smoke test.

Boundary: interval policies affect row membership, not raster timestamps. Loading
returns available source observations; missing slots remain grouping metadata
until the caller explicitly resamples/reindexes native raster data.


## Verification

- Native-row smoke: 12 passed before implementation, confirming the existing row
  accessor and explicit temporal transforms already support the intended flow.
- Final targeted run: 363 passed, 12 deselected, 19 warnings. Includes vector,
  raster support, raster accessors, temporal transforms and model processing.
- Broader initial runs stalled and were interrupted. Verbose diagnosis stopped
  at test_tree_round_trip_preserves_groups_and_grid, which writes Zarr. Final run
  excluded Zarr-named cases and four additional tree storage cases; default
  marker deselections also apply. No claim that Zarr or the full suite passed.
- Ruff, generated API freshness, and git diff --check pass.
- Vector BasedPyright: 0 errors. Broader touched-source check: one pre-existing
  possibly-unbound fill error in raster/utils/statistics.py:156. Confirmed the
  same class_percent body in HEAD:geodata/utils/summary.py; behavior unchanged.
- Deleted vector/preparation.py rather than retaining a second transform path.
  Existing temporal.interpolate semantics, including duration-based max_gap,
  are unchanged. No GeoSample wrapper or implicit frame attrs introduced.
