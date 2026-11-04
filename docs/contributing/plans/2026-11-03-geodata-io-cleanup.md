# Geodata I/O cleanup

The reviewed direction is to keep native objects and the public `to_catalog`
workflow while clarifying asset publication and catalog persistence. This work
uses the existing checkout so its ongoing refactors remain available; it does
not commit, reset, or move unrelated changes.

## Design

- Rename raster `catalog.write` to `publish.publish_rasters`. Build and validate
  observation metadata and identities before asset writes; retain actual writer
  destinations in the returned catalog. Saving the table remains `to_geoparquet`.
- Keep template expansion pure. Let raster writers own format preflight and
  batch execution; remove format validation loops from accessors. Share one
  single/group execution mechanism and preserve native Dask delayed results.
- Rename storage `write` to `write_file`, remove the unused async GeoTIFF path,
  and remove its direct dependencies if they have no remaining source consumers.
- Share URL suffix extraction across openers, preserving signed URLs themselves.
- Put shared vector driver options in `io/vector/gdal.py::prepare_read_options`.

Renaming every format writer would add noise: qualified `netcdf.write` already
identifies its output. Combining metadata translation with file serialization
would mix responsibilities. Compatibility aliases are not added.

## Execution

- [x] Add failing tests for invalid Item IDs, duplicate identities, and signed
  vector URLs; verify failures leave no assets or existing assets untouched.
- [x] Implement publication preflight and naming; run catalog round trips.
- [x] Consolidate raster batch validation/execution; verify later invalid COGs
  prevent earlier writes, including deferred groups.
- [x] Clean storage and vector ownership; update callers and documentation.
- [x] Run focused I/O checks, broader tests, lint, API reference refresh, and
  independent review. Record exact results and limitations.

## Review focus

Deferred writes recheck destinations at compute time. Encoding follows selected
variables. Group destinations cannot collide. Publication metadata uses actual
written paths. Item metadata failures occur before any raster computation.
I/O failures after serialization begins do not promise batch rollback.

## Validation

- Focused I/O, raster accessors, STAC translation and ingest: 447 passed,
  17 deselected. Final shared vector driver rename: 8 passed, 4 deselected.
- Full suite: 1,412 passed, 32 failed, 41 deselected, 3 xfailed.
  All 32 failures reproduced against pre-change I/O/accessor source. They are
  generated workspace/training templates still using `ModelSpec.chips`.
- Scoped Ruff passed; type checking reported zero errors and one existing lazy
  STAC export warning. Generated API freshness and documentation build passed.
- Independent review found incompatible-property Arrow conversion could fail
  after writes. Conversion now runs before writing; new/existing output tests
  failed before the fix and passed after it. Follow-up review had no findings.

Low-level breaking changes: `storage.write_file` replaces `storage.write`,
`publish.publish_rasters` replaces `catalog.write`, shared vector driver options
move to `gdal.prepare_read_options`, unused async GeoTIFF helpers are removed,
and raster batch writers derive suffixes from the selected format instead of
accepting `suffixes=`. Public `gs.to_catalog` continues returning an in-memory
catalog after writing its assets; `gs.to_geoparquet` persists the catalog table.
