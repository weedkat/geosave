# Public openers and discoverable options

Implement the accepted convenience API: `open_raster` and `open_vector`, with
explicit shared arguments, format-specific `open_*` / `write_*` names, and
literal self-contained docstrings suitable for IDE help.

## Decisions

- Preserve stored child groups as DataTree; root-only stores return Dataset.
  Retire folder/mapping assembly and `open_tree`; compose native trees explicitly.
- Ruling: convenience `chunks={}` uses Dask with backend-preferred chunks across
  formats. This changes the previous NetCDF/Zarr convenience default; callers
  wanting backend indexing can pass `chunks=None`. Precise format defaults remain.
- Ruling: storage options use the filesystem instead of mutating GDAL credentials
  globally. NetCDF/GeoJSON/GeoPackage stage downloads; Rasterio stages explicit
  filesystem reads until the Dataset closes. Global `configure_gdal` remains
  available for native GDAL URL access without explicit filesystem options.
- Geometry remains selected when vector columns omit it. Bbox delegates spatial
  filtering to the backend; unsupported GeoParquet coverings fail explicitly.
- Typed format options own their descriptions; a generator copies these and one
  storage reference into literal opener docstrings. Nested SDK dictionaries are
  documented extension points, not an exhaustive promise about arbitrary versions.
- Ruling: keep the shared dirty checkout and compare against the pre-task copy at
  `/tmp/geosave-openers-before`, not HEAD. No unrelated changes are reverted.

## Validation ledger

- First I/O run: 225 passed, 5 failed (obsolete global-S3 mutation expectations).
- Updated contract tests: 334 passed, 1 failed, 16 deselected. The real memory
  GeoTIFF round trip caught duplicated VSI registration during rioxarray reopen.
- Temporary staging replaced the failed custom-opener approach. A lifetime test
  verifies downloads remain available for lazy reads and are removed on close
  and on failed reads.
- Independent review found two regressions, both reproduced before fixing:
  xarray engine chunk control (2 failing cases) and HF repository staging
  (1 failing case). All three now pass.
- Reviewed affected suite: 338 passed, 16 deselected, 43 warnings.
- Final opener/backend/local-notebook smoke: 36 passed, 4 deselected, 3 warnings.
- Ruff passes; literal docstring freshness and API reference freshness pass.
  Zensical build passes; rendered API coverage is 1291/1291. The docs workflow
  now checks opener docstring freshness.
- First broad run: 1406 passed, 37 failed, 41 deselected, 3 xfailed; failure
  names exactly match the prior full-suite baseline.
- Final broad run: 1411 passed, 39 failed, 41 deselected, 3 xfailed, 118
  warnings. The 37 baseline failures remain; two additional failures belong to
  newly added concurrent Explorer tests:
  `test_obsolete_catalog_result_does_not_change_selection[True-close]` and
  `[False-close]`. Neither test existed in the pre-task snapshot. Both fail
  because viewer.close emits a table update; no opener is involved in their
  failing assertion. Concurrent Explorer implementation/tests are left untouched.
- Automatic approval review rejected broad restoration of historical docs.
  No restoration executed; inspected exact differences and reversed only this
  task's specific renames in six historical plan/spec files. Other work preserved.

- Ruling: the installed Rasterio/rioxarray custom opener crashed in cleanup.
  Use temporary staging for explicitly configured raster filesystems instead.
  Cost: these reads download the whole file; native GDAL URL access without
  explicit options retains windowed remote reads. Document this distinction.

- Final isolated I/O verification: 339 passed, 16 deselected, 43 warnings in 30.75s.
