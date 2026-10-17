# Geodata test review — 2026-10-13

Scope: all **92 Python test/support files** present at the immediate pre-edit snapshot,
plus all newly created and relocated owner suites. All original test contents were
reviewed for live imports, registered fixtures, assertion meaning, optional dependency
boundaries, duplication and current source ownership. Native calls were cross-checked
against current implementations and exports. The resulting inventory contains
**94 Python files**, with **1,097 -> 1,091 test definitions** and **15,449 -> 15,415 lines**.
Parameterized test cases are preserved by the consolidations below; only one duplicate
runtime case was removed. Current collection selects **1,449 / 1,469 cases**, with
20 existing integration cases deselected by project configuration.

The shared worktree changed during inspection. In particular, the other active work
renamed `io/readers.py` and its test to `openers.py`, renamed format opening APIs,
and corrected the GeoTIFF datetime metadata contract and explorer typing.
Those current changes were preserved, not reverted to HEAD or the earlier root snapshot.
The immediate pre-edit tests are recorded under
`/tmp/geosave-docs-tests-audit/geodata-pre-edit/`.

## Changes and retained evidence

- Replaced obsolete core `test_base`, `test_axes`, `test_rgb` and `test_model_io`
  module names by placing shared accessor, constructor, RGB, native-array and
  anchor/vector/stack behavior with their actual source owners. All existing
  numerical values, failure cases, laziness checks and persistence assertions remain.
- Gave GeoArray its own source-mirroring suite. Its layout/statistics/vectorize/
  `to_raster` cases had previously been scattered across raster/vector/model-I/O tests.
- Moved GeoTIFFTags units to `io/raster/test_tags.py` and RasterFile units to
  `io/raster/test_file.py`. COG and GeoTIFF integration round trips remain in place.
- Renamed package initializer suites to `test_init.py`, moved the package-owned
  reference dispatcher cases into `test_openers.py`, and placed StacQuery-only
  validation in `stac/search/test_query.py`. `load_attrs` remains tested against
  its actual owner, `stac/translate/__init__.py`.
- Consolidated the five identical attr-group key coverage assertions into one
  parameterized base contract with their same Nodata, Packing, Spectral, GDALBand
  and Legend instances. Both failed GeoParquet encoding states remain separate
  parameter cases (`new`, `overwrite`) with their original transactional assertions.
- Removed only `test_a_file_naming_no_date_carries_no_time`: it duplicated
  `test_an_undated_file_reads_with_no_time`, using the same undated native raster,
  COG writer, GDAL opener and absence-of-time assertion. Neither filename carries
  a date and no distinct branch was exercised.
- Shared exact duplicate band, STAC asset, catalog rows and visualization grid
  builders only within their direct owner scopes. Reused the existing geodata
  grid builder for the identical two-by-two temporal/packing grids. Helpers remain
  ordinary functions and pytest fixtures; no test framework was introduced.
- Repaired the claimed far-edge crop/window case: its old `(0, 1)`, `(1, 1)`
  request stayed inside the two-by-two source. It now requests `(3, 2)` at that
  same offset, checks right/bottom fill pixels against an independent literal,
  verifies the resulting grid and protects lazy execution with a Dask callback.
- Replaced module-wide `pytest.importorskip("torch")` with a function-scoped
  optional fixture, and made dtype parameter declarations import-independent.
  A process that rejects every torch import runs all **12** original NumPy
  conversion cases and skips exactly the **11** tensor cases.
- Removed two obsolete `hvplot` import guards from live GeoStack plot tests.
  The current path uses core HoloViews/Matplotlib dependencies and never imports
  hvplot. Both previously skipped tests now run and pass.

## Checks

- Before-edit behavioral smoke: **114 passed**, 1 warning in 3.93 s.
- First changed-owner smoke: **554 passed, 2 skipped**, 32 warnings in 16.75 s.
  The two skips exposed the stale hvplot guards; those were removed afterward.
- Isolated previously skipped plot behavior: **2 passed**, 37 deselected in 5.40 s.
- Final affected-scope suite: **719 passed**, 57 warnings in 37.23 s, zero skips.
  Command:

  ```bash
  .venv/bin/python -m pytest tests/geodata/core tests/geodata/attrs tests/geodata/stac/extensions tests/geodata/stac/translate/test_init.py tests/geodata/stac/test_init.py tests/geodata/stac/search/test_query.py tests/geodata/io/raster/test_cogs.py tests/geodata/io/raster/test_file.py tests/geodata/io/raster/test_geotiff.py tests/geodata/io/raster/test_tags.py tests/geodata/io/test_init.py tests/geodata/io/test_openers.py tests/geodata/io/vector/test_geoparquet.py tests/geodata/io/vector/test_geojson.py tests/geodata/io/vector/test_stac_json.py tests/geodata/transform/test_crop.py tests/geodata/transform/test_nodata.py tests/geodata/transform/test_temporal.py tests/geodata/viz/test_map.py tests/geodata/viz/explorer -q --durations=10
  ```

  Persistence and local runtime tests ran with the runtime/socket permissions
  the root baseline established Zarr requires in this environment.
- Torch-unavailable subprocess smoke: **12 passed, 11 skipped**, 270 deselected
  in 0.21 s. Uses `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and an import finder that raises
  `ModuleNotFoundError` only for torch; pytest selects the original NumPy/tensor
  cases by their current names. Tensor skips are deliberate dependency absence.
- `.venv/bin/python -m pytest tests/geodata --collect-only -q`:
  **1,449 / 1,469 collected**, 20 deselected, no collection errors.
- `.venv/bin/ruff check tests/geodata`: passed.
- `git diff --check` for every file changed by this review: passed.
- AST inventory reconciliation proves the only removed test names are the five
  repeated key assertions, two encode-failure bodies and the duplicate undated
  GeoTIFF case. Their two parameterized replacements retain the seven original
  serialization/failure cases. No same-file test-name collisions were introduced.

## Remaining risks and deliberate exceptions

No production source, notebooks, public APIs or dependencies changed in this review.
No new source defect was found that needed weakening an assertion or adding an xfail.
The existing narrow ODC lazy/eager numerical xfails and month-end TimeSpec concern
remain source/dependency work. No new slow marker was justified: measured normal
conversion/persistence checks stay in the default suite, and remote service tests
already carry the integration marker. Final integrated default/slow checks belong
to the root review and supersede these scoped results.

Companion suites remain where they exercise cross-module/public workflows or distinct
large regression contracts: frame labels, warp edges, stack metadata, Zarr stack
storage, real remote services, native producer Items and observation dataflow.
They are listed explicitly in the ledger, so filename matching is not mistaken
for a reason to delete useful behavior. Timings are check evidence, not a speedup
claim against the changing baseline checkout.

## Per-file coverage ledger

Paths are relative to `tests/geodata/`; definition counts include test methods,
not expanded parameter cases. Fixtures and other support files have zero test definitions.

| File | Definitions before -> after | Inspection and disposition |
| --- | --- | --- |
| `attrs/conftest.py` | 0 -> 0 | One shared band factory for parse/stamp; no new abstraction. |
| `attrs/groups/test_acdd.py` | 3 -> 3 | ACDD key filtering and absent/default fields retained. |
| `attrs/groups/test_base.py` | 4 -> 5 | Registration, aliases, omitted defaults/None retained; five concrete serialization-key cases parameterized here. |
| `attrs/groups/test_gdal.py` | 4 -> 3 | Enum/native interpretations, unknown colors and GDAL names retained; generic key assertion shared in base. |
| `attrs/groups/test_is_flag.py` | 5 -> 0 | Moved intact to `attrs/groups/test_legend.py`; removed old filename. |
| `attrs/groups/test_legend.py` | 13 -> 17 | CF flags, bit masks, palettes, dtype checks and all format round trips retained; categorical guard cases moved here. |
| `attrs/groups/test_nodata.py` | 10 -> 9 | NaN/integer/float precision and overflow, aliases and Zarr round trip retained; generic key assertion shared in base. |
| `attrs/groups/test_packing.py` | 5 -> 4 | CF scale/offset, strings and zero-scale refusal retained; generic key assertion shared in base. |
| `attrs/groups/test_spectral.py` | 4 -> 3 | Spectral fields, strings and wavelength refusal retained; generic key assertion shared in base. |
| `attrs/groups/test_timespec.py` | 4 -> 4 | Half-open start/end labels, gaps and frequency multiples retained; known month-end semantics deferred to source work. |
| `attrs/test_combine.py` | 10 -> 10 | Strict pixel semantics, conflicting/absent fields, NaN equivalence and per-key root merges retained. |
| `attrs/test_parse.py` | 4 -> 4 | Group absence, dtype validation, text/NumPy values and root attrs retained; duplicate factory consolidated. |
| `attrs/test_stamp.py` | 5 -> 5 | Alias replacement, removal, source preservation and no-compute stamping retained; duplicate factory consolidated. |
| `benchmarks/test_dynamic_world.py` | 9 -> 9 | Local download fixtures preserve staging/reuse, checksums, extraction failures and path traversal coverage; no slow/external marker needed. |
| `conftest.py` | 0 -> 0 | Native raster/grid/scene/label builders, COG templates and Item recovery retained; exact duplicate catalog rows fixture shared here. |
| `core/conftest.py` | 0 -> 0 | Existing optical builder shares the existing geodata grid factory; torch import skips only tensor consumers. |
| `core/test_accessor.py` | 8 -> 27 | Shared registration/types, variable identity, times, foreign-grid refusal, lazy decoding and pooled statistics; shared tensor dtype guards retained. |
| `core/test_anchor.py` | 4 -> 10 | Immutable span replacement, extent, centroid, place lookup and template validation; anchor cases moved from obsolete base suite. |
| `core/test_array.py` | 0 -> 28 | Native array layout, metadata splitting, writable tensors, statistics and eager vectorization; cases moved from model_io/raster/vector. |
| `core/test_axes.py` | 2 -> 0 | Moved intact to `core/test_create.py`; removed old filename. |
| `core/test_base.py` | 23 -> 0 | Moved intact to `core/test_accessor.py; anchor, array and vector owner suites`; removed old filename. |
| `core/test_catalog.py` | 10 -> 10 | Catalog dispatch, Item conversion, all collection metadata and writer policies retained; rows factory shared. |
| `core/test_create.py` | 38 -> 43 | Explicit dimensions, exact reference coordinates, inherited attrs, GCP grids and persistence retained; CF axes and constructor cases moved here. |
| `core/test_model_io.py` | 21 -> 0 | Moved intact to `core/test_raster.py; array, stack and accessor owner suites`; removed old filename. |
| `core/test_raster.py` | 45 -> 48 | Dataset metadata/writers, array stacking, RGB interpretation and NumPy/tensor conversion; array/shared/constructor cases moved to owners. |
| `core/test_rgb.py` | 10 -> 0 | Moved intact to `core/test_raster.py`; removed old filename. |
| `core/test_stack.py` | 35 -> 36 | Root/group grid and time, independent storage, lazy mapping and conversion retained; obsolete hvplot skips removed from live HoloViews plot cases. |
| `core/test_vector.py` | 38 -> 31 | Native geometry, queries, predicates, temporal columns and rasterization retained; array vectorization and reader provenance moved to owners. |
| `cuts/conftest.py` | 0 -> 0 | Local per-source raster series and actual COG/catalog fixtures retained. |
| `cuts/test_chips.py` | 11 -> 11 | Tiling, halos, fringe padding, grids, parent keys and validation retained. |
| `cuts/test_frames.py` | 14 -> 14 | Local-file frame behavior and scene-per-instant loading retained. |
| `cuts/test_frames_labels.py` | 17 -> 17 | Companion frames suite retained: pure label pairing, strides, gaps, ties and missing/duplicate instants; no raster I/O duplication. |
| `cuts/test_merge.py` | 9 -> 9 | Numerical reconstruction, tapered overlap, invalid pixels, missing output and COG round trip retained. |
| `cuts/test_sample.py` | 9 -> 9 | Window-to-lazy-stack, grids, padding, persisted rows, multiband identity and catalog metadata recovery retained. |
| `cuts/test_stacks.py` | 10 -> 10 | Anchor joins, partial cover/duplicates warnings, collections, absolute hrefs and source metadata retained. |
| `cuts/test_windows.py` | 3 -> 3 | Accessor integration retained alongside direct cuts function behavior. |
| `features/conftest.py` | 0 -> 0 | Deterministic Sentinel reflectance/SCL/sun metadata builder retained. |
| `features/test_cloud_mask.py` | 11 -> 11 | Float/unsigned boundaries, chunk halos, NaNs, repeated selectors, model scheduling and metadata preservation retained. |
| `features/test_dask_mapping.py` | 5 -> 5 | Mixed eager/lazy backends, halos, leading-axis independence and grid ordering retained. |
| `features/test_shadow_mask.py` | 7 -> 7 | Physical shadow directions/distances, per-date azimuth, chunk independence and refusal cases retained. |
| `features/test_spectral_indices.py` | 8 -> 8 | All selected-band formulas across float/packed/decoded eager/lazy data, absent pixels and unsigned subtraction retained. |
| `io/raster/test_cogs.py` | 38 -> 35 | Template layouts, date/band identity, storage, native grid slicing and mixed-resolution reads retained; RasterFile cases moved to file. |
| `io/raster/test_file.py` | 0 -> 3 | RasterFile headers, native array/tree behavior and local-path protocol cases moved from COG suite. |
| `io/raster/test_gdal.py` | 17 -> 17 | GDAL configuration, metadata, lazy native file chunks, S3/HF option propagation and state replacement retained. |
| `io/raster/test_geotiff.py` | 50 -> 42 | CF/GDAL metadata and pixel round trips, palettes, COG layout, GCP refusal and streamed writes retained; tag-only units moved; one exact no-date duplicate removed. |
| `io/raster/test_netcdf.py` | 7 -> 7 | CRS/dtype/time round trips, deferred writes, file URLs, remote-write limits and foreign-grid normalization retained. |
| `io/raster/test_tags.py` | 0 -> 7 | Native GeoTIFFTags field precedence, projected unit conversion, precise instant and map-scale validation moved from GeoTIFF suite. |
| `io/raster/test_zarr.py` | 18 -> 18 | Profile/packing/fill/order, deferred writes, remote stores, overwrite and foreign-grid round trips retained. |
| `io/raster/test_zarr_stack.py` | 6 -> 6 | Companion Zarr regressions retained: child CRS/scalar time, changed cwd, selected stacked attrs and rewritten native children. |
| `io/test_backend.py` | 6 -> 6 | Native xarray engine registration, read/write parity, lazy option forwarding and DataTree loading retained. |
| `io/test_init.py` | 0 -> 1 | Public package export contract moved from public_surface companion name. |
| `io/test_openers.py` | 17 -> 20 | Actual suffix dispatcher and owned resource cleanup; old reference interoperability cases and reader provenance moved here. |
| `io/test_public_surface.py` | 1 -> 0 | Moved intact to `io/test_init.py`; removed old filename. |
| `io/test_reference.py` | 2 -> 0 | Moved intact to `io/test_openers.py`; removed old filename. |
| `io/test_remote.py` | 7 -> 7 | Companion S3/Hugging Face end-to-end suite retained with existing integration marker and explicit environment requirements. |
| `io/test_storage.py` | 17 -> 17 | Location conversion, option merging, staging/overwrite/failure and remote upload semantics retained. |
| `io/vector/test_geojson.py` | 6 -> 5 | Nested mappings, remote upload, failed overwrite and native driver options retained; GeoParquet-only case moved. |
| `io/vector/test_geopackage.py` | 5 -> 5 | Native layer naming, overwrite, remote URLs and duplicate driver options retained. |
| `io/vector/test_geoparquet.py` | 10 -> 10 | BBox/columns, empty data, file URLs, stale staging and remote failure states retained; failures parameterized and nullable-assets case moved here. |
| `io/vector/test_stac_geoparquet.py` | 36 -> 36 | Item/table/asset identity, metadata, schema fields, movable hrefs, collection persistence and load-back pixels retained. |
| `io/vector/test_stac_json.py` | 8 -> 8 | Static JSON catalog, validation, movable relative assets, links, duplicates/overwrite and remote filesystem retained; rows fixture shared. |
| `stac/conftest.py` | 0 -> 0 | Actual local multiband raster assets and catalog/config/anchor fixture retained. |
| `stac/extensions/conftest.py` | 0 -> 0 | One direct asset factory replaces four identical factories; owned by schema test scope. |
| `stac/extensions/test_classification.py` | 7 -> 7 | STAC class writes/reads, invalid names, bit masks and Raster extension precondition retained. |
| `stac/extensions/test_eo.py` | 4 -> 4 | Per-band identity, spectra, descriptions and missing spectral metadata retained. |
| `stac/extensions/test_projection.py` | 3 -> 3 | Affine/EPSG/WKT projection facts and absent-grid failure retained. |
| `stac/extensions/test_raster.py` | 7 -> 7 | Dtype/nodata/packing/unit translation and no-compute header writes retained. |
| `stac/search/test_api.py` | 6 -> 6 | Transport preservation, 404-vs-503 distinction, provider constructors and deferred signing retained. |
| `stac/search/test_query.py` | 0 -> 1 | Query constructor validation moved from source suite to actual query owner. |
| `stac/search/test_table.py` | 12 -> 12 | Local STAC queries, spatial/time filters, native DuckDB options, local S3 endpoint and Collection/load parity retained. |
| `stac/test_create.py` | 36 -> 36 | Collection/Item builders, scene boundaries, template IDs, extensions, native ODC interoperability and no eager reopening retained. |
| `stac/test_dataflow.py` | 3 -> 3 | Companion public workflow retained: observations -> decoded reflectance -> index -> persisted lazy dataset; schema validation remains integration. |
| `stac/test_init.py` | 0 -> 2 | Package exports and root aliases moved from namespace name. |
| `stac/test_namespace.py` | 2 -> 0 | Moved intact to `stac/test_init.py`; removed old filename. |
| `stac/test_native_items.py` | 7 -> 7 | Companion producer interoperability retained: native Items, new fields, role filtering, bbox edits and portable assets. |
| `stac/test_source.py` | 19 -> 18 | Query merging, source config, lazy/eager loads, alias indices, authoritative nodata/packing and ACDD retained; query-only constructor guard moved. |
| `stac/translate/test_asset.py` | 4 -> 4 | Core-band precedence, legacy lists, nullable fields and assets without bands retained. |
| `stac/translate/test_band.py` | 2 -> 2 | Band attrs and empty metadata translation retained. |
| `stac/translate/test_collection.py` | 2 -> 2 | ACDD root attrs and observed collection extents retained. |
| `stac/translate/test_init.py` | 0 -> 12 | load_attrs is owned by translate/__init__; disagreement warnings, missing assets, loader precedence and metadata selection retained. |
| `stac/translate/test_item.py` | 10 -> 10 | File and scene grouping, multiband names, extension/core metadata and collection/date/grid identity retained. |
| `stac/translate/test_load_attrs.py` | 12 -> 0 | Moved intact to `stac/translate/test_init.py`; removed old filename. |
| `test_conventions.py` | 8 -> 8 | Foreign dimensions normalized lazily, native/gridless objects, root/group behavior and clear failures retained. |
| `transform/conftest.py` | 0 -> 0 | Direct placed/packed/classified lazy raster fixtures and regular-grid extraction retained. |
| `transform/test_color.py` | 3 -> 3 | CF class colors, unknown-code NaN, scalar coordinates and lazy computation retained. |
| `transform/test_combine.py` | 48 -> 48 | Exact bands/time, nodata/packing/class/units agreement, output grids, mosaics and overlap order retained. |
| `transform/test_crop.py` | 19 -> 19 | CRS-aware geometry crop, mask pixels, native types/time/grids, halo/reflection and laziness retained; far-edge test now checks actual constant fill. |
| `transform/test_nodata.py` | 42 -> 42 | Representable fills, all native kinds, no mutation, mask shape/dims, lazy decode/unpack and dtype precision retained; identical packing-grid helper removed. |
| `transform/test_stack_metadata.py` | 1 -> 1 | Companion cross-operation regressions retained: concat/reproject keep shared root metadata and derived grid. |
| `transform/test_temporal.py` | 17 -> 17 | Bucketing, categorical restrictions, nodata policy and real elapsed-time interpolation retained; duplicate grid helper removed. |
| `transform/test_vector.py` | 8 -> 8 | Coordinate-accurate vector conversion, value dtype boundaries, valid connectivity and geographic round trips retained. |
| `transform/test_warp.py` | 34 -> 34 | CRS, resolutions, output grids/fills, categorical kernels, warning attribution and multi-grid alignment retained. |
| `transform/test_warp_edges.py` | 16 -> 16 | Companion numerical Rasterio-reference suite retained, including chunk boundaries and three narrowly scoped dynamic strict xfails. |
| `utils/test_color.py` | 2 -> 2 | Color spellings and invalid input boundaries retained. |
| `utils/test_geocode.py` | 1 -> 1 | Rounded coordinate/cache request behavior retained with mocked network. |
| `viz/conftest.py` | 0 -> 0 | One scoped grid builder for webmap and explorer replaces exact duplicate geometry setup. |
| `viz/explorer/conftest.py` | 0 -> 0 | Deterministic real raster/vector/catalog fixtures retained; shared viz grid reused. |
| `viz/explorer/test_app.py` | 15 -> 15 | Layer order, native panel layout, dates, actions/identification and public accessors retained. |
| `viz/explorer/test_docks.py` | 8 -> 8 | Native UI controls, style/legends, selected-item opening and bidirectional selection retained. |
| `viz/explorer/test_layers.py` | 21 -> 21 | Raster/vector/catalog rendering, time/color limits, identify/highlights and datashader paths retained. |
| `viz/test_map.py` | 11 -> 11 | Lazy coarse/detail views, screen grids, class/nodata colors, geometry families and basemap rendering retained; duplicate grid shared. |
| `viz/test_plot.py` | 28 -> 28 | Matplotlib output, pixel/class palettes, composition, captions, sparse codes and polygons with holes retained. |
| `viz/test_style.py` | 10 -> 10 | Style type, palettes, percentiles, nodata and sparse-code positions retained. |
