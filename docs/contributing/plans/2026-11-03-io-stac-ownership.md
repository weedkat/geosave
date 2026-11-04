# I/O, STAC and geographic anchor migration

Status: implemented in the existing shared checkout; no unrelated changes are
staged, reset or committed. Spec: `../specs/2026-11-03-io-stac-ownership.md`.

## Binding decisions

- GeoAnchor contains CRS-aware geometry and optional time only. Exact grids
  remain native GeoBox values, exposed on raster accessors and passed explicitly.
- `StacQuery.set_anchor` replaces space/time selectors; ModelSpec retains its
  fill-missing search behavior and explicit-ID bypass.
- Selected native Items feed typed standalone Dataset/DataTree loaders.
- DataTree loading preserves the existing relative-resolution behavior. The
  finest family uses the requested grid, and other families use scaled grids.
- Keep provider clients, Parquet search, metadata/property fusion and late signing.
- Preserve ModelSpec configuration and loading behavior with a separate explicit
  grid argument. Flows construct their configured grid independently of anchors.
- Do not edit notebooks, add compatibility aliases, or alter unrelated work.

## Tasks and verification

1. Capture a focused baseline; add geometry/query/accessor contract tests.
   Implement anchors, selection aggregation and `set_anchor` without pixel reads.
2. Extract `io.odc.load_dataset/load_datatree` with typed native load options.
   Check values, exact grids, laziness, property fusion and metadata precedence.
3. Migrate model recipes and flows, separate grid construction, remove StacSource;
   retain portable model configuration and endpoint fallback. Make StacTable an
   independent table client. Test local Parquet discovery and signed loading.
4. Consolidate formats under `io/formats`, storage-independent paths under
   `io/paths`, native asset reads/writes under `io/stac`. Move STAC native-object
   conversion and metadata modules into their owning package. Update callers,
   mirrored tests and maintained documentation.
5. Run focused round trips and consumer checks, lint, then a bounded broad suite.
   Compare failures with baseline and record any incomplete checks accurately.

## Execution record

- Pre-flight: anchor consumers require explicit grid migration before removal of
  the grid attribute. Source removal requires recipe migration and signing first.
- Ruling: preserve DataTree relative-resolution behavior; changing it to absolute
  native resolutions would be an additional behavior change not yet agreed.
- Ruling: work in the existing checkout, whose extensive uncommitted source is
  the reviewed baseline; a fresh HEAD checkout would omit that required state.
- Snapshot: `/tmp/geosave-io-stac-before`; baseline log:
  `/tmp/geosave-refactor-baseline.log`.

- Completed: geometry/time anchors and aggregation, independent selected-Item
  loaders, model/flow grid migration, independent StacTable, canonical format
  modules, STAC metadata ownership, and public guide/API updates.
- Independent review corrected canonical format exports, timeless STAC sentinel
  aggregation, and native ODC option typing. Core imports remain STAC-lazy.
- Final focused verification: 444 passed, 17 deselected, 43 warnings. Covers I/O,
  STAC, anchors, DataTree, model recipes and flows. Local STAC guide code blocks
  also ran successfully against real files. Ruff passed on 86 changed Python
  files; API generation and freshness checks passed.
- Full current suite: 37 failed, 1394 passed, 41 deselected, 3 xfailed,
  118 warnings. See `/tmp/geosave-final-suite.log` for exact failures.
- Persistence checks ran outside the restricted sandbox because sandboxed Zarr
  execution stalled. No remote STAC service availability claim is made.
- Breaking: GeoAnchor grid fields/constructors and StacSource are removed;
  loaders take selected Items; ModelSpec raster loading/ingest requires geobox;
  flow anchor configs return (anchor, geobox); format imports use io.formats;
  rasterize accepts a native grid or raster. No compatibility aliases added.
- Retained limitation: DataTree families come from the first Item and preserve
  relative resolution, not guaranteed original source alignment.
- Saved pre-refactor full suite: 38 failed, 1390 passed, 41 deselected,
  3 xfailed, 115 warnings. All 37 current failing nodes also fail in that
  baseline; a separate direct baseline run reproduced all 37. No new failing
  nodes were found. The baseline additionally failed the GeoTIFF mask warning
  check, which passed in the current run; no claim of fixing that issue is made.
  Logs: `/tmp/geosave-before-full-suite.log` and
  `/tmp/geosave-baseline-failed-nodes.log`.
