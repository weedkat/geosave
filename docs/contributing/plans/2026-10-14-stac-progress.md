# STAC translation implementation ledger

Plan: 2026-10-14-stac-translation.md. Implementation authorized.

Ruling: use the existing checkout on refactor/stac-translation, preserving its dirty snapshot in /tmp/stac-baseline. A clean worktree would omit the implementation being refined. No staging or commits.
Ruling: stacks continues accepting one item table; GeoCatalog.concat merges Collection definitions before stacking. Keep multiple collections supported; publish one per file as a convention, not a restriction.
Ruling: store JSON Collection definitions in DataFrame.attrs['stac:collections']; expose native copies through gs.collections. Explicit writer definitions replace attached definitions by id. Missing referenced definitions are errors; unused definitions are ignored at publication.
Pre-flight: Scene.coords feeds Item properties and dates; band translation feeds loading; table conversion feeds catalog/search/I/O. Collection metadata belongs to the catalog, not individual rows or raster attrs.

Pending: temporal/property contracts, Scene coordinate preservation, schema ownership, native asset loading, table/I/O separation, Collection round trips and concat, documentation and review.

First regression gate: six publication tests failed before implementation and passed after changes (Collection ownership/concat/missing definitions, immediate/deferred property propagation, native JSON overwrite).
Ruling: retain single-store Dataset.to_netcdf/to_zarr return types; explicit per-date publication selects each time before writing. Their existing layout helper supports split templates internally, but widening the public return type is outside this refactor.
Ruling: JSON overwrite replaces reachable documents without recursive cleanup. Old unlinked JSON may remain, but unrelated assets cannot be deleted and old Items are no longer reachable from the new root.

Implemented: exact-name property config; Scene coords through immediate/deferred/per-band layout; EO/core scalar publication; nominal time plus explicit TimeSpec bounds; multidate observation rejection; footprint-based identity; native attached-Asset schema writes; ordered core/extension band translation; direct loaded-variable metadata; native file-backed sample loading; shared assembly of opened rasters; table conversion relocation; native JSON writer and shared duplicate validation; Collection metadata ownership/read/write/concat; GeoCatalog public export.

Verification: initial broad run 728 passed / 13 failed; failures exposed old tests for deliberately changed APIs plus time precision and Scene comparison issues. Ordinary instants now stay instants unless explicit TimeSpec bounds exist. Existing Scene comparison excludes xarray coords because mapping equality is array-valued. Focused rerun 199 passed / 1 failed; the remaining unused-definition concat conflict reproduced and fixed. Contract + sample gate: 31 passed.
Ruling: concat merges only definitions referenced by each input Item table. Filtering retains attrs, so unused definitions must not block combining new observations. Writer overrides remain explicit by id.
Collection link regression found clear_links discarded license links; retain nonstructural links and resolve Collection asset hrefs before dropping structural links.
Remaining: full verification, fresh review, documentation freshness and type-check baseline comparison.

Final review: fresh agent reviewed /tmp/stac-review.diff against the pre-change snapshot. Fixed every reported issue: unrelated JSON destination collisions (two failing tests before preflight, then passing), Item relative descriptive links, equal temporal endpoints, and composite bounds through native-file loading and StacSource. The latter uses exact start_datetime/end_datetime coordinates and ODC's own property grouping; instants contribute their endpoints when mixed with composites. Source property smoke also caught ODC's default name sanitization; native name=key now prevents it without aliases or post-load renaming.

Final checks:
- Relevant regression suite (STAC, raster/vector I/O, core, cuts, catalog raster layers, root imports): 795 passed, 2 integration tests deselected.
- Final catalog/publication/table checks after the concat Item-versus-sampling invariant: 51 passed.
- Explicit upstream schema integration checks: 2 passed.
- Scoped Ruff: all checks passed; 47 touched Python files formatted.
- STAC package BasedPyright: 0 errors, 0 warnings.
- Broader touched raster/core/I/O scope: 8 existing findings (Location/PathLike in catalog, Affine typing in assembly, Dask Delayed typing in file, optional layer key in layout). The corresponding findings exist in the pre-change source snapshot; no repository-wide type-check success is claimed.
- API reference freshness and git diff --check passed.
- Full docs build blocked by the existing manual reference to removed geosave.geodata.cuts.GeoDataTrees in docs/reference/cuts.md; Zensical panicked, and the stalled process was stopped.
- Broader explorer UI tests had 5 failures in app.py/docks.py; comparison showed independent edits arriving in those files after the initial snapshot. Their baseline tests passed. Those edits were preserved; catalog-layer tests passed.

Tasks 0–6 implemented and verified; Task 7 documentation and relevant checks completed with the unrelated limitations above. No pending STAC implementation work, no staging/commits, no external publication.

Breaking changes: missing referenced Collection definitions now refuse publication; exact property keys only; multidate stores cannot become one observation Item; pure table conversions moved to translate.table.from_items/to_items; read_collections and resolve_asset_hrefs replace old format helper names; stac_json.write takes native Items; fabricated inverse Scene/RasterFile APIs removed. JSON overwrite retains old unlinked documents but refuses unrelated destination collisions. Full multidate cube loading/publishing remains outside scope.
