# Native STAC Publication Implementation Plan

The core implementation and Python caller migration are implemented. The final
regression run reports 1,623 passed, 1 failed, 42 deselected and 3 xfailed.
The remaining failure executes notebook code that combines different observation
coordinates; notebooks were excluded from edits by the repository instructions.

**Goal:** Publish xarray observations as saved raster assets and native PySTAC
Items, then represent them in GeoDataFrames and compliant STAC GeoParquet files.

**Architecture:** Native tree traversal, dimension slicing, variable partitioning
and name formatting are separate operations. Concrete raster accessors compose
them with persistence and standalone Item construction. Catalog conversion and
file serialization consume native Items/Collections without raster dependencies.

**Tech Stack:** Existing xarray, Dask, rasterio, PySTAC, stac-geoparquet, PyArrow,
GeoPandas and storage dependencies; no new dependency proposed.

**Spec:** [Native STAC publication](../specs/2026-11-07-stac-publication.md).

## Global constraints

- COG-only asset publication is confirmed. No Zarr asset implementation.
- Inspect existing functionality before adding a helper. Reuse/extend concrete
  `gs.validate` for xarray structural checks. Publication-specific ID/path checks
  and Collection comparisons stay with their owning operations.
- Use explicit loops, readable functions and blank lines between meaningful
  steps. Standalone functions mean clear inputs and responsibilities, not a new
  wrapper for every operation or a parallel validation system.
- Native xarray, PySTAC and GeoDataFrame public values; no throwaway containers.
- Names: `format_id`, `format_paths`, `create_stac_item`, `create_catalog`.
- Explicit `id` overrides the ID template for one Item; one output defaults to
  asset key `asset`. Multi-output identity is a separate stage-2 decision.
- `to_item` saves assets and returns one Item; `to_catalog` reuses publication.
- No DataTree slicing wrapper; use `subtree_with_keys` or existing `gs.datasets`.
- Keep variant-specific behavior in concrete accessors. Shared code must implement
  genuinely identical behavior, not dispatch by the wrapped xarray type.
- STAC extension imports remain in `stac/extensions/`.
- Preserve unrelated dirty changes. Stage only intentional files/hunks if committing.
- No compatibility aliases; migrate callers and remove each superseded path together.
- Planning and inspection remain lazy; do not compute raster pixels for metadata.

## Review focus

1. Auxiliary observation coordinates and CF encoding surviving slicing: stage 1.
2. Naming collisions in later observations detected before any write: stage 2/6.
3. Packed storage metadata and different grids correctly described: stage 3.
4. Collection conflicts and relative href bases surviving round trips: stage 4/5.
5. Reject incompatible observation coordinates within a selected tree root;
   preserve different spatial grids: stage 6.

## Order and gates

| Stage | Independent deliverable | Depends on | Gate before implementation |
| --- | --- | --- | --- |
| 0 | Old raster publication removed; dependent callers tracked | Live caller review | Applied; restore dependents during implementation |
| 1 | Lazy labeled slices | Existing xarray accessors | Approved slice rules |
| 2 | Names and output selection | 1 | Approved naming defaults and actual collision checks |
| 3 | Items from existing raster files | Existing storage/rasterio | Final constructor interface |
| 4 | Native Items/Collections to catalog | Native PySTAC; no raster stage required | Implement confirmed comparison policy |
| 5 | Compliant table file round trip | 4 | Confirm target format versions and schema assertions |
| 6 | Raster publication accessors | 1–5 | Shared observation-coordinate validation and caller integration |
| 7 | Integrated GeoCatalog and cleanup | 6 | All scoped workflow checks pass |

Execute sequentially for review clarity. Stages 3 and 4 are architecturally
independent of slicing; their tests must remain usable without raster accessors.
Collection comparison policy is settled; it does not introduce a dependency for slicing.

## Stage 0: Incompatible implementation cleanup

The user approved deletion before replacement, and the raster publication
implementation has been removed. The inventory distinguishes that completed
cleanup from shared functionality retained for independent callers.

| Existing implementation | Disposition proposed for discussion | Active callers/behavior affected |
| --- | --- | --- |
| `io/paths.py::expand_paths`, `_expand_paths` | Remove mixed traversal/selection/naming implementation once cleanup sequence is agreed | DataArray, Dataset and DataTree raster writing; STAC publication |
| `io/paths.py::select_collections` | Replace path-implies-Collection policy; retain only explicitly agreed selection semantics | `io/stac.py::write_assets`, expansion |
| `stac/item.py::create_item` | Remove xarray-based storage-metadata builder | `io/stac.py::write_assets` |
| `stac/asset.py::build_asset` | Replace xarray-dependent asset construction; retain useful extension operations after inspecting their other callers | Existing Item builder |
| `io/stac.py::write_assets`, `group_files_by_date` | Replace pre-write Item assembly, date-only grouping and `image` asset fallback | All raster `to_catalog` accessors |
| `stac/catalog.py::from_items` | Rename to `create_catalog`; retain proven codec behavior | Search API/table, ModelSpec, publication, exports and tests |
| `io/formats/stac_geoparquet.py` accessor callbacks | Remove callbacks into `.gs`; retain writer functionality | STAC GeoParquet serialization |
| `core/vector/catalog.py::concat` first-definition-wins block | Replace only the Collection metadata policy, not generic frame concatenation | Catalog concatenation |

Do not delete the unrelated sampling `select_collections` merely because its name
matches the path helper. Preserve native traversal, storage normalization,
format writers, useful extension helpers and semantic round-trip tests. Remove
obsolete implementation-coupled tests only with an explicit inventory of which
behavioral protections survive. Do not blanket-delete STAC modules or test files.

- [x] User explicitly chose deletion before replacement, accepting the removal
  of the old raster publication path.
- [x] Inspect the old publication callers and remove the approved implementation.
- [x] Remove publication-only tests and exports; retain independent reader/writer
  behavior. Record dependent tests needing migration; add no stubs or aliases.

Remaining interface decisions are tracked in the spec. Do not reopen confirmed
slicing, temporal, tree or Collection rules unless new evidence exposes a conflict.

## Stage 1: Native grouping and metadata feasibility

**Inspect:** concrete raster accessors, existing `expand_rasters`, `gs.validate`,
`io/formats/geotiff.py` and their tests. No public `iter_slices` or duplicated
coordinate-label dictionary is planned.

**Interfaces:** use native Dataset/DataArray GroupBy and native DataTree traversal.
Publication will consume native raster objects whose coordinate metadata remains
attached. Keep grouping implementation inside its concrete owner; extract a helper
only when actual callers need the same operation.

- [ ] Review existing `expand_rasters` callers and identify obsolete behavior;
  do not add another expansion interface alongside it.
- [ ] Verify unique time and time/member grouping, singleton coordinate retention,
  ordering, encoding and zero pixel computation using representative rasters.
- [ ] Preserve Dataset-level observation coordinates during variable partitioning.
  Resolve the demonstrated `time_bounds(edge)` COG writer failure without losing
  the interval used by Item metadata. Extend existing writer/metadata owners.
- [ ] Validate COG output shape for multi-entry groups. Do not silently squeeze,
  drop, mosaic or average duplicate entries, or reject duplicates globally.
- [ ] Inspect/extend existing `gs.validate` for shared observation coordinates
  within selected tree roots while permitting different spatial grids.
- [ ] Exercise actual COG writes and read-back values, transforms and observation
  metadata before downstream publication relies on this stage.
- [ ] Run focused existing and new tests. Report independent writer failures
  separately from tests still depending on the intentionally removed publication.

**Known evidence:** native grouping passed lazy/grid/encoding checks and plain COG
persistence. Grouped temporal bounds failed in the current writer; extracting a
DataArray dropped those bounds. This stage is not complete until that integration
is resolved and checked.

## Stage 2: Selection orchestration and naming

**Files:** `io/paths.py`, `utils/template.py`, concrete raster persistence callers;
`tests/geodata/io/test_paths.py`, `tests/geodata/io/formats/test_geotiff_files.py`.
Paths here and below are relative to `src/geosave/geodata/` unless under `tests/`.

**Consumes:** native raster groups with their coordinates, variable names and
relative node paths. No duplicate label dictionary is part of the grouping interface.
**Produces:** relative names plus selections, then resolved output locations for
existing writers. Return plain tuples/mappings; do not introduce an output-plan class.

- [ ] Settle a naming table for one file (`asset`), variables, multi-node files,
  band-bearing Dataset variables, Item defaults and explicit ID overrides.
- [ ] Specify the small formatting interface and accessor-owned orchestration
  before writing code. Formatting must consume labels, not xarray objects.
- [ ] Preserve observation coordinates during variable partitioning; plain
  `dataset[[name]]` can drop unrelated coordinates such as time bounds.
- [ ] Use explicit split configuration in publication/persistence orchestration;
  templates only render names. Remove xarray type dispatch, Collection selection
  and attrs policy from formatting. Test rendered collisions even when all
  expected fields are present, including two times rendered with `{time:%Y}`.
- [ ] Test unknown template fields, unsafe relative paths, collisions caused by
  truncated dates, unchanged encoding, and zero pixel computation.
- [ ] Preserve whole-batch destination preflight. Migrate actual non-STAC writer
  callers in the same change and remove their superseded expansion path.

## Stage 3: Standalone raster Item construction

**Files:** `stac/item.py`, `stac/asset.py`, `stac/band.py`, `stac/extensions/`,
`stac/__init__.py`; matching `tests/geodata/stac/` files.

**Interface:** `create_stac_item(sources, *, id, ...) -> pystac.Item`, with the
proposed full signature and parameter meanings in the spec. No xarray inputs.

- [ ] Settle time requirements and the exact extraction flags before implementation.
- [ ] Build tiny actual COGs, including different grids and uint16 packing with
  scale/nodata. Assert Asset metadata matches the files rather than source xarray.
- [ ] Implement readable inspection, asset construction and Item assembly steps.
  Use `item.add_asset`; allow native customization afterward. No public asset factory.
- [ ] Test absolute advertised hrefs, explicit footprint, intervals and multiple
  assets. Confirm header-only extraction does not invoke pixel statistics.
- [ ] Validate complete core/extension schemas. Record unavailable network schema
  checks as unavailable, not passing. Retain the old builder only while callers
  still require it; it is not a public compatibility alias.

## Stage 4: Collections and in-memory catalog

**Files:** `stac/catalog.py`, public export modules, direct callers;
`tests/geodata/stac/test_catalog.py`.

**Interfaces:** `create_catalog(items, *, collections=()) -> GeoDataFrame` and
existing `to_items(frame) -> list[pystac.Item]`. Native Collections are optional;
no new Collection constructor or equality framework.

- [ ] Compare same-ID Collection definitions excluding only the five approved
  navigation relations. Reject differing extents and other metadata with a useful
  field diff; preserve distinct IDs and do not mutate input Collections.
- [ ] Test equal definitions, contradictory licenses, missing definitions,
  extent-only differences and caller-object immutability under the approved rule.
- [ ] Rename `create_catalog` and migrate its callers together; remove the old name.
- [ ] Exercise Items → Arrow → GeoDataFrame → Items, including empty data,
  heterogeneous asset keys, timestamps, intervals, property collisions and IDs.
- [ ] Restore available Collection associations without fetching unresolved links.

## Stage 5: STAC GeoParquet persistence

**Files:** `io/formats/stac_geoparquet.py`, `io/formats/geoparquet.py`;
`tests/geodata/io/formats/test_stac_geoparquet.py` and `test_geoparquet.py`.

- [ ] Pin target STAC GeoParquet and GeoParquet versions separately; verify the
  existing dependency can emit the required schema before changing defaults.
- [ ] Remove `.gs` callbacks from writers; call standalone conversion functions.
- [ ] Round-trip actual files and assert nested fields, timestamps, bbox, geometry,
  separate metadata footers, Collection definitions and restored absolute hrefs.
- [ ] Test relocated files with relative asset references, empty catalogs, projected
  analysis frames and failed serialization without mutating input Items/frames.

## Stage 6: Raster publication

**Files:** concrete `core/raster/{dataarray,dataset,datatree}.py`, `io/stac.py`,
`attrs/groups/stac.py` only if required; corresponding accessor tests and
`tests/geodata/io/test_stac_writes.py`.

- [ ] Inspect existing `gs.validate` and extend it for shared observation
  coordinates within the selected tree root. Reject mixed/mismatched temporal
  structures before writing; allow different spatial grids. Implement no time
  union, nearest-time matching or static-asset replication.
- [ ] Implement `to_item` on concrete accessors: select → name → preflight → save
  → inspect → enrich → return native Item. Never build storage metadata first.
- [ ] Carry xarray Collection metadata or an explicit native Collection, with
  explicit input taking precedence and no mutation of input definitions.
- [ ] Make `to_catalog` partition observations and use the same Item publication
  path, then `create_catalog`. Preflight the full batch before the first write.
- [ ] Test missing time, duplicate IDs, a later destination collision, mixed grids,
  encoded persistence, metadata retention, and approved tree grouping behavior.
- [ ] Remove the xarray-coupled old Item builder when its last caller migrates.

## Stage 7: GeoCatalog integration and handoff

**Files:** `core/vector/catalog.py`, `core/vector/item.py`, `stac/search/table.py`,
their matching tests, public exports, relevant guides and generated references.

- [ ] Use the Collection reconciliation operation for catalog concat; preserve
  footer metadata deliberately rather than taking the first definition silently.
- [ ] Check selected/renamed geometry and CRS context in row-to-Item conversion.
- [ ] Run connected publication/search/read workflows and relevant sampling
  regressions. Do not broaden into a sampling redesign.
- [ ] Search all Python/docs/templates callers for removed API names, update
  examples, regenerate references, and delete remaining obsolete paths.
- [ ] Run focused tests, Ruff and applicable typing/docs checks. Report exact
  failures and environmental limitations; never infer full-suite success.

## Evidence available before implementation

Native DataTree traversal and lazy slicing have been smoke-tested in this
conversation. Item/Arrow/GeoDataFrame and temporary Parquet feasibility smokes
passed, with explicit limitations recorded in the spec. None of these smokes
implements or validates the proposed new public APIs.

## Applied cleanup and current limitations (2026-11-07)

Removed the three raster `to_catalog` methods; `io/stac.py::write_assets` and
`group_files_by_date`; `stac/item.py::create_item` and construction-only helpers;
`stac/asset.py`; and `band.write_band_fields`. Removed the `create_item` export,
unused imports and publication-only tests in `test_stac_writes.py`, `test_asset.py`
and direct old-builder cases in `test_item.py`. Native reading, extension modules,
Item-table conversion and independent raster writing remain.

No replacement or compatibility stub was added. The common path expander remains
because independent format writers still use it. ModelSpec's publication callers,
other publication integration tests and `tests/geodata/conftest.py::create_file_items`
still require the removed path. Keep their relevant behavioral expectations visible
until publication and test fixture setup are redesigned; do not mistake this
transitional state for passing repository-wide checks.

Validation: Ruff check passed on touched Python files. Native import and unlabeled
xarray positional-selection smoke passed. Focused tests: 28 passed, 2 failed,
9 deselected, 1 dependency warning. Both failures are the path test calling removed
DataTree `to_catalog` (scalar/missing-CRS variants). No full-suite claim.

Confirmed naming distinction: `{var}` is a Dataset variable, `{band}` is an array
band coordinate. Collection metadata conflicts are rejected with differing fields;
automatic extent merge is withdrawn. Multiple Collection IDs may coexist; only
conflicting definitions of the same ID are rejected. The approved slice/time/tree
rules in the latest confirmation below supersede earlier pending questions.

The earlier same-date restriction is withdrawn: STAC supports asset-specific
temporal properties. Separate Item temporal coverage from catalog partitioning.
Unselected dimensions must remain untouched; no positional coordinate labels are
to be synthesized without an agreed rule.

## Current publication contract and implementation discipline

- Default `split_item=("time",)`; no time dimension anywhere means one Item.
  `split_item=()` explicitly selects one Item.
- Default Dataset `split_vars=True`; each variable retains its internal bands.
  A single output is `asset`; multiple outputs use variable names, qualified by
  relative node paths for DataTree outputs.
- `format_paths` is `str | None`; templates never change partitioning. Check actual
  rendered paths/IDs across the whole batch, including date-precision collisions.
- Selected DataTree roots describe the same observations across different grids.
  Use native traversal and `gs.validate`; do not reconcile mismatched times or
  replicate static nodes. No per-node path-template mapping is needed.
- Slicing omits absent coordinate labels; templates requiring them fail before
  writing. Never invent observation timestamps.
- Item time is the common instant or combined coverage interval, with explicit
  input taking precedence; require input if no usable time exists.
- Saved-file storage metadata is authoritative. Keep differing semantic values
  on assets, promote shared values to the Item, and apply explicit semantic input.
- Different Collection IDs coexist. Compare same-ID definitions excluding only
  self/root/parent/child/item links; reject other differences, including extents.

Before each stage, verify its prerequisite contracts against the current spec.
After each stage, run its smoke and behavior checks and inspect affected callers.
If an implementation reveals an unresolved semantic choice, report the concrete
input, alternatives and downstream effect, then wait for the user's decision on
that choice. Continue only independent work. Do not encode a guessed policy and
let subsequent stages inherit it. No completion claim while known failures remain.

The source review found `io/odc.py::load_datatree` feeds the same matched Items and
grouping settings to its grid-family loads. Native xarray can represent more
general trees, but publication validates the supported selected-root structure
instead of introducing handlers for unrelated observation series.

## Approved final defaults

DataArray uses `split_asset=("band",)` by default. Dataset uses `split_vars=True`
and retains bands within each variable. Item partitioning defaults to
`split_item=("time",)` with the agreed no-time fallback. Default Item IDs derive
from observation time/interval plus other split-coordinate values. Default asset
paths are `{id}/{asset}.tif`. Validate all names before writing; never silently add
counters to hide collisions. These defaults supersede earlier open D2/D3 entries.


## Verification, 2026-11-07

- `create_catalog` accepts one native Item or a sequence; both forms preserve
  native properties and asset descriptions in the behavioral tests.
- Shared grouping, CF bounds, native Item publication, and temporal seams:
  88 passed in the focused run.
- Saved catalog Items pass PySTAC core and extension schema validation when
  their Collections have real saved JSON document addresses.
- Final full suite: **1,627 passed, 1 failed, 42 deselected, 3 xfailed**.
  Failure: `tests/examples/test_notebooks.py::test_catalog_examples_load_chips_and_restore_pixels`.
  The notebook combines dated image nodes with a differently shaped label time
  coordinate. It requires migration to the agreed observation contract; the
  library deliberately does not broadcast or infer alignment for that input.
- Ruff and `git diff --check` pass. Scoped typechecking of the changed STAC,
  publication, raster-accessor and ModelSpec modules reports zero errors.
  A wider raster-package typecheck separately reports a possibly unbound `fill`
  in `core/raster/utils/statistics.py`, outside this change.
- No commit was made: the shared working tree contains substantial pre-existing
  changes, including changes within the touched files.


## Expansion inventory refactor

- Removed all concrete `_raster_groups` methods and the GeoTIFF `write_groups`
  path. Publication and direct GeoTIFF writing now consume native raster
  inventories indexed by observation.
- Concrete `expand_rasters` methods own slicing, variable separation, and tree
  traversal. Internal `expand_assets` composes observation expansion before
  asset expansion; it does not name outputs or convert arrays.
- Naming retains variable, band, and node-path labels independently. GeoTIFF
  preparation owns conversion to Dataset bands. Shared and inherited metadata
  survive expansion without changing source objects or computing pixels.
- GeoTIFF metadata writes accept NumPy values. Nodata and packing remain GDAL
  band properties rather than duplicate dataset JSON tags.
- Focused expansion/publication checks: 52 passed. Raster/STAC persistence:
  217 passed, 6 deselected. Follow-up metadata/catalog/GeoTIFF checks:
  101 passed, 2 deselected. The final full-suite result is recorded above.
- Scoped type checking: zero errors. Ruff and `git diff --check` pass.
- `read_pixels` and Explorer initialization are unchanged and remain separate
  discussion topics.
