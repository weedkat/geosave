# Raster validation and expansion implementation plan

The approved conversation defines `gs.validate` with independent spatial,
georeferenced, affine, and same-grid checks, and `gs.expand_rasters(dim=...)`
with time expansion by default. Expansion returns a DataFrame of lazy native
DataArrays with path, variable, grid, spatial metadata, and selected temporal
coordinates. Explicit empty dimensions retain complete variables. Spatial
axes cannot be expanded. Static variables are not broadcast over observations.
Native spatial names are accepted by inspection and validation.

Implementation stays in the current working tree to preserve the ongoing review
fixes. No existing unrelated edits are discarded or committed.

- [x] Add failing workflow tests for expansion, inherited intervals, additional
  dimensions, static variables, lazy pixels, and selective validation.
- [x] Implement shared accessor methods in raster/base.py using dependency axis
  discovery; keep metadata inspection separate from operation requirements.
- [x] Replace the tree node traversal name `rasters` with `datasets`, and remove
  the old `grids` inventory. Migrate publication planning and consumers.
- [x] Centralize Chips structural checks while retaining unreferenced chipping;
  accept discovered spatial names in chip slicing and grid preservation.
- [x] Update public examples and obsolete tests, run focused raster, publication,
  transform tests and Ruff, then review the final changes.

Review focus: no pixel tasks during inspection/validation; a grid without CRS
is not georeferenced; missing grids cannot pass same_grid; different tree grids
are valid unless same_grid is requested; rotated and GCP grids remain distinct;
empty and invalid inputs produce errors naming their variable/node.

## Verification and decisions

- New tests first failed for missing methods and accessor rejection of native
  names; implementation then passed the focused raster workflow suite.
- Review reproduced irregular, singular, and nonfinite coordinates accepted by
  ODC inference. Regression tests failed first, then passed after shared
  validation checked spacing and finite, nonsingular affine transforms.
- 101 tree/STAC/NetCDF tests passed (12 deselected); 315 transform/vector and
  inspection tests passed, with 3 existing lazy-warp xfails.
- Zarr-inclusive runs stalled and were bounded by timeout; no complete Zarr
  suite claim. Non-Zarr COG/NetCDF round trips passed.
- `datasets` retains whole-node traversal; `expand_rasters` is the variable and
  observation inventory. No compatibility aliases retain the old names.
- Native axes are supported by inspection, validation, chips, NumPy conversion,
  and CRS annotation. Other transform APIs may still require explicit `to_yx`.
- Publication's Dataset branch retains its existing requirements; tree nodes
  explicitly require affine, georeferenced, shared grids within each node.

## Conventions follow-up

The approved follow-up moves spatial discovery to `gs.spatial_dims` and removes
`_spatial_pair`. Trees report a shared pair only when their data-bearing nodes
agree; metadata-only nodes are ignored. `to_yx` owns normalization directly,
including recursive tree mapping; `_rename_yx` is removed. Item ID formatting
is a publication choice: public methods forward `id_format=None` to
`stac.item.create_item`, which owns the unchanged time/centroid fallback.
`ITEM_ID_FORMAT` is removed from conventions. Native/CF axis and Item ID tests
failed first for the missing property/default argument before implementation.

## Concrete accessor ownership follow-up

Approved scope: declare `validate`, `spatial_dims`, and `geobox` on the base
as abstract contracts; implement individual-raster checks on DataArray and
container checks on Dataset/DataTree. Preserve independent boolean options,
contextual errors, native axes, and lazy pixels. Shared derived properties
and coordinate-writing functions retain their existing ownership.

- [x] Implement concrete discovery, grid lookup, and validation with focused
  behavior tests, including container disagreement and missing grids.
- [x] Make viewport reading consume discovered axes, preserving nonspatial
  coordinates for intersecting and empty windows. Convert the selected tree's
  raster contents without interpreting organizational depth as Collections.
- [x] Resolve publication Collection selection in I/O once and reuse it for
  metadata and path planning; preserve selection checks and write preflight.
- [x] Review integrated changes, regenerate API docs, and run focused raster,
  publication, and transform checks plus lint and type checks.

Work remains in the existing dirty workspace without commits or unrelated
cleanup. The public signatures stay unchanged; nested tree conversion now
accepts organizational groups, and Dataset spatial properties describe
agreement across variables rather than whichever axes ODC discovers first.

Review verified concrete accessor ownership and lazy metadata inspection.
Two integration findings were fixed with regression tests: viewport reading
requires common spatial dimension names through `validate(same_dims=True)`,
and explicitly supplied Collection metadata takes precedence over carried
tree metadata. Tree conversion now rejects empty or unreferenced raster
subtrees consistently, including a single data-bearing node.

Ruff, API reference freshness, and diff checks pass. Type checking passes for
the raster accessors and path planning; `io/stac.py:207` retains an unrelated
reader error where `open_raster` can return Dataset or DataTree but the next
operation requires Dataset. Zarr-related tests remain excluded because the
environment's persistence runs stall; no full-suite or Zarr claim is made.
Final integrated raster, path/publication, GeoTIFF, and transform checks:
539 passed, 15 deselected, 3 existing lazy-warp xfailed, 48 warnings.

## Dimension-driven expansion schema

The approved revision removes the universal `RASTER_COLUMNS` and DataArray's
`_describe_raster`. DataArray rows expose requested dimension labels (positions
when unlabeled), the native raster, and only applicable scalar temporal metadata.
Dataset adds variable names; DataTree adds node paths. Spatial metadata remains
on the raster. Empty results preserve their applicable columns. Reserved output
name collisions raise before construction rather than overwriting identities.
Regression coverage includes expansion labels, empty schemas, scalar vs retained
time, static variables, and identity collisions; guides and API docs are updated.

## Readable validation and actionable errors

The approved iteration removes `_validate_grid` and nested validator calls.
DataArray validates identified spatial axes once; if a grid check is requested,
it discovers placement and validates affine placement once, then checks CRS.
Dataset validates variables and compares them; DataTree validates datasets and
compares nodes. Helpers return no metadata. `axes` uses an explicit loop.

`RasterValidationError(ValueError)` carries the problem, variable, node path,
and optional operation context. Only this error is caught to add context;
unrelated TypeError and ValueError exceptions are not relabeled. Messages
suggest corrections without referring users to internal boolean flags.
No result caching or boolean return was introduced.

Verification: 538 passed, 15 deselected, 3 existing lazy-warp xfailed, 47 warnings
across raster, path/STAC publication, and transform tests. Zarr-related tests
remain excluded. Ruff, formatting, affected-module type checking, API reference
freshness, and diff checks pass. Review also corrected mixed placed/unplaced
comparison messages to suggest known placement before alignment.

## Core raster representation conversions

Approved: Dataset `to_dataarray` owns lazy variable stacking (casts before
combining); DataArray `to_dataset` wraps one named variable unless a dimension
is explicitly split. Remove obsolete `to_array`, `to_raster`, and raster
`to_datatree` methods without aliases. Native DataTree construction owns explicit
paths and preserves node metadata. Catalog `to_datatree` remains an asset reader.
Tree `to_dataset` merges compatible grids without implicit alignment or resampling.
Keep the current `align` API for explicit calls, and record its parameter design
as an open decision rather than redesigning it in this change.

Work is split into array conversion, tree conversion, documentation, and caller
migration. Preserve existing dirty edits and ModelSpec publication layout.

Implemented the conversion contracts and migrated callers and documentation.
Model preparation and catalog display retain their existing finest nested-grid
default through shared grid selection; other layouts require an explicit target.
Smoke coverage checks lazy display alignment and rejects shifted layouts.

Verification: raster, publication, transforms, and summary tests report 550 passed,
15 deselected, and 3 existing xfailed; model processing separately reports 9 passed.
Ruff, affected-module type checking, formatting, API generation, and diff checks
pass. Combined runs timed out; verbose viewer isolation last reached
`test_checking_a_raster_row_adds_its_layer` without completing. Zarr remains excluded. A first smoke fixture
also exposed singleton spatial axes being squeezed by the existing asset reader;
that separate reader issue remains unresolved.

## Core supporting modules

Moved grid placement, expansion dimension selection, raster summaries, and tensor
conversion to `core/utils/{grid,expansion,summary,tensor}.py`. Mirrored tests move
with their modules. Accessors and Chips remain under `core/raster`; general
helpers remain under `geodata/utils`. RasterValidationError joins the existing
exceptions in `geodata/errors.py`; public geodata imports remain available.
Updated imports, architecture guidance, and generated API references without
compatibility modules or behavioral changes.

Verification: 273 passed and 12 persistence tests deselected across core helpers,
raster accessors, and path handling. Public import smoke confirms geodata still
does not import torch. Summary type checking exposes an existing possibly-unbound
`fill` in `class_percent`; the labels depend on the final band's nodata value.
That aggregation policy needs a separate behavior review; summary implementation
is unchanged by this relocation.

## Grid validation cleanup

Grid placement delegates spatial structure checks to `.gs.validate(same_dims=True)`
and uses `.gs.spatial_dims` instead of requiring trailing y/x axes. Recognized
spatial names and axis order are preserved. CRS writing validates the resulting
placement once; the private CF metadata writer does not rediscover or revalidate
it. Supplied-grid CRS, shape, naming, and selection failures use
RasterValidationError; these checks concern GeoBox arguments rather than existing
raster structure. The standalone `write_cf_axes` helper is now private.

Verification: 229 passed, 12 persistence tests deselected; Ruff, affected-module
type checks, API generation, and diff checks pass. Coverage includes lazy
latitude/longitude placement in either container, reversed dimension order,
invalid Dataset variable context, GCP placement, and NetCDF round trips.

## Grid naming ownership and vector follow-up

Moved `name_grid`, `name_grids`, and unit abbreviations to `io/paths.py`.
Grid-derived node names are a loader/publication convention, not a DataTree
structural requirement. Renamed `_select_finest_grid` to `select_nested_grid`
in core utilities and updated model preparation and visualization callers.
No compatibility aliases or layout changes were introduced.

Verification: 85 passed, 4 deselected across grid, paths, STAC asset loading,
ODC loading, and model processing. Ruff, affected-module typing, API generation,
and diff checks pass.

Next review: vector accessors and their sampling, metadata, I/O, and visualization
callers. Apply the same principles: trace ownership before moving code; centralize
structural validation with actionable errors; separate checks from discovery;
keep variant-specific behavior on its owner; simplify obsolete mechanisms instead
of rearranging them; preserve native values and lazy work; use explicit control
flow and clear inputs/outputs. Review utility ownership by domain responsibility,
not caller count, and test public workflows rather than private helper structure.

Prior review cross-check for vector work (`review-2026-11-04.md`):

- #24 remains partly open: sampling returns only row positions, and sample
  footprints recompute intersection or the smallest selected Item. The founding
  Region is not retained. Chips now takes prepared Datasets, so do not revive
  its old implicit union/reprojection behavior to address this handoff.
- #25 is resolved in live sampling via `read_instant`, with interval and timeless
  regression coverage. Preserve that policy when consolidating temporal code;
  nearest-date pairing and interval-overlap filtering are distinct operations.
- #15 is resolved: search reads Collection metadata once. Preserve the ownership
  split between STAC metadata interpretation, I/O asset reading, and search.
- #21 was withdrawn: catalog planning and GeoSamples membership are intentional
  responsibilities. File length alone does not justify moving sampling.
- New inspection targets: scattered CRS/geometry/Item checks and temporal-row
  extraction across catalog anchors, query filtering, sampling, and row access.
  Separate pandas accessor eligibility (AttributeError) from operation validation.

Vector smoke initially found one stale mixed-grid conversion assertion after the
raster API change. Updated that assertion and the row-reader example to align
explicitly; no vector runtime behavior was changed in this cross-check.

## Vector step 1: structural validation and sampling arguments

Implemented frame `validate(geometry=True, crs=False, stac_item=None)` and record
`validate(geometry=True, stac_item=None)`. Checks return None, do not repair data or
open assets, and raise `VectorValidationError(ValueError)`. Item checks explicitly
cover required columns, not complete STAC schema validity. Series do not inherit
or claim a parent CRS. Accessor eligibility still raises AttributeError.

Geometry validation is shared by record/frame accessors. CRS, footprint, spatial
query, Item-table conversion, and rasterization use the requested structural
checks. Removed rasterization's duplicate validator that returned its input.
Catalog anchors use the validated footprint union, avoiding a repeated frame
scan while preserving GeoRegion construction invariants.

Sampling rejects empty/string/duplicate pair names, unsupported keep values,
negative or missing tolerances, and noninteger window/step values before planning.
Zero-area errors identify row positions and include Item IDs when available;
ordinary features need no ID column. Area remains a sampling-specific check.
Temporal interpretation, sampling decomposition, Region retention, and asset
loading ownership remain separate follow-up steps.

Verification: new regressions initially failed (9 cases). Final focused suite:
147 passed, 2 deselected across vector accessors, anchors, rasterization, STAC
conversion, and GeoParquet. Affected-module typing, Ruff, API generation, and diff
checks pass. Scoped review found no blocking issues; its duplicate anchor-scan
finding was addressed and the suite rerun.
