# Native operations and accessor-owned geospatial transforms

Internal module layout is superseded by [raster transform ownership](2026-11-10-raster-transform-structure.md). Historical verification below describes the earlier refactor.

```python
decoded = observations.gs.mask_and_scale()
monthly = decoded.gs.composite("MS", "mean")
selected = monthly.gs.select_time("2024-01")

# Ordinary array operations remain native and explicit.
series = xr.concat(
    decoded_rasters,
    dim="time",
    join="exact",
    combine_attrs=attrs.combine_attrs,
)
filled = monthly.chunk(time=-1).interpolate_na(
    dim="time", method="nearest", fill_value="extrapolate", max_gap="45D"
)

# Existing conveniences keep their accessor interface.
chip = scene.gs.crop_to_grid(grid)
warped = scene.gs.reproject(grid, resampling="bilinear")
colors = labels.gs.colorize()
```

Status: implemented, 2026-11-10; integrated verification is recorded below.
`composite` and `select_time` are now concrete accessor interfaces.

## Goal and scope

Use xarray and ODC directly for their native behavior. Retain GeoSave operations
where they own a geospatial invariant or a useful domain result. Give each
retained operation one public accessor interface and one implementation.
Remove the public `geodata.transform` package after its consumers migrate.

Success is fewer concepts and execution paths, not merely fewer files or lines.
Do not copy the transform package into accessor utilities, add a generic
dispatcher, or replace it with another public operations facade.

Scope includes attrs combination, CF representation, raster/vector accessors,
the transform callers in I/O and visualization, tests, active documentation and
the known GEDI Python consumer. ModelSpec execution, training architecture,
sampling membership, new storage mechanisms and new dependencies are excluded.
Do not inspect or edit notebooks as part of this work. Report their unverified
migration status when removing old imports; do not claim all consumers migrated.

The checkout contains extensive existing uncommitted changes. Work from its
live source, preserve those changes, and keep a scoped execution record. A clean
checkout of HEAD would not contain the implementation reviewed here. Do not
reset, stash or rewrite unrelated work to make the baseline easier.

## Decisions and proposed defaults

### Confirmed with the user

1. Ordinary array operations use native xarray; geospatial conveniences belong
   on `.gs`. Moving every wrapper without deleting behavior is insufficient.
2. Composite retains observed `start_datetime` and `end_datetime` coordinates.
   No custom `.gs.resample()` object or parallel reducer hierarchy is introduced.
3. Packed inputs must be explicitly decoded with `.gs.mask_and_scale()` before
   `.gs.composite()`. It must not silently decode them.
4. Retained GeoSave operations automatically warn when combining attrs drops
   keys, including when the caller supplies a native policy string. The selected
   native policy determines the returned attrs; GeoSave adds diagnostics.
5. This turn authorizes a plan, not production implementation.

### Proposed implementation defaults

- Preserve the existing `drop_conflicts` default at retained GeoSave combination
  sites unless a phase explicitly changes it. This differs from native concat
  and merge's `override` default; document it rather than calling it native.
- Direct `xr.concat`/`xr.merge` retain native defaults and diagnostics. Examples
  pass `attrs.combine_attrs` when GeoSave warnings are desired. There is no
  monkeypatch or global `keep_attrs` setting.
- `attrs.combine_attrs` remains a native policy adapter with dropped-key
  warnings, not a validator or group merger. Keep its dependency on xarray's
  internal `merge_attrs` isolated; do not reimplement the policy engine.
- Retain metadata groups, group replacement and concrete dtype checks. Do not
  delete `Quantity` or other exported groups merely because internal consumers
  are absent; that is a separate public-interface decision.
- Native NetCDF/Zarr encoding remains native. Explicit `scale_and_fill` remains
  a checked conversion, and GeoTIFF retains its existing use of it. Universal
  checked writers are a separate change, not a prerequisite for this refactor.
- Preserve display/identify values in stored units where they currently use
  missing-value masking without scaling. Changing display units is out of scope.
- Preserve existing declared-extent semantics: ordinary crop/reprojection keep
  the declaration; `crop_to_grid` clips it. The extent specification explicitly
  chose preservation. Reconcile documentation of this exception rather than
  silently changing the policy.

These defaults are recommendations for the implementation review. The explicit
decisions in the temporal section must be closed before implementing that phase.

## Ownership contracts

| Concern | Owner | Contract |
| --- | --- | --- |
| Attr key combination | xarray, with `attrs.combine_attrs` diagnostics | Native key/value semantics; no pixel reads, alias normalization or group validation. |
| Metadata interpretation | `attrs/groups` | Read validated values from ordinary mappings; serialize their keys. No traversal or pixel conversion. |
| Group replacement | `attrs.stamp_attrs` / `clear_attrs` | Replace all owned keys, including aliases, on a shallow copy. Dtype checks remain at concrete raster writes. |
| One variable's conversion | `GeoDataArray` and private CF operations | Active attrs describe stored pixels; encoding describes a reverse storage recipe after decoding. |
| Dataset traversal | `GeoDataset` | Own variable selection, per-variable operations and result assembly. |
| Tree traversal | `GeoDataTree` | Own node traversal, inherited-coordinate handling, node errors and assembly. |
| Time-coordinate interpretation | Existing `RasterTime` | Identical DataArray/Dataset coordinate behavior; never dispatch on wrapped type. |
| Composite | Concrete DataArray/Dataset accessors | Native reduction plus observed periods, scene properties and method-specific metadata changes. |
| Source compatibility before fusion | STAC loader | Establish whether one output variable can represent the selected sources before metadata is lost or pixels are fused. |
| Output representation | Format writer / explicit encoder | Native format handling; preserve existing explicit checked conversion without double encoding. |

Keep short single-owner behavior on its accessor. Private modules are justified
only for substantial kernels or a genuinely shared operation with clear plain
inputs/outputs: CF preparation, grid/window calculation, period aggregation or
polygonization. They must not combine type dispatch, traversal, metadata policy
and format behavior. Do not pre-create an empty hierarchy of helper modules.
If adding internal modules, use private names so the API generator does not
accidentally advertise them as the replacement transform interface.

Genuinely identical base behavior can remain. In particular, `select_time` can
live in `RasterTime`; DataArray/Dataset composite traversal cannot. Existing
`mask_and_scale`/`scale_and_fill` delegation through concrete `_map_variables`
may remain if the base stays free of wrapped-type dispatch. Review this through
all three concrete accessors, including DataTree inherited coordinates.

## Complete public migration map

| Current interface | Target interface / implementation owner | Migration consequence |
| --- | --- | --- |
| `temporal.concat` | `xr.concat(..., dim="time", ...)` | Native alignment, promotion, variable and metadata rules become explicit. No new `.gs.concat`. |
| `temporal.interpolate` | `.interpolate_na(dim="time", ...)` | Caller chooses rechunking, extrapolation and method; no composite-bounds prerequisite. No new `.gs.interpolate`. |
| `temporal.select` | `.gs.select_time(when)` | Preserve interval overlap, unlike `.sel`; correct NaT interpretation. |
| `temporal.composite` | DataArray/Dataset `.gs.composite(...)` | Explicit decoding, original observation context, native reducers. Remove `skip_nodata`. |
| `combine.merge` | `xr.merge` in DataTree and STAC assembly owners | Retain required grid/identity checks locally; no new generic `.gs.merge` or renamed copy of strict merge. |
| `packing.mask_and_scale` | Existing `.gs.mask_and_scale()` | One variable implementation plus accessor-owned traversal. |
| `packing.scale_and_fill` | Existing `.gs.scale_and_fill()` | Retain explicit lazy integer safety and native codec behavior. |
| `nodata.mask` | Existing DataArray/Dataset `.gs.mask()` | DataArray calls native `where`; Dataset owns iteration. |
| `nodata.read_mask` | Private mask normalization | Retain ndarray grid interpretation and necessary validation; no second public entry point. |
| `nodata._mask_missing` | Private CF operation | Consolidate preparation with full decoding; keep masking-only semantics for I/O/display. |
| `extent.crop` | Concrete DataArray/Dataset `.gs.crop()` | DataTree retains its own traversal; preserve geometry/CRS and integer-mask behavior. |
| `extent.crop_to_grid` | Existing `.gs.crop_to_grid()` | Preserve exact pixel placement, clipped declaration and chip behavior. |
| `extent.pad` | Private implementation behind crop-to-grid | Do not add public `.gs.pad` without a concrete use case; native `.pad` alone does not reconstruct affine coordinates. |
| `warp.reproject` | Existing concrete `.gs.reproject()` | DataArray owns one warp, Dataset owns variable kernels/assembly; DataTree calls Dataset accessor. |
| `warp.read_pixels` | Existing Dataset `.gs.read_pixels()` | Retain bounded eager reads, float32 display output and no-overlap behavior; remove generic transform delegation. |
| `vector.vectorize` | Existing DataArray `.gs.vectorize()` | Keep geometry conversion, validity and lossless dtype checks. |
| `vector.rasterize` | Existing GeoDataFrame `.gs.rasterize()` | Vector accessor owns frame handling and burn; migrate GEDI consumer. |
| `color.colorize` | Existing DataArray `.gs.colorize()` | One implementation; Explorer uses the accessor. |

Move shared `Padding` and `Resampling` types beside their surviving operation,
not into an unrelated global types module. Inline short `GeoBox | SomeCRS`
destination annotations where appropriate. Preserve accepted BoundingBox crop
inputs in annotations and tests. Remove temporal aliases used only by deleted
wrappers. Keep reducer types with the surviving composite operation.

Standalone geospatial `pad` is intentionally withdrawn. Native `.pad()` is
available for array padding but is not a georeferenced replacement.
`.gs.crop_to_grid()` covers the target-grid use case only; document the loss of
the other standalone convenience rather than implying every old call is
equivalent to a new one.

## Behavior changes that must be visible

### Native concat and merge

Deleting `temporal.concat` deliberately removes its general-purpose checks for
matching dtypes, variables, active packing and CRS. `join="exact"` only checks
indexes; it does not prove equal CRS, compatible units or equal class meanings.
Recipes must decode and align explicitly before combination. Native dtype
promotion/fill behavior must not be described as GeoSave's old guarantee.

Required raster assembly guarantees remain in DataTree conversion and STAC
asset reading. Specify variable identity, dimension compatibility, requested
order, target GeoBox and CRS-coordinate normalization at those owners. Preserve
resource ownership: combining open arrays does not close their source objects
or assume ownership of their lifetimes.

Preserve these existing assembly contracts during the replacement; relaxing
them requires separate evidence and a deliberate interface change:

| Condition | DataTree to Dataset | STAC scene asset assembly |
| --- | --- | --- |
| Variable names | Unique across selected nodes; reject collisions. | Unique requested output names; reject collisions between asset results. |
| Spatial grid | Exact shared affine GeoBox after any explicitly requested reprojection. | Every read has the requested target GeoBox. |
| Variable dimensions | Same dimension names/order; no implicit broadcasting. | Preserve current equal-dimension requirement for assembled windows. |
| Scalar time / other coordinates | Require equality; caller selects observations explicitly. | Require compatible equal coordinates for the one scene/window. |
| CRS carrier | Normalize equivalent grid-mapping placeholders only after proving grid equality. | Same normalization on the verified target grid. |
| Ordering | Preserve selected-node and variable order. | Restore requested variable order. |
| Attrs | Child-over-parent precedence, then explicit selected native policy plus warnings. | Explicit drop-conflicts plus automatic warnings as today. |

Place each check once at its responsible owner. Reuse existing validation
where it already establishes the invariant; do not copy the entire former
strict-merge implementation into both callers.

DataTree conversion currently overlays parent attrs with child attrs and then
uses drop-conflicts. Preserve that order, add an explicit `combine_attrs`
argument if needed to expose the choice, and retain automatic drop warnings.
Do not introduce a public generic merge just to share two assembly call sites.

### Attr policy and warnings

Keep native strings/callables supported, including callback context forwarding.
Warnings report keys absent from the final mapping; they do not promise to
detect changed values retained by `override`. Explicit `drop` may consequently
warn for every omitted key. Do not add warning modes or another policy class.
Avoid redundantly wrapping GeoSave's own warning callback when it is supplied
as the policy. Custom callables still receive native context and retain their
own behavior; do not add recursive callable inspection or warning suppression.

`Dataset.gs.to_dataarray` already adds semantics beyond native stacking:
compatible shapes/packing and combined band attrs. Preserve those requirements
and warnings; do not silently replace it with native Dataset root-attr behavior.
Keep its default explicit because native `to_dataarray` has no equivalent
band-attrs combination policy argument.

Group-aware validation remains outside the callback. Native key-by-key results
can contain incomplete legends, and JSON strings compare as strings. This is
an explicit limit of the combination contract, not a reason to silently
normalize groups. Operations consuming semantic metadata validate the fields
they need. STAC common-properties intersection remains a separate operation.

### CF decoding and STAC loading

Consolidate missing-code normalization, ODC `nodata` alias handling and native
codec preparation. Preserve full decode versus masking-only behavior. Correct
the reproduced float32 `missing_value=0.1` masking discrepancy through a real
consumer, not just a private-helper test.

Before ODC loads/fuses selected bands, reject incompatible effective scale and
offset for both raw and decoded output. Do not add automatic per-source
decoding/fusion machinery in this refactor. Resolve only selected bands and
honor `stac_cfg` aliases/band indexes. Validate absent scale/offset defaults
against supported STAC/ODC semantics before writing the check; distinguish a
missing asset from a present band with an omitted field.

Do not require equality of every source `Packing` field. ODC may legitimately
remap differing source nodata to one output fill value. Its output nodata stays
authoritative. Reassess the synthetic loader-packing precedence test against
actual loader pixels rather than preserving a hybrid scale/offset dictionary.

### Composite and interval selection

Proposed interface retains current resampling controls:

```python
monthly = decoded.gs.composite(
    "MS", "mean", properties="intersect",
    closed=None, label=None, origin="start_day", offset=None,
)
april = monthly.gs.select_time("2024-04")
```

- Inputs with active CF packing/missing-value declarations must decode first;
  inspect attrs, not historical storage encoding. Already unpacked arrays need
  no marker declaring that they were decoded. Identity-only declarations still
  go through the explicit decoder to keep one straightforward input contract.
- Reject before executing lazy pixel tasks. Remove `skip_nodata`, internal
  decoding, and the stored-value reduction branch.
- Use native resampling once for pixels, with equivalent bin settings for
  period/property aggregation. Original observations remain available until
  all three results are assembled. Return a native DataArray/Dataset.
- No bounds means instantaneous observations at `time`. Scalar shared bounds
  broadcast to time labels before processing. Finite pairs must be ordered.
- A pair of NaT bounds means no observed coverage, never an open interval.
  One finite/one NaT bound is invalid. Selection excludes no-coverage rows.
  Review adjacent `time_bounds`, `write_time_bounds` and `timespan` behavior:
  mixed finite/empty periods must not erase actual observed coverage.
- Bounds describe the observed envelope, not calendar bin edges. Preserve this
  distinction for month-end labels, nested composites and STAC publication.
- Empty calendar buckets carry NaN pixels, NaT periods and empty properties.
  Observed but all-missing pixels follow the chosen native reducer behavior;
  native sum returns zero without `min_count`. Do not silently claim that this
  case is identical to a bucket containing no observations.
- Preserve `properties="intersect" | "stack"` and source ordering. These are
  scene records, not object attrs, and must not use drop-conflicts union.
- Append the mapped CF method to existing `cell_methods`, preserving earlier
  spatial and temporal operations: `std` means `standard_deviation`, `var`
  means `variance`, `min`/`max` mean `minimum`/`maximum`, and `first`/`last`
  mean `point`. Retain categorical first/last restrictions.
- Follow native data-encoding loss after reduction; do not restore a previous
  integer storage recipe. Preserve valid CRS/grid coordinate metadata for
  extraction and persistence.
- Dataset owns decisions about time-dependent/static variables and output
  assembly. Use native resample behavior unless a documented domain requirement
  demands otherwise; do not manufacture cell-method claims for unreduced values.
- Interpolation does not invent observed coverage. Proposed rule: later native
  reduction includes finite reconstructed pixels, while the observed-period
  envelope and scene properties include only originally observed inputs.
  A finite reconstructed pixel with NaT periods is not an empty numerical
  bucket. Protect interpolation → recomposition with a regression.
- No new DataTree temporal facade. Verify explicit native mapping against
  inherited non-index period/property coordinates; do not assume every tree
  mapping API exposes them. Show an explicit node workflow where required.

**Settled temporal decisions:**

1. Variance omits input `units`; quantity identity (`standard_name`) remains
   with the appended CF `time: variance` cell method. Callers can explicitly
   assign correct output units. No unit parser or string synthesis is added.
2. All-missing sum uses native xarray behavior (zero). No reducer kwargs layer.
3. `write_time_bounds` accepts paired NaT for absent observed coverage. Partial
   NaT is invalid; selection and recomposition exclude absent coverage.
4. Finite interpolated values contribute numerically to later composites, but
   never invent observed intervals or scene properties.

### Spatial behavior

Preserve real domain behavior during relocation: exact grids, native axis
names, integer fill masking, affine-coordinate reconstruction, wide lazy
reflection/wrapping, categorical kernels and windowed reading. Recheck ODC
workarounds with direct dependency comparisons before retaining or deleting them.

For `.gs.mask`, retain native index alignment and explicitly reject conflicting
known georeferencing before `where` can drop a CRS carrier. Unreferenced boolean
arrays remain accepted under the current shape/dimension contract. Do not
silently reproject a mask. Preserve the caller's valid grid metadata.

Existing fill rules differ: mask/crop require a fill, warp may choose one with
a warning, pad may promote dtype, and crop-to-grid defaults to zero. Record and
test these existing contracts. Unifying them is a separate decision; a move
must not silently change numerical padding or chip labels.

## Phased implementation

Each phase is a reviewable change with its own focused checks. Do not mix file
relocation, broad formatting and unrelated behavior changes in one diff.

### Phase 0 — Capture baseline and contract decisions

- [x] Recheck live status/source and record touched files before editing.
- [x] Record dependency versions and run the focused baseline below; classify
      failures/xfails/deselections separately from regressions.
- [x] Close temporal decisions above with a small smoke before settling design.
- [x] Record proposed defaults accepted for native writer safety, attrs defaults
      and extent preservation; avoid silently expanding their scope.
- [x] Inventory direct imports, callable strings, type-only imports and docs
      directives, excluding notebooks. Include the GEDI consumer.

Gate: all retained behaviors, deliberate removals and numerical changes have
named owners and observable tests. No claim of a clean whole-repository baseline.

### Phase 1 — Establish attrs and CF lifecycle

- [x] Keep native combination semantics and automatic dropped-key warnings at
      retained GeoSave call sites; preserve custom policies and context.
- [x] Consolidate CF preparation under the one-variable conversion owner;
      migrate private masking consumers in I/O and Explorer.
- [x] Repair source scale/offset compatibility before ODC fusion. Replace the
      test that requires conflicting scale to disappear with real-value coverage.
- [x] Preserve authoritative loader nodata, raw/decoded round trips and explicit
      checked encoding. Correct contradictory metadata documentation.

Gate: local STAC loads cannot return falsely decoded values; full decode and
masking-only consumers agree on missing pixels; CF conversion remains lazy.

### Phase 2 — Move operations with one clear owner

- [x] Implement colorize/vectorize on DataArray and rasterize on the vector
      accessor, retaining private numerical kernels only where useful.
- [x] Replace Explorer's direct colorize call with `.gs.colorize()`.
- [x] Move boolean masking into concrete DataArray/Dataset ownership and add
      the known-CRS conflict check without restricting legitimate broadcasting.
- [x] Plan/migrate GEDI's two rasterize calls to `cells.gs.rasterize(...)` with
      the same return type, grid, dtype and fill. Do not regenerate data or
      modify its quality-screening policy.

Gate: codes/colors, vector geometry, grids, lazy behavior and GEDI target
values remain equivalent. No public and private copies of the same operation.

### Phase 3 — Replace temporal wrappers

- [x] Implement shared interval selection in `RasterTime.select_time` and
      concrete DataArray/Dataset composite behavior.
- [x] Preserve and recompute periods/properties from original observations.
- [x] Add regressions for decoded reductions, NaT selection, scalar periods,
      cell-method history and reduction encoding.
- [x] Migrate native concat/interpolate uses and examples with explicit changed
      defaults, rechunking and extrapolation choices. No compatibility aliases.
- [x] Move surviving temporal tests to their accessor owners; remove tests
      whose sole purpose is the deleted wrapper's restriction or delegation.

Gate: decode → composite → select → extract band → save/load/publish works;
native recipes clearly state which previous safeguards they no longer provide.

### Phase 4 — Move spatial operations and raster assembly

- [x] Put crop/crop-to-grid on concrete accessors; keep DataTree traversal local.
- [x] Retain private padding/grid calculation for chips without a new public pad.
- [x] Put one-variable warp on DataArray and per-variable kernel traversal on
      Dataset; have DataTree call those accessors.
- [x] Put bounded display reading behind Dataset `.gs.read_pixels`; preserve
      its eager window and missing-region behavior.
- [x] Replace strict merge at DataTree/STAC owners with native merge plus their
      required local checks and shared CRS normalization where truly identical.
- [x] Move shared annotations and remove imports of transform-owned aliases.

Gate: exact-grid chip extraction, DataTree assembly, requested STAC variable
order, categorical protection and Explorer reads retain their contracts.
Existing native-ODC warp discrepancies remain separately reported.

### Phase 5 — Remove the public transform package and migrate documentation

- [x] Remove transform exports/imports in `geodata/__init__.py` and delete the
      old modules only after every live Python consumer has migrated.
- [x] No aliases, deprecated duplicate paths or generic replacement facade.
- [x] Update API docstrings and active architecture/metadata/catalog-window
      guides. Replace the transforms reference with native/accessor recipes or
      integrate it into existing raster docs; remove stale autodoc directives.
- [x] Regenerate references with `scripts/api_reference.py`, then check them.
- [x] Update navigation and mark conflicting historical plans superseded where
      relevant, without rewriting their historical evidence.
- [x] Report excluded notebook/external consumer migration; do not silently
      edit notebooks or claim compatibility with removed import paths.

Gate: no live supported production imports of the removed package, accurate
generated API inventory, no hidden replacement operation surface.

### Phase 6 — Integrated verification and handoff

- [x] Run focused workflows and required broader checks below after edits stop.
- [x] Compare to baseline; investigate new failures, and record existing gaps.
- [x] Review import-time dependencies, resource lifetimes, lazy execution and
      exact output metadata alongside numerical values.
- [x] Report changed interfaces, checks, breaking changes and remaining risks.
      Do not describe a run containing xfails/deselections as fully green.

Dependencies: Phase 0 precedes all changes; Phase 1 precedes temporal migration
and CF-consuming spatial changes. Phase 2 and temporal work can be reviewed
separately once ownership is settled. Phase 4 precedes deletion; Phase 5 follows
all consumer migrations; Phase 6 follows the final edit. Prefer sequential
integration even if implementation research is parallel.

## Caller and documentation checklist

Required production review:

- `core/raster/{base,dataarray,dataset,datatree,chips}.py` — public delegation,
  traversal, shared types, assembly and window reads.
- `core/vector/catalog.py` — rasterization.
- `io/odc.py`, `stac/band.py` — selected-source semantics and shared metadata.
- `io/stac.py` — multi-asset assembly, masking and open-resource lifetimes.
- `io/formats/{geotiff,netcdf,zarr}.py`, `io/rasterio.py` — active versus encoded
  CF state, single encoding and format differences.
- `viz/explore/layers/raster.py` — colorization, identify and bounded reads.
- `model/processing.py`, feature consumers and template processing modules —
  existing decode/accessor workflows must keep working without redesign.
- `workspace/gedi-biomass/modules/targets.py` — known direct rasterize import.

Test consumers beyond `tests/geodata/transform` include `tests/geodata/conftest.py`,
attrs group tests, raster/vector accessor tests, STAC Item tests, I/O STAC and
GeoTIFF tests, Chips, Explorer, model processing and template tests. Check
fixtures as real callers; their concat helpers can hide changes in defaults.

Documentation includes `docs/concepts/{metadata,catalog-windows}.md`,
`docs/basics/attrs.md`, `docs/reference/{attrs,rasters,io,transforms,public-api}.md`,
`docs/reference/generated/data.md`, architecture guidance and `zensical.toml`.
Use the generator for generated content. Update architectural guidance about
pixel-changing implementations only as needed for the accepted ownership plan.

## Verification matrix

| Workflow / failure mode | Required evidence |
| --- | --- |
| Native attr policies + GeoSave warnings | Same result as native for supported strings/callables, missing keys, NaN and conflicts; warning only for omitted keys; no pixel execution. |
| Raw/decoded CF lifecycle | Real TIFF/NetCDF/Zarr round trips preserve physical values, nodata, CRS and names; decode/encode exactly once. |
| STAC source scales | Actual local assets; mixed effective scales rejected before invalid output for raw and decoded loads; uniform scales decode correctly; loader nodata remains authoritative. |
| Masking-only precision | float32 missing code removed through a window/identify consumer; valid DN unchanged; full decode agrees on missing positions. |
| Checked integer conversion | Existing overflow, fill-collision and missing-fill failures occur when pixels evaluate; native writer contract stays explicit. |
| Composite arithmetic | Explicitly decoded `[1,3]`, scale 2/offset 10 gives sum 28/std 2/variance 4 if retained; raw input rejected without computing pixels. |
| Period lifecycle | Scalar and per-time periods agree; Jan/Mar empty Feb does not match April; paired NaT, partial NaT and reversed pairs have consistent read/select/recompose behavior; month-end/nested periods remain observed spans; mixed empty/finite coverage remains usable. |
| Composite metadata | Existing cell methods retained, per-scene properties rebuilt, categorical rules protected, data encoding not wrongly restored, chosen variance/missing-sum rules enforced. |
| Dataset static variables | Preserve native resample shape behavior but do not claim temporal reduction for variables that lacked time in the source. |
| Interpolation then composite | Finite reconstructed values follow the chosen numerical rule; coverage/properties follow the original observed inputs. |
| Composite persistence | Composite → band extraction → NetCDF/Zarr → reopen and STAC publication retains periods/properties, values and grid. |
| Spatial mask | Known CRS conflict rejects before metadata loss; matching-grid masks, ndarray input and permitted broadcasting still work lazily. |
| Chips and padding | Exact target GeoBox, dtype/fill and axis order; wide reflection/wrap and outside coverage; no eager full-raster read. |
| Declared extent | Ordinary crop/reprojection preserve the declaration; crop-to-grid clips it as documented. |
| Warp | Native eager reference, lazy dependency comparison, per-variable categorical kernels, output CRS and dimensions; known xfails remain narrow. |
| Raster assembly | Duplicate variables, mixed grids, scalar time, CRS placeholders, parent attrs, requested band order and source resource ownership. |
| Vector/color | Burn/polygonize values, representable dtype, exact grid, valid masks, legend colors and lazy RGB. |
| Consumer integration | Explorer bounded reads, GEDI target rasterization, model decode/tensor preparation, template processing; no task redesign. |

Tests follow surviving source owners. Do not mechanically copy all old wrapper
tests or test exact exports/removed names. Preserve distinct behavior protection;
use source searches and generated-reference checks for import cleanup. Add
regressions for the numerical/lifecycle defects before their fixes, then remove
duplicate coverage only after the surviving public workflow is protected.

Baseline commands (paths change as tests migrate):

```bash
uv run pytest tests/geodata/attrs tests/geodata/core/raster tests/geodata/utils/test_mapping.py
uv run pytest tests/geodata/io/test_packing.py tests/geodata/io/test_odc_metadata.py tests/geodata/stac/test_band.py tests/geodata/core/raster/test_dataset.py
```

After migration, run affected accessor/Chips/STAC/I/O/viz/model/template tests,
then the default suite with `--ignore=tests/examples` to honor notebook exclusion.
Report that exclusion and default slow/integration deselections explicitly.
Run applicable local slow checks separately when they exercise import/lifecycle
contracts changed here; do not trigger unrelated external services as a default.
Inspect project-specific GEDI tests before choosing its bounded validation.

Run Ruff and basedpyright on affected source/tests, comparing existing errors
against the recorded baseline. Regenerate and check API references, build docs,
check rendered references, and run `git diff --check` on changed paths. Avoid
whole-repository autoformatting or notebook-mutating pre-commit hooks.

```bash
uv run python scripts/api_reference.py
uv run python scripts/api_reference.py --check
uv run zensical build
uv run python scripts/api_reference.py --check --site site
```

## Review evidence, not implementation completion

Previous read-only review in this conversation, using installed xarray 2026.4.0:

- Attrs/transform/mapping suite: **237 passed, 3 xfailed, 15 warnings**.
- Related I/O/STAC/Dataset suite: **105 passed, 2 warnings**.
- Three existing warp xfails cover native lazy/eager discrepancies; they are
  not passing cases. The whole repository suite was not run for this review.
- Restricted Zarr execution stalled; the same broader checks completed outside
  the sandbox. Do not treat that stall as a product regression.
- Local-file probes confirmed lost source scale before decoding and native
  NetCDF overflow versus explicit checked conversion. In-memory probes confirmed
  masking precision drift, partial legends, NaT selection and packed reductions.

Planning smoke on 2026-11-10 independently verified that explicit decode followed
by native lazy resampling gives sum 28, std 2 and variance 4; native period
aggregation gives Jan 8–24 for observations in that span; native sum distinguishes
all-missing from empty groups and `min_count=1` changes the former. Direct native
attr policy and the callback produced identical outputs with callback warnings.
These probes support the direction; they do not prove the proposed accessor
implementations, which do not yet exist.

## Breaking changes and completion conditions

Breaking changes include removal of transform import paths and their aliases,
native concat/interpolate defaults and safeguards, explicit decoding before
composite, removal of `skip_nodata`, and any explicitly selected reducer metadata
changes. Keep a caller-oriented migration example for each removed public path.

Implementation is complete only when all retained domain behaviors have one
owner, all known in-scope Python callers migrate, native policies plus automatic
GeoSave warnings are consistent at retained combination sites, numerical and
persistence regressions are protected, docs describe live behavior, and the
handoff identifies notebook/external consumers and remaining test limitations.


## Implementation record — 2026-11-10

Concrete DataArray/Dataset accessors now own composite, crop, masking and warp;
DataArray owns colorization/polygonization, GeoCatalog owns rasterization, and
Dataset owns bounded pixel reading. DataTree and STAC assemble with native
merge plus local grid/coordinate checks. `geosave.transform` and the entire
public `geodata.transform` package are removed without aliases. Private modules
hold only CF conversion, numerical kernels and coordinate/window calculations.

The CF boundary fixes float32 missing-code precision and integer/text scale
metadata that otherwise turns floating NaNs into integer sentinels. ODC rejects
selected sources with incompatible effective scales/offsets before fusion.
Native attr results are unchanged; GeoSave retains automatic dropped-key
warnings, including avoiding duplicate warnings for its own callback.

Temporal coverage supports scalar periods and paired NaT, preserves inherited
DataTree observation coordinates, appends CF cell methods, and distinguishes
interpolated numerical contributions from observed provenance. Wide lazy
reflect/wrap padding, CRS-safe masking and static-variable broadcasting have
regression coverage. Python consumers and active documentation are migrated;
notebooks are intentionally untouched. GEDI target code uses its vector
accessor directly.

Environment: xarray 2026.4.0, odc-geo 0.5.1, odc-stac 0.5.2, dask 2026.3.0,
NumPy 2.4.4 and rasterio 1.5.0.

Verification (counts are separate runs, not additive):

- Pre-change focused baseline: **342 passed, 3 xfailed**.
- Spatial/assembly/accessor integration: **146 passed, 2 deselected, 3 xfailed**.
- Temporal/base/vector samples/STAC/GeoTIFF consumers: **165 passed,
  4 deselected**; final temporal regression run **28 passed**.
- Vector/color/rasterize: **29 passed**. Wide-padding regressions: **4 passed,
  28 deselected**. Final packing regression run: **16 passed**.
- Broad final geodata check: **992 passed, 2 failed, 16 deselected, 3 xfailed**.
  Both failures were pre-existing tests attempting to stack incompatible
  packing. Their workflows now explicitly decode and assert physical values;
  the two containing modules subsequently passed **56 tests, 2 deselected**.
- Default suite excluding `tests/examples` first stopped on three collection
  errors. Continuing collection produced **1509 passed, 2 failed, 35 deselected,
  3 xfailed, 3 collection errors**. The two failures are the migrated tests
  above. The untouched model tests `test_clay`, `test_prithvi` and dependent
  `test_huggingface` still import absent `attrs.TimeSpec`; the initial snapshot
  confirms this predates the refactor. This is not a clean whole-suite result.
- Explorer local-server lifecycle: **5 passed, 3 deselected** (notebook/browser
  checks excluded). Fresh import smoke confirms `torch` stays unloaded.
- GEDI in-memory rasterization/aggregation smoke passed. Its two selected
  workflow tests stop in an unrelated stale `to_cog(path_format=...)` fixture;
  no generated target data was modified.
- Changed Python paths pass Ruff and scoped whitespace checks. Changed-source
  basedpyright reports **0 errors, 2 existing lazy-export warnings**. A broader
  raster-directory check also finds the unchanged possibly-unbound `fill` in
  `utils/statistics.py`; this is recorded rather than hidden.
- API generation/check and Zensical build pass; rendered coverage is
  **1294/1294** public definitions/members.

Review used the pre-implementation snapshot, because HEAD includes extensive
uncommitted earlier work. Standards review confirmed concrete ownership and
identified readability cleanups, which were applied. Spec review found the
integer-scale decoding issue and missing inherited-period regression, both
fixed. Changes remain uncommitted to avoid including overlapping pre-existing
work. External consumers and notebooks need the migration recipes in
[the transforms reference](../../reference/transforms.md).


## Example and stale-test follow-up

The example migration is now included. Notebooks 2 and 3 use native concat
with exact index alignment and the GeoSave attrs callback. Their packing,
collection publication and writer arguments use current interfaces. Image and
label rasters publish separate Items with native Collection definitions;
`Chips` remains lazy until the Dataset or prediction loop converts a chip to
a tensor. Notebook 4 explicitly writes multiband review imagery as one asset.
Notebook outputs and execution counts were not regenerated.

The two obsolete `TimeSpec` bucket-label tests and their imports were deleted.
Other Clay/Prithvi context, checkpoint and release tests remain. The previous
three collection errors are resolved by this cleanup.

Local example tests: **5 passed, 5 integration tests deselected**. Smoke checks
execute the actual updated concat, season/map catalog publication and review
COG cells using tiny local rasters, asserting dates, grids, decoded values,
laziness and written artifacts. All five notebooks validate and parse; scoped
Ruff and whitespace checks pass. Full network downloads and training notebook
runs are not part of this local verification.

Final default suite (including local example tests): **1545 passed, 41 deselected,
3 xfailed, 151 warnings**. No failures or collection errors. Default markers
exclude slow/integration checks; the three existing ODC warp xfails remain.
