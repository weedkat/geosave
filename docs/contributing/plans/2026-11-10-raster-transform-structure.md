# Raster transform ownership and explicit function contracts

Status: implemented; final verification recorded below. The user selected `raster/transform`
for pixel-changing implementation and `raster/utils` for structural support.
This plan corrects the internal layout and helper contracts of
[the native operations refactor](2026-11-10-native-transforms.md), retaining its
accepted public behavior. It supersedes that plan's instruction to use private
module filenames to control generated documentation.

```python
# Caller: native objects and existing accessor conveniences.
decoded = raster.gs.mask_and_scale()
monthly = decoded.gs.composite("MS", "mean")
chip = decoded.gs.crop_to_grid(grid)

# Maintainer: substantial operation implementations have a named home.
from .transform import temporal

# Inside a concrete accessor, after its variant-specific checks:
result = temporal.composite(self._data, freq, method, ...)
```

The last call illustrates ownership, not a finalized signature. No new public
standalone transform API, resampler wrapper, compatibility alias, or accessor
hook framework is proposed.

## 1. Start with native input and output

Both DataArray and Dataset already support native resampling over their datetime
coordinate. A composite implementation receives that raster, not separately
constructed arrays of starts, ends, positions and records.

Smoke evidence from installed xarray 2026.4.0:

- Source observation times: January 5, January 20, March 8.
- Source values: 1, 3, 9, with lazy Dask pixels.
- Source has `start_datetime`, `end_datetime`, and `stac_properties` coordinates.
- `grouped = raster.resample(time="MS")` supports both native raster types.
- `grouped.mean(keep_attrs=True)` produces January/February/March values
  2/NaN/9. In this fixture only `time` survives as an output coordinate.
- Iterating that same `grouped` object yields the January and March source
  groups, including their original metadata coordinates. It does not yield an
  empty February group.
- Constructing the result and iterating source groups executed no pixel tasks.

This verifies a native mechanism, not the completed composite implementation.
Reference: [Dataset.resample](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.resample.html)
and [DataArray.resample](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.resample.html).

For this structural refactor, preserve current observed-coverage behavior:
January's output covers January 5–20, February has paired NaT, and March covers
March 8. Existing source intervals are read from the raster when present.
These are GeoSave metadata rules, not prerequisites of native resampling.
Calendar boundary semantics are not introduced by this refactor.

## 2. Directory ownership

```text
core/raster/
  base.py, dataarray.py, dataset.py, datatree.py, chips.py
  utils/
    grid.py          # grid discovery/calculation and coordinate placement
    statistics.py    # summaries
    tensor.py        # explicit representation conversion
  transform/
    __init__.py      # no facade or duplicate exports
    packing.py       # CF decode, checked encode, missing-pixel masking
    temporal.py      # composite a native raster
    extent.py        # crop and exact-grid crop/padding
    nodata.py        # mask normalization/application where shared
    warp.py          # one-array reprojection and related numerical behavior
    vector.py        # polygonize a raster plane
```

The list defines operation ownership, not a requirement to create shallow
modules. Retain short single-owner methods such as colorization on their
accessor unless inspection demonstrates a cohesive operation worth extracting.

- Structural support does not select/resample/change pixel values. Existing
  grid assignment, statistics and explicit tensor conversion keep their roles.
- Transform implementations change pixel values, select/rearrange pixels, or
  derive geometry from pixels. They use native xarray/ODC operations directly.
- DataArray owns one-array policy. Dataset owns variable selection, validation
  differences, per-variable kernels and result assembly. DataTree owns nodes
  and inheritance. No helper dispatches on the wrapped xarray object's type.
- Genuinely identical DataArray/Dataset operations may use a constrained generic
  `T` and return `T`; do not copy a body merely to keep two accessor methods.
- Rasterization stays under vector ownership. Audit `_rasterize.py` against the
  existing vector utilities convention; do not move it into raster ownership
  solely because its output happens to be a raster.

## 3. Function contract rules

Before extracting or moving a function, write its input, output, side effects,
required invariants, and actual callers. Remove a helper when that exercise
shows only forwarding or incidental intermediate packaging.

| Concern | Input | Output | Responsibility |
| --- | --- | --- | --- |
| Composite | Native DataArray or Dataset plus frequency/reducer and existing options | Same native type | Native temporal reduction plus defined output observation metadata |
| CF conversion | One DataArray | One DataArray | Explicit representation conversion; traversal remains with concrete accessors |
| Exact-grid crop | Native raster plus target GeoBox and padding options | Same native type | Select/pad pixels and write the resulting exact grid |
| Reprojection | One DataArray plus explicit destination and kernel | DataArray | One-array numerical warp; Dataset selects kernels and assembles variables |
| Polygonization | One raster plane and explicit polygonization options | GeoDataFrame | Validity, representable values and geometry construction |

Rules:

- Keep coordinates attached to the observations they describe. Avoid parallel
  arrays/lists whose correspondence is maintained only by position.
- Return the actual result, usually the native raster or geometry object.
  Do not return a dictionary of fixed intermediate outputs for callers to
  reconstruct the operation.
- Fixed multi-value numerical results can use explicit tuples where this is
  clearer than a native object. Do not add throwaway result classes.
- Dictionaries remain appropriate for actual arbitrary-key attrs/STAC property
  data. They must not stand in for a fixed result schema. Narrow value types
  where the true contract supports it; a type alias alone is not a redesign.
- Mutation, computation, decoding and encoding must be explicit. Preserve lazy
  pixels and caller-owned input objects. Metadata inspection must not execute
  raster pixel tasks.
- Helpers stay with their operation. Do not create global configuration bags,
  generic dispatchers, callback pipelines, or helpers only to shorten methods.

## 4. Temporal correction first

Retire `aggregate_periods(labels, bounds, bins)` and
`aggregate_properties(labels, scenes, present, bins, output, how)` after their
behavior is protected through the surviving composite interface. The withdrawn
`resample_periods(start, end, bins)` proposal is not part of this plan.

Implement and smoke-test a single understandable flow:

1. Concrete accessor checks its own variables/representation and categorical
   rules. Retain explicit decoding, static-variable behavior and reducer rules.
2. The temporal implementation receives the source raster with coordinates.
3. Construct one native resampler with the requested frequency, closed/label,
   origin and offset options; perform the named native numerical reduction.
4. Use the source groups exposed by that same native grouping to calculate the
   required output observation metadata. Do not build a second resampler over
   synthetic start/end/position arrays.
5. Match metadata to the numerical result by time labels. Empty result labels
   receive paired NaT and empty properties. Do not assume group iteration
   includes every output label or use positional zip against the output.
6. Return the complete native raster. Keep variable-specific cell-method and
   quantity updates with the concrete owner when DataArray/Dataset differ.

Preserve original intervals, source property order, paired NaT behavior and
interpolated-pixel versus observed-provenance rules. STAC JSON encoding remains
with the existing STAC coordinate owner. Reading JSON must not spread into
unrelated grid, attrs or pixel functions.

Do not move the six-argument properties helper unchanged behind another name.
First keep the group loop local to the operation; extract a module-local helper
only when its complete inputs and return meaning are independently clear.
No final helper signature is approved before this flow passes smoke checks.

## 5. Spatial and CF consolidation

- Decompose `_spatial.py` by responsibility: grid/window mathematics belongs
  with grid support; crop/padding with extent; mask rules with nodata.
- Remove duplicate crop/crop-to-grid bodies through the genuinely identical
  operation, preserving concrete accessor behavior rather than type dispatch.
- Move substantial one-array warp behavior together; keep Dataset kernel
  selection/assembly on Dataset. Do not preserve a helper module consisting
  mainly of destination parsing, mapping lookup and policy fragments.
- Move `_packing.py` into the operation layer; preserve its shared use by I/O
  and display code. Do not duplicate masking or decoding at those consumers.
- Give polygonization one meaningful cross-module entry point. Keep dtype and
  geometry kernels local instead of importing three underscored functions into
  the accessor. Apply the same review to vector rasterization.

## 6. Naming and visibility

- `function`: a deliberate interface used by other modules. Living in an
  internal implementation package does not make it a supported user API.
- `_function`: used only inside its defining module; do not import it elsewhere.
- Supported user APIs are explicitly exported/documented accessors or functions.
- Normal descriptive module filenames under the agreed packages; no parallel
  flat underscore-module hierarchy.
- Update API inventory policy explicitly to distinguish supported API from
  internal implementation. Preserve intentional public summaries/other existing
  exports; do not blindly hide every module under utils.
- Reconcile AGENTS.md, CLAUDE.md, architecture docs and migration references with
  the chosen ownership. Earlier evidence remains historical; supersede only
  the conflicting design decisions.

## 7. Implementation sequence and gates

1. **Snapshot and baseline.** Preserve the dirty tree and capture task-local
   diffs. Re-run relevant smoke cases before relocating code.
2. **Temporal vertical slice.** Establish native grouping and complete raster
   input/output for DataArray and Dataset. Review its real caller and maintainer
   code before broad relocation. No new interface solely to fit a folder plan.
3. **CF and spatial operations.** Consolidate one operation at a time, migrate
   callers, then delete the superseded helper. Separate behavior changes from
   moves; preserve required checks.
4. **Visibility and documentation.** Update inventory rules and architectural
   guidance. Check for cross-module underscore imports and stale paths.
5. **Integrated verification.** Run affected workflows, then the default suite
   once after edits stop. Report failures, skips, deselections and xfails
   separately. No commit of unrelated pre-existing changes.

Required temporal cases: DataArray/Dataset equivalence to native pixel
reductions, empty/all-missing groups, scalar/existing periods, recomposition,
nondefault labels/closed/origin/offset, static variables, categorical guards,
source properties, inherited DataTree coordinates, interpolation provenance,
and save/reopen/STAC publication. Assert no pixel tasks during construction.

Required spatial/CF cases: wide lazy padding, exact grids, nodata and packing
round trips, known-CRS mask rejection, categorical warp kernels, display windows,
DataTree assembly and existing example workflows. Preserve the known native ODC
warp xfails; do not weaken them for an architectural cleanup.

Completion requires fewer coordination points and clear callable contracts,
not merely renamed files or a passing test count. No production changes are
authorized by this planning document alone.

## Optional interpolation after compositing

User clarification: add an interpolation option to `composite`, rather than
requiring a separate user operation. Proposed call:

```python
monthly = decoded.gs.composite("MS", "mean", interpolation="linear")
```

`interpolation=None` preserves existing behavior. When selected, first compute
the native reduction and then interpolate missing pixel values along the
result's existing time coordinate. Delegate to native `interpolate_na`; do not
create a second timeline, silently extrapolate, or interpolate observation
metadata. Keep the existing categorical protections; interpolation over class
codes requires its own explicit policy and must not silently blend them.
Finalize supported methods and lazy chunk handling through implementation smoke
checks, rather than introducing a general interpolation options container.

`.interp(time=...)` instead evaluates at supplied target coordinates. A smoke
with both DataArray and Dataset confirms that `.interp(time=result.time)` does
not fill a missing February value, whereas `.interpolate_na(dim="time",
method="linear")` does. Native `resample(...).interpolate(...)` is a separate
upsampling path, not a mean reduction followed by gap filling. The requested
composite option uses the latter sequence: reduce, then optionally fill gaps.

References: [interp](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interp.html),
[interpolate_na](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.interpolate_na.html).
Implemented with `dim="time"`, optional linear/nearest interpolation, and lazy
rechunking of the selected dimension.
