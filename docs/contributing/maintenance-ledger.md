# Maintenance closure — 2026-10-11

The previous run started at `d2e81f7` and left its fixes in the shared working
tree. This follow-up finishes structural cleanup only, as the user confirmed.
Existing fixes, generated workspace content, and notebook files are preserved.

## Closed structural decisions

| Prior item | Result |
| --- | --- |
| G1-6 | `GeoSample` moved from `geodata.core` to `geodata.cuts.sample`; importing cuts registers the pandas row accessor. |
| G2-6 | `stac._item` became public `stac.item`; `cog_files` and `instant` are exported from stac. File records read native PySTAC assets and EO extensions. `instant` reads serialized source snapshots. |
| G3-2 | Spatial Dask mapping moved to features; geometry, variables, statistics and geocoding moved to core. Tests and imports follow their owners. |
| G3-1 | Unused Clay-specific sensors package and its isolated tests removed. |
| G3-4 | `crop.window` accepts constant, edge and reflect, matching chips; wrap is rejected even when no padding is needed. |
| G2 SAFE | Unimplemented SAFE reader removed from advertised format exports; internal dispatch still reports its explicit NotImplementedError. |
| WC-1/2 | Pure-raise dense tasks removed. The pending dense flow is no longer registered as a CLI command; the implementation of label indexing is retained. |
| ML-2 | `ml/data.py`, `inputs.py`, `merge.py`, `predict.py` and `apply_thresholds` removed. Validation scores chips over valid pixels. |
| ML-3 | Task datasets removed. Each DataModule declares a `Sample` TypedDict and takes one Dataset per split; FixMatch and Mean Teacher add an `unlabeled` Dataset. |
| Dependency extra | Duplicate viz extra removed; visualization dependencies remain core dependencies. Lockfile regenerated. |
| Internal documents | Active proposals, open decisions and this ledger moved to `planning/`, outside the published docs tree. Completed ML plan removed; architecture and guides describe the implemented behavior. |
| Root artifacts | Generated segmentation/regression smoke workspaces moved under ignored `workspace/`, preserving their changed train configs. Templates remain under `src/geosave/templates`. |

## Previous-run fallout reconciled

- The empty-input Dataset test now disables context so it actually reaches
  Dataset validation rather than the new ModelSpec context invariant.
- Prediction item-table persistence is typed as `GeoCatalog`, whose writer
  owns Collection metadata.
- Obsolete strict xfails removed from the working ingest tests.
- Public guides now describe the implemented ModelSpec preparation and
  prediction writers rather than referring to those APIs as pending stubs.

## Deferred features

The user explicitly excluded checkpoint initialization, catalog folders and a
CI doctest job from this cleanup. Other open product design questions remain in
`open-decisions.md`. The dense Prefect flow and its optional scaffold remain
explicitly unimplemented; use `ModelSpec.prepare` directly.

## Validation

- Before extraction: focused baseline 166 passed, 1 failed (the fixture issue
  above), 2 deselected.
- Module moves: 20 passed, including lazy row recovery and padded pixels.
- Shared merge and task evaluation: 49 passed, 1 deselected.
- Ruff: clean. Zensical build: passed.
- Corrected CLI, prediction and pixel-window checks: 48 passed, 1 deselected.
- Native accessor import/row recovery checks: 12 passed.
- Multiband declarations and recovery: regression failed with `('image',)`
  before the fix and passes with `('red', 'nir')`; snapshot schema check passes.
- Scoped BasedPyright on moved helper modules, row accessor, native item helper,
  shared ML dataset/merger and both task modules: 0 errors, 0 warnings.
- Repo-wide BasedPyright: 16 errors, 4 warnings remain in existing conventions,
  cuts blending, raster reader, storage and lazy exports. These are dependency
  overload/type-inference gaps (GeoDataFrame/NumPy/xarray/Affine/UPath and
  option dictionaries), outside this structural cleanup. No broad suppressions
  were added.
- `uv lock --check --offline`: passed.
- Final full default suite with local runtime access: **1901 passed,
  36 deselected, 3 xfailed**, in 203.72 seconds. The three expected failures
  retain the existing ODC lazy-interpolation limitation.
- `git diff --check`: passed. Generated site contains no internal planning pages.

## Final review

A read-only reviewer identified missing STAC extension declarations in cuts
source snapshots after the native asset-reader conversion. A multiband catalog
regression reproduced the missing band names; snapshots now preserve
`stac_extensions`. Recovery also checks pixels, grid, times and laziness.
The existing snapshot-schema expectation was updated to include the declarations
needed by native PySTAC extension access. No other actionable structural
findings were reported.

Whole-chip retention in evaluation predates this extraction; memory behavior
was preserved.

An unrelated edit to `examples/00_basics.ipynb` appeared during validation and
was left untouched; notebooks were not changed by this cleanup.
