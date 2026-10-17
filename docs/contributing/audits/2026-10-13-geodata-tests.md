# Geodata core and transform test audit — 2026-10-13

> Audit snapshot. Findings, paths, and check counts describe the working tree
> reviewed in this record and may be superseded by later changes. Consult
> the [current guides](../../activities.md) and [API reference](../../reference/index.md)
> for supported behavior.

Scope: all 11 files under `tests/geodata/core`, all 10 files under
`tests/geodata/transform`, `tests/geodata/test_conventions.py`, and
`tests/geodata/conftest.py`. All **23 files / 5,730 lines before removal**
were read in full, including decorators, fixtures, comments, helper bodies,
failure assertions, and the conditional typing assertion.
Source ownership is reviewed separately by the root audit.

## Data flow inspected

```text
native arrays + explicit dims + GeoBox/native reference
    -> raster/array constructors -> gs accessors
    -> explicit nodata/packing/CRS operations
    -> stack shared/root coordinates or independent group grids
    -> lazy crop/window/warp/align/merge/concat/mosaic
    -> native Dataset/DataArray/DataTree
    -> explicit eager NumPy/tensor/statistics/vectorization boundaries
    -> actual GeoTIFF/NetCDF/Zarr persistence and native STAC tables
```

These tests exercise native objects and metadata as contracts, rather than
an obsolete wrapper hierarchy. Existing negative API assertions confirm
the vector accessor remains spatial and persistence does not implicitly
accept catalog/id arguments. Their purpose is to prevent retired execution
paths returning; they are not themselves dead code.

`tests/geodata/conftest.py::read_items` traces through
`stac_geoparquet.to_item -> translate.item.files -> cogs.open_raster`, so internal
CogFile records remain necessary to recover external asset-key identities.
The xarray accessor typing assertion under `TYPE_CHECKING` is intentional
static validation and was retained even though it is not runtime-called.
Every ordinary helper was checked for actual test callers, and pytest
fixture decorators/injected arguments were treated as registrations.

## Changes

- Removed `tests/geodata/core/test_vector.py::_written` and its now-unused
  format imports. Its sole caller was a timeless `scene.tif` provenance
  check; the Zarr branch, time-axis squeeze and sample-layer abstraction had
  no remaining callers. That test now calls the ordinary public
  `build_raster().gs.to_cog(...)` directly. The before/after provenance smoke
  both passed; no asserted behavior was dropped.
- Corrected the old band-first wording in
  `test_model_io.py::test_to_numpy_stacks_variables_ahead_of_the_other_axes`.
  The existing assertion used two dates and two bands, so `(2, 2, 8, 8)`
  could not distinguish axis order. The renamed test uses three dates and
  expects `(3, 2, 8, 8)`, matching the inspected
  `GeoRaster.to_numpy` contract `(*axes, band, y, x)`. No production
  conversion behavior changed.

## Retained flags and TODO proposals

- `test_model_io.py` calls `pytest.importorskip("torch")` at module scope,
  after defining NumPy-only tests. On a torch-free installation pytest skips
  the entire module, including those native NumPy tests. TODO: give tensor
  tests their own optional-dependency boundary so the base geodata contract
  remains tested. Retained because these are live tests, not dead source.
- `test_crop.py::test_a_window_past_the_edge_is_filled_as_asked` requests
  offset `(0, 1)` and shape `(1, 1)` from the shared two-by-two raster: it
  never reaches past an edge. TODO: correct its name or inputs and check the
  claimed fill pixels. Negative-offset and larger edge/reflection windows
  elsewhere already exercise padding; this is an inaccurate test claim,
  not evidence that `window` is unused.
- The three dynamic strict xfails in `test_warp_edges.py` are intentional
  ODC lazy interpolation/averaging limitations, not abandoned stubs. They
  are attached only after grid/laziness/computation assertions succeed and
  restrict failure type to numerical `AssertionError`. Retain until the
  corresponding dependency behavior actually changes.

## Exhaustive coverage

| File | Full-read evidence and disposition |
| --- | --- |
| `tests/geodata/core/test_accessor.py` | Retain: shared delayed source computed once, band order and empty statistics schema. |
| `tests/geodata/core/test_anchor.py` | Retain: immutable span replacement keeps grid and original time. |
| `tests/geodata/core/test_axes.py` | Retain: projected/geographic CF axis identity. |
| `tests/geodata/core/test_base.py` | Retain: native registration/types, spans, formatting/geocoding, lazy decoding and foreign-axis refusal. |
| `tests/geodata/core/test_catalog.py` | Retain: vector vs Item accessor dispatch, native Item conversion and writer policy. |
| `tests/geodata/core/test_create.py` | Retain: explicit dimensions/references, exact coordinates, variable metadata, GCP placement and persistence. |
| `tests/geodata/core/test_model_io.py` | Correct stale axis wording/assertion; retain native NumPy/tensor conversion and read-only buffer checks; flag optional torch skip. |
| `tests/geodata/core/test_raster.py` | Retain: COG paths/templates, shared band attrs, statistics, metadata writes and removed writer arguments. |
| `tests/geodata/core/test_rgb.py` | Retain: channel identity, explicit recoloring, no mutation and invalid-channel checks. |
| `tests/geodata/core/test_stack.py` | Retain: group/grid/time/root metadata, independent persistence, mapping, warping and display contracts. |
| `tests/geodata/core/test_vector.py` | Remove old persistence helper only; retain native geometry/query/rasterize/vectorize semantics and spatial-only API. |
| `tests/geodata/transform/conftest.py` | Retain: placed packed/categorical/lazy rasters and grid extraction, used across transform tests. |
| `tests/geodata/transform/test_color.py` | Retain: class colors, unknown-code NaN, lazy pixels and scalar coordinates. |
| `tests/geodata/transform/test_combine.py` | Retain: exact band merge, temporal stacking, strict pixel semantics and mosaic overlap/nodata policy. |
| `tests/geodata/transform/test_crop.py` | Retain: CRS-aware geometry crops, masks, grids/time/type/laziness and padding; flag inaccurate far-edge test. |
| `tests/geodata/transform/test_nodata.py` | Retain: exact-representable fills, lazy mask/decode, source preservation and categorical packing refusal. |
| `tests/geodata/transform/test_stack_metadata.py` | Retain: shared root attrs survive concat/reproject rebuilds. |
| `tests/geodata/transform/test_temporal.py` | Retain: bucket metadata, nodata exclusion, categorical restrictions and elapsed-time interpolation. |
| `tests/geodata/transform/test_vector.py` | Retain: exact target coordinates, dtype support, connected valid geometry and geographic round trips. |
| `tests/geodata/transform/test_warp.py` | Retain: target grids/CRS, class kernels, explicit fills, align extents and native grid selection. |
| `tests/geodata/transform/test_warp_edges.py` | Retain: numerical Rasterio references, lazy/eager pixels, native resolutions and narrowly scoped dependency xfails. |
| `tests/geodata/test_conventions.py` | Retain: foreign spatial names normalize lazily, unsupported unlabeled axes unchanged, trees and clear failures. |
| `tests/geodata/conftest.py` | Retain: real raster/grid/scene/label fixtures, public COG layouts and native Item recovery used across packages. |

## Checks

- Provenance smoke before helper removal: **1 passed** in 0.09 s; after:
  **1 passed** in 0.14 s.
- Strengthened asymmetric NumPy-layout assertion: **1 passed** in 1.31 s.
- `ruff check` on all three edited tests passed. Scoped `git diff --check`
  passed for the helper removals; the root audit owns integrated validation.
- The attempted old-name pytest selection overlapped the test rename and
  exited with collection error; this was a validation-command timing issue,
  not a failing product assertion. The original symmetric assertion was
  subsequently reconstructed from HEAD and executed separately.
- Broad tests were not repeated here after the root audit confirmed the
  Zarr synchronization stall needed local runtime access. The root audit
  owns the final integrated suite under that runtime.

No source behavior or public API was changed by this additional test audit.
