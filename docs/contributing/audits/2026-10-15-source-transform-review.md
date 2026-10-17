# STAC source and transform ownership review

Scope: live `stac/source.py`, all seven transform modules, their accessors,
reader integration, and focused tests. Notebooks were excluded. No transform
entry point was proven obsolete from the inspected callers and documentation.

## Integration and ownership

| Module | Integration | Decision |
| --- | --- | --- |
| `color` | DataArray colorization accessor | Keep the focused operation. |
| `combine` | Native file/Item assembly merges distinct variables | Keep exact-grid validation; this is not temporal concatenation. |
| `extent` | Raster crop accessors and sample crop/padding | Keep grid-changing operations together. Clear stale source encoding after padding. |
| `nodata` | Raster mask, unpack, and mask-and-scale accessors | Keep per-variable mapping; its native tree preservation differs from rebuilding changed grids. |
| `temporal` | Assembly uses concat; sample reading uses select; public resample/reduce/interpolate are documented workflows | Split tree concatenation and validation from the public operation. Enforce corresponding-variable dtype and time-coordinate contracts. |
| `vector` | GeoVector rasterize and DataArray vectorize accessors | Existing validation, polygonization, and burning helpers each own concrete steps. No deletion justified. |
| `warp` | Raster accessors, DataTree alignment, sample reading, and file assembly | Keep target resolution, kernel validation, and pixel warping distinct. Resolve chunk-boundary behavior before reorganizing further. |

## Source simplification

ODC execution stays in `StacSource`. Native file reading stays in `io`.
Property-coordinate restoration is now separate from ODC request preparation
and loading. Both interval coordinates are assigned together. Preparing mixed
instant/composite Items uses native cloning and explicit datetime fallback,
without a serialization/deserialization round trip. ISO8601 parsing accepts
STAC interval strings without requiring identical formatting.

Metadata agreement already uses `attrs.intersect` and band translation already
uses `band.read_attrs`. Replacing the former with `attrs.combine_attrs` would
change the source policy from warn-and-drop to rejection for certain conflicts.
Native reading and ODC loading have different metadata precedence and alias
contracts, so their complete readers were not merged.

ODC's installed public `load` accepts native Items and parses them internally;
it does not accept the parsed objects used by GeoSave's metadata resolution.
Removing the second metadata parse would require a different integration seam,
not deleting a redundant call. No private ODC loader path was introduced.

## Behavior and validation

- Temporal concat now rejects missing time coordinates and incompatible input
  dtypes. Scalar times and empty time axes remain valid. Different variables
  within one Dataset may still have different dtypes.
- Padding clears the changed object's source-file encoding without mutating its
  input; zero padding keeps the original object.
- Corrected mask documentation: an explicit fill replaces nodata metadata.
- Narrowed two already-validated CRS types at transform call sites; no new
  runtime validation policy was added.
- Four concat/padding regression cases failed before the changes. Added a mixed
  instant/composite source load smoke covering laziness and Item immutability.

## Open work

- STAC multi-resolution reference-grid selection remains an explicit TODO.
- Three strict expected failures in `test_warp_edges.py` document chunked
  bilinear/cubic interpolation and average results differing from full-grid
  reference values. Do not call transform numerically complete while these remain.
- Categorical guards in temporal reduction currently inspect the reduced result.
  Moving the guard earlier should avoid reliance on private xarray resampler state.
- No new generic operation dispatcher or wrapper classes are warranted by this
  review. Public function naming is already verb-based; internal readers and
  validators should be renamed only when their responsibility changes.

## Checkout limitation

A broad `tests/geodata/core` collection failed on two pre-existing staged files:
`test_dataset.py` and `test_datatree.py`. Comparison showed duplicates of their
`core/raster/` counterparts, differing only in obsolete imports of `core.create`,
`core.datatree`, and `core.conftest`. They were left untouched because they are
staged changes outside this pass. Verification uses the canonical raster/vector
paths plus the core anchor tests, not those duplicate copies.

Final checks: 1,278 passed, 25 deselected, 3 expected failures across transform,
STAC, I/O, canonical core raster/vector and anchor tests, sampling, model/spec,
and layering. Focused BasedPyright reported zero errors/warnings. Ruff lint and
format, generated API freshness, and whitespace checks passed. The three
expected warp failures and the obsolete staged test copies remain open.

Subsequent approved simplification: the source interval-restoration approach
above was removed. StacSource now delegates dates, grouping, and property
representation entirely to ODC, then adds raster-band and Collection metadata.
There is no automatic interval reconstruction, property-coordinate promotion,
or search-limit squeezing. See the updated STAC guide for explicit publication.

## Transform follow-up: temporal validation and warp isolation

`temporal.reduce` now validates input class metadata before calling the reducer.
It reads metadata from the first group using public resampler iteration, without
private `_obj` access or a custom resampler wrapper. The actual reduction still
uses xarray's named methods and passes `keep_attrs=True` explicitly. This preserves
units, legends, and TimeSpec even under `xr.set_options(keep_attrs=False)`.
Four regression cases failed before the change and now pass: early categorical
rejection and lazy metadata preservation for both DataArray and Dataset.

The three warp expected failures were reproduced with `--runxfail` (77 passed,
3 failed). Direct ODC calls produce identical pixels to GeoSave for each failing
kernel. With odc-geo 0.5.1, rasterio 1.5.0 and xarray 2026.4.0, the 4x4 fixture
warped to EPSG:3857 showed the following maximum differences from eager ODC:

| Kernel | 2x2 source chunks | One 4x4 source chunk |
| --- | ---: | ---: |
| bilinear | 0.0643420 | 0.0000319481 |
| cubic | 0.0643420 | 0.0000319481 |
| average | 0.0000352859 | 0.0000352859 |

The installed ODC chunk planner selects intersecting source tiles without a
kernel-dependent neighbour margin. That explains the interpolation-neighbour
risk, but chunk-local numerical differences remain even with one source chunk.
No automatic rechunking, eager full-raster fallback, private dependency patch,
or weaker assertion tolerance was added. The three strict xfails remain open;
this pass does not claim numerical parity for chunked warp.

STAC extension reader consolidation is recorded separately in TODO.txt, rather
than expanding the transform work back into another STAC refactor.

Validation: 58 temporal tests passed; the broader transform, core/raster,
io/raster, sampling and model/spec run passed 884 tests with the three existing
warp xfails (exit zero). Focused Ruff, formatting, BasedPyright, API-reference
freshness and whitespace checks passed. No public signatures changed.

## Transform follow-up: explicit alignment and module ownership

Alignment now requires a target and defaults to its exact extent (`match`).
Union and intersection remain explicit options. A reference DataTree must have
a shared grid; otherwise the caller selects a group. Reprojection to a bare CRS
still covers every source tree group. Native-resolution nesting remains an
explicit option and does not promise equal pixel grids.

Removed the public `find_common_grid` planner; a small private helper owns grid
planning inside warp. GeoDataTree delegates to `warp.align` instead of repeating
planning and reprojection. Removed `select_resampling`: its sole production
caller already had resolved variable kernels and now selects them directly.
Updated callers, API references, and grid documentation without compatibility
aliases.

Reviewed all seven transform modules and their accessor integration. Extent
crop/pad, nodata mask/unpack, variable merge, temporal operations, representation
conversions, and colorization retain distinct contracts. In particular, merge
requires aligned inputs rather than silently aligning, and vector conversion
validation does not warrant a generic grid dispatcher. Corrected the accessor
mask docstring to describe replacement of nodata metadata. No additional public
helper removal was justified by the current call paths.

Validation: the wider transform, core/raster, core/vector, sampling, io/raster,
and model/spec run passed 984 tests with three existing warp xfails. After adding
the in-place tree metadata/laziness regression, all eight selected alignment
tests passed. Focused BasedPyright reported zero errors, warnings, or notes;
Ruff lint/format and generated API freshness passed. Breaking changes are the
required alignment target, default match extent, rejection of ambiguous target
trees, and removal of the two public helpers. The existing ODC warp parity
limitations remain open.
