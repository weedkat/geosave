# Native Item opening and ODC replacement feasibility

Run the local, network-free probe:

```bash
.venv/bin/python docs/contributing/benchmarks/2026-11-06-native-open/feasibility.py
```

The script writes small real GeoTIFF assets and compares `io.open_item` with
`io.load_dataset` / `io.load_datatree`. It asserts the reported differences.
It uses the installed ODC dependencies; these are observations of GeoSave's
current wrappers, not claims that ODC cannot be configured or extended.

## Results

| Case | Native Item opening | Current ODC loading |
| --- | --- | --- |
| One aligned COG | Lazy uint16, scalar time | Lazy float32, length-one time dimension; equal grid and pixel values |
| Two 10 m assets with origins offset by 5 m | Two exact native grids | One output grid in `load_datatree` |
| Composite with nominal date and start/end | Keeps temporal bounds as coordinates | Keeps nominal date but omits temporal-bound coordinates |
| Timeless Item | No time dimension | Constructs a time dimension |

The dtype example intentionally supplies no raster-extension dtype declaration.
It demonstrates that moving existing Items between readers is not necessarily
dtype-preserving without additional options/metadata. It is not a universal
ODC dtype rule.

Existing integration tests also exercise native multi-date stacks, Collection
ordering, mixed CRS/resolution assets, pixel laziness, publication round trips,
and resource cleanup on read/assembly failures.

## Caller evidence and decision

`ModelSpec.prepare` calls `rows.gs.to_datatree()` and only then explicitly aligns
its selected Collection rasters. Samples may include optical observations and
a timeless label/DEM Collection. The catalog contract preserves native grids,
date bounds, and a Collection/grid layout. ODC loaders currently accept one
Collection per call and derive output grids.

**Complete replacement is not behavior-preserving today.** The native catalog
path is retained for this concrete caller and the existing catalog contract.
Replacing it would require decisions about grid changes, static time semantics,
temporal bounds, dtype policy and multi-Collection assembly. ModelSpec redesign
remains deferred.

Implemented simplification:

- `io.open_item(item)` is the public single-Item reader; `open_scene` is removed.
- `open_items` and `open_collection` are removed as public entry points.
- Their required catalog behavior is consolidated into one private
  `io.stac._open_catalog` operation, reached through `rows.gs.to_datatree()`.
- One ExitStack owns all opened Items across Collections. Returned trees retain
  those closers; failures release everything opened so far.
- No `open_stack`, compatibility aliases, implicit reprojection or ODC fallback.

For new explicit grid assembly, use ODC loaders. Native catalog conversion is
still available when original grids and temporal representation must survive.

## Verification

- STAC I/O, catalog, sampling and model-processing selection: **91 passed,
  2 deselected**, 16 warnings.
- ODC, catalog-row and STAC GeoParquet tests: **46 passed, 1 failed** initially.
  The failure was public-schema DNS access in the sandbox; the exact validation
  test passed on rerun with network access (**1 passed**, 1 warning).
- ModelSpec preparation: **5 passed, 29 deselected**, 1 warning.
- Four-case local feasibility script passed after the API rename.
- Targeted Ruff, generated API freshness and whitespace checks passed.
- Targeted type checking reports one unchanged error in `_open_asset`:
  `open_raster` can return a DataTree where the helper requires a Dataset.
  The same diagnostic reproduces from the HEAD version of that function.
- Independent scoped review found no important regressions.
- This is scoped verification, not a full-library suite claim.

## Performance

See the [performance benchmark](performance.md) for measured setup and compute
costs on equivalent aligned local TIFF inputs at two raster sizes.
