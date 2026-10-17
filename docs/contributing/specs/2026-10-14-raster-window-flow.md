# Explicit row reading and alignment

Status: accepted API direction; implemented in the row reader.

## Goal

A GeoDataFrame row describes sources and selections. Its `gs.to_xarray()`
reader returns native xarray objects. Alignment is an explicit reader option,
using the existing transform implementation. No new pipeline object or public
`select_window` function. ModelSpec design remains outside this change.

```python
rasters = row.gs.to_xarray()
aligned = row.gs.to_xarray(
    target=reference,
    extent="match",
    resampling={"*": "nearest", "optical/red": "bilinear"},
)
optical = row.gs.to_xarray(
    collections=["optical"], chunks={"x": 256, "y": 256}
)["optical"].dataset
```

## Contracts

- Source and frame rows retain native group grids unless `target` is supplied.
  Native reads assemble each collection's files; they must be compatible with
  the existing assembly contract. A collection cannot implicitly choose a grid
  for differently gridded bands or mosaic same-date duplicate bands.
- `target`, `extent`, `resolution`, and `resampling` delegate to existing
  alignment. Reader extent defaults to `"match"`, so an explicit reference is
  adopted rather than expanded. Source alignment options require `target`.
- A chip already defines its target. The reader preserves that grid, temporal
  selection, and boundary padding, slicing same-lattice sources and warping
  others through existing assembly/warp primitives. A conflicting target,
  non-match extent, or resolution override raises an error.
- Resampling is a single kernel or a per-variable mapping. DataTree keys are
  `collection/variable`; Dataset keys are variable names. `"*"` is a fallback.
  Nearest is used when a target requires alignment and no kernel is supplied.
- An Item row's `to_xarray()` returns a Dataset. A sample row's `to_xarray()`
  returns a DataTree, even for one collection. `collections` selects DataTree
  groups in caller order before files open; select a node's `.dataset` to
  work with one group's raster.
- `chunks` forwards to each file opener: mapping, integer, `"auto"`, or None.
  The default empty mapping retains lazy native Dask chunks.
- Fixed string choices use Literal-based types shared with alignment. Collection
  names remain strings because they come from the user's data.
- Reading does not collapse differing collection observation dates. Frames
  retain their existing source selection and repeated observations.

## Flow

Source catalog -> grouped row -> DataTree -> explicit preprocessing -> prepared
rasters/catalog -> frame/chip rows -> DataTree -> tensors -> predictions.

Preparing and persisting aligned rasters is optional. Raw source assets can be
read directly onto a chip's declared grid when the processing permits it.
Alignment and preprocessing are not automatically reordered.

## Ownership

- `core/sample.py`: collection selection, row semantics, alignment arguments,
  temporal repetition, and chip padding.
- `stac/load.py`: native asset opening, STAC metadata, chunk forwarding.
- `io/raster/assemble.py`: existing same-grid slices, warp, band/time assembly;
  accept per-variable kernels in addition to a single kernel.
- `transform/warp.py`: existing grid selection and resampling; no second policy.
- Existing ModelSpec preparation explicitly requests the row's grid to retain
  its previous behavior. This does not redesign ModelSpec.

## Verification

Real COG fixtures cover mixed 10 m/20 m groups, native reads, equivalence to
explicit DataTree alignment, collection selection with broken unselected hrefs,
chunk forwarding, extent/native-resolution options, mixed-grid chip resampling,
padding, and conflicting chip targets. Existing row, cuts, assembly, STAC load,
and warp tests exercise the connected path.

## Deferred improvements

Stable parent IDs, frame metadata on reconstructed predictions, source mosaic
assembly and intersects membership remain separate changes. The planning module
is now `geosave.geodata.sampling` (formerly `cuts`). This API change does not claim to implement them.

## Same-collection series

The callable was renamed to `group_series()` on 2026-10-15; the description
below records the original implementation name.

`catalog.gs.series()` now groups Items by collection and normalized WGS84
footprint. It preserves complete source Items in the existing `sources` JSON,
orders them chronologically, and records their date span. Duplicate instants
and repeated Item IDs are rejected. A singleton undated Item is supported;
multiple undated Items or a mixture of dated and undated Items are ambiguous.
The earliest observation supplies the reference grid; no raster is opened.

The resulting rows use the existing Dataset/DataTree readers, frames, chips,
and GeoParquet persistence. Explorer recognizes grouped rows and opens them
through the row reader. Its Item-table path delegates grouping to `series()`;
its date filter uses interval overlap for series rows.

Catalog operations now live in `geodata/sampling`; the core accessor remains
a thin entry point. No compatibility package remains at `geodata/windows`.
