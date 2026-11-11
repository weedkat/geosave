# Raster structure cleanup

Transform ownership is superseded by [native transforms](2026-11-10-native-transforms.md). The implementation now uses concrete accessors and native xarray operations. Historical evidence below is retained.

Ownership between `core/raster/` accessors, `core/raster/utils/`, `transform/`
and `io/` grew ad hoc. This plan fixes ownership, naming and type dispatch
before Chips caching and ModelSpec work continue.

## Rules

- **Accessors** (`core/raster/{base,dataarray,dataset,datatree}.py`) read and
  write metadata on one object and delegate. They are the only place that
  dispatches on xarray type: a DataTree maps over its datasets.
- **`core/raster/utils/`** holds the internals behind accessor methods that
  change no pixels (grid placement, expansion, statistics, tensor conversion).
  **`transform/`** holds explicit operations that change pixels, extent, grid or
  time. Helpers used by one operation stay private in its module.
- **Types.** A utils or transform function is either generic over
  `(xr.DataArray, xr.Dataset)` with one body and no type branches, or takes one
  concrete type. Where the two types differ, the function calls the accessor
  method that owns the difference (for example `gs.write_extent`). Never
  `DataTree`, never `isinstance` on xarray types.
- **Returns.** Fixed sequences are tuples.
- **Names.** An implementation uses the public method's name. A function is
  named for what it does, and a variable for the value it holds. STAC terms
  (Item, Asset) replace invented ones ("inventory", "labels").
- **Validation.** Structural requirements go through `gs.validate`, never
  through hand-rolled `isinstance(geobox, GeoBox)` checks.

## Steps

### 1. `validate(affine=True)` (done)

Add an `affine` flag to every accessor's `validate`: pixels sit on one affine
`GeoBox`, not on ground control points and not unplaced. Replace the 11
hand-rolled checks (chips, warp, vector, combine, extent, dataset.read_pixels,
datatree, utils/grid) with `validate(affine=True)` or
`validate(georeferenced=True, affine=True)`, followed by a read of `gs.geobox`.
Every structural failure then raises `RasterValidationError`.

### 2. Grid writes named as their public methods (done)

`utils/grid.py`'s `place_on_grid`/`write_reference` become `write_grid`/
`write_crs`, generic over `(DataArray, Dataset)`. `gs.grid_mapping_coords` names
every CRS-carrying coordinate, replacing `_is_grid_mapping` and its inline copy
in `read_pixels`. Delegates stay per accessor: `GeoRaster` also bases DataTree,
whose `to_tensor` returns one tensor per node.

### 3. No implicit target for `align` (done)

`tree.gs.align()` defaults to the finest node grid, the choice
`_pick_finest_grid` already makes. This replaces the copies of the
`geobox is None` guard + `select_nested_grid` + `align(how=...)` in
`model/spec/model.py` and `viz/explore/page/table.py`; `select_nested_grid`
goes. Move
`Dataset.read_pixels`'s resampling body into warp.

### 4a. Extent, nodata and Chips without type branches (done)

`select_grid` is now `crop_to_grid`. `crop`, `pad` and `crop_to_grid` are
generic over `(DataArray, Dataset)` without branches; `gs.clip_extent`, backed by
`Extent.clip`, narrows declared extents for either type. `nodata.mask`/`to_nan`/
`unpack` take one variable, with `read_mask` validating a mask against a raster;
each accessor applies them through its own `_map_variables`, and DataTree
overrides `crop` and `mask` to map over its nodes. Chips uses `Padding | "drop"`.

### 4b. DataTree as a native container (done)

A DataTree is a hierarchy of Datasets: siblings may sit on different grids, and
coordinates on a node are inherited by its descendants, which xarray requires
to align exactly. GeoSave follows that contract instead of treating a tree as a
raster.

1. **Loaders hoist shared coordinates.** `io/odc.load_datatree` and `io/stac`
   move each collection's shared non-spatial coordinates (`time`, time bounds,
   `stac_properties`) onto the collection node; grid nodes keep `y`, `x`,
   `band` and `spatial_ref`. xarray then refuses a grid node whose time differs.
2. **Node-wise operations.** `tree.gs.crop`, `to_nan`, `unpack` and
   `mask_and_scale` map over raster nodes, as `tree.isel` does natively.
   `tree.gs.mask` (one mask cannot fit several grids) and `tree.gs.reproject`
   go.
3. **`to_dataset(grid=...)` replaces `align`.** `grid` is a node path relative
   to the called node, or a GeoBox; every node is reprojected onto it, then
   merged. Kernels are keyed by variable name. Without `grid`, nodes must
   already share one grid. `tree.gs.align` and `warp.align` go, with warp's
   tree branch, grid planning (`extent=`, `"native"`) and path-keyed kernels;
   single-raster `reproject` keeps GeoBox and CRS targets.
4. **`to_catalog` on any node.** The called node is one Collection: from
   `collection=`, its `stac_collection` metadata, or its name. Every data node
   below it becomes asset material, at any depth. It checks:
   - affine, georeferenced placement per node;
   - `same_footprint`: each node's footprint (declared extent, else grid
     extent) lies within the others grown by one coarsest pixel;
   - no data node defines its own `time`, so observations are inherited and
     agree by construction.
   `same_observations` and the ancestor-attrs walk go.
5. **No tree branches in transforms.** `temporal.select`/`concat`,
   `vector.rasterize(like=...)` and `conventions.to_yx` take rasters; trees use
   `map_over_datasets`.

### 5. Split rasters natively; io owns publication (done)

`DataArray.gs.expand_rasters(dim)` returns `tuple[DataArray, ...]` and
`Dataset.gs.expand_rasters(dim)` returns `tuple[Dataset, ...]`: one lazy slice
per position along `dim`, taken with `isel`. Each slice carries its labels as
scalar coords; a dim without coords gets its position as one. Absent dims
raise. `DataTree.gs.expand_rasters` goes; `tree.gs.datasets` already yields
`{path: Dataset}`.

`core/raster/utils/expansion.py` goes: `select_dims` (None → time default,
time/band exemptions), `group_dimensions`, `read_labels` and `expand_assets`.
The `split_vars` flag goes; io iterates `data_vars` when it splits variables.
Accessor writers (`to_item`, `to_catalog`, `to_cog`, `to_gtiff`) become
delegates passing the raster to io, which owns the Item/asset split, reads
template values from each slice's scalar coords, plans files and checks path
collisions in one place. `read_asset_labels`, the `"var"` alias and the `"."`
path repair go; `prepare_raster` names bands itself. Add Item and Asset to
`GLOSSARY.md`.

Before settling, smoke-test: static and time-series variables side by side,
nested tree paths, and repeated labels reaching the collision check.

### 6. Grid node names (done)

`name_grid` becomes `format_resolution` in `io/paths.py`; io is its only user. `name_grids` goes:
only `io/stac.py` can see two grids at one pixel size, and it numbers them
inline in asset order. `io/odc.py` keys families by pixel size, so it cannot
collide.

### 4c. Item metadata after native loading (done)

`load_datatree` is ODC only, so metadata the GDAL opener read from file tags
must come from the Item. Tracked by strict `xfail` tests:

- **Legend:** `create_stac_item(with_raster=True)` writes no
  `classification:classes`; `stac/extensions/classification.write_band` has no
  caller. Read the legend from each band's tags, as nodata and packing are
  read, and write the classes onto its raster band; `read_attrs` already
  restores them on load.
- **Legend (done).** `io/rasterio.read_legend(src, index)` reads a band's
  legend from its CF tags and colour table, and both `_attrs` and
  `create_stac_item(with_raster=True)` call it. `classification.build_classes`
  and `write_band` take a `Legend`.
- **Time period (done).** A composite label covers a period, but xarray's `time`
  holds a point. `resample` records nothing, its label is one bin edge (it can
  fall outside the bin: `"ME"` with `label="left"` labels June 05-31), and it
  drops every non-index coordinate on `time`. CF `time_bnds (time, bnds)` is a
  NetCDF file convention that only climate tools read; `ds["red"]` drops it,
  because `bnds` is not one of the variable's dims.
  1. The period is two 1-D coords, `start_datetime (time)` and
     `end_datetime (time)`, on DataArrays and Datasets alike. They slice with
     `time`, survive `ds["red"]`, use the STAC names and round-trip through
     NetCDF and Zarr unchanged. `time_bnds`, `write_time_bounds`/`time_bounds`
     on both accessors and the DataArray/Dataset split go.
  2. `TimeSpec` goes. A composite label covers the observed span of its
     scenes: the minimum of their `start_datetime` and the maximum of their
     `end_datetime`, computed with xarray's own resample over the same
     buckets, skipping NaT. An instant's start and end are its time. Every
     caller writes `reduce(resample(x, freq), method)`, so they become one
     call, `temporal.composite(data, freq, method, ..., properties=...)`,
     which holds the scenes it needs. `reduce`'s single-bin path has no caller
     and goes. An empty bucket keeps NaN pixels and a NaT period, and a later
     composite skips it, so an empty month does not erase its quarter.
  3. Resampling rebuilds `stac_properties` per label from its scenes, with an
     option: `"intersect"` keeps the keys every scene states alike;
     `"stack"` keeps every key as a list with one entry per scene, None where a
     scene lacks it. Open: publishing copies stacked lists into Item
     properties as they are, and cannot tell a stacked list from a list-valued
     property such as `instruments`.
  4. The ODC load fills both coords from each Item's
     `start_datetime`/`end_datetime`.
  5. Writers read the two coords: STAC properties and GeoTIFF
     `time_coverage_*` as now. `to_netcdf`/`to_zarr(time_bounds=True)` also
     write CF `time_bnds`; opening a file with CF bounds converts them to the
     two coords. `conventions.py` owns both conversions.

  Smoke-tested in a prototype (lazy pixels; intersect and stack; `"ME"` with
  `label="left"`; monthly → quarterly; `ds["red"]`; `isel`; concat; NetCDF and
  Zarr with and without CF bounds; tree hoisting). A tree node inherits only
  indexed coordinates by default, so the hoisted period is read through
  `to_dataset(inherit="all_coords")`, as `tree.gs.datasets` already does;
  native `map_over_datasets` does not see it. Load → publish is tested once
  publication reads the two coords.
- **Declared extent.** A raster carries two boundaries: the grid (GeoBox,
  published as `proj:shape`/`proj:transform`) and the declared extent, its
  area of interest. `load_dataset(region=...)` declares the region and loads
  the whole grid around it, unmasked. Reading file tags per band is dropped
  with the GDAL opener. Publishing writes the region as the Item `geometry` and
  the grid as `proj:*`; `catalog.gs.to_datatree()` and
  `row.gs.to_datatree()` read the Items' geometry back as the region, beside
  the whole grid. `write_extent(None)` publishes the whole grid instead.

### 7. One mapping intersection (done)

Seven hand-written merges keep the keys every mapping states alike, or refuse
any difference. `utils/mapping.py` gets `intersect_mappings(mappings) -> dict`
and `stack_mappings(mappings) -> dict[str, list]` (step 4c's `"stack"`).
Callers that warn compute dropped keys as the union of keys minus the result.
Attr groups fold to one validated value per group before intersecting and
unfold after, so a group is kept whole or not at all.

| Site | Today | Becomes |
| --- | --- | --- |
| `attrs/combine._intersect_attrs` | private `merge_attrs("drop_conflicts")`, then removes keys some inputs lack | fold groups → `intersect_mappings` → unfold |
| `stac/raster.build_stac_metadata` | hand loop over Item properties | `intersect_mappings` |
| `stac/band.read_shared_attrs` | `attrs.intersect`, warns with each Item's value | unchanged caller of `attrs.intersect` |
| `io/stac._read_observation`, `publish_items` | `attrs.intersect` over properties and labels | `intersect_mappings` (plain mappings, no groups) |
| `core/raster/dataset.to_dataarray` | `attrs.intersect` over band attrs | unchanged caller |
| `io/odc._build_collection_tree` | hand loop with `.equals` over coords | `intersect_mappings` with xarray equality |
| `core/vector/catalog` footer merge, `stac/catalog.merge_collections`, `io/odc` pixel sizes by band | refuse differing values | stay strict; share one helper only if their messages allow |

Done: `intersect_mappings` compares arrays element-wise, NaN as equal and
xarray objects with `equals`; `attrs` folds each group into one value keyed by
its class, so `merge_attrs` is no longer imported. The strict merges stay as
they are.

Related, not mappings: `spatial_dims` and `geobox` on the Dataset and DataTree
accessors repeat one "one shared value, else None" loop (four copies).

## Out of scope

`model/processing.to_tensor` duplicates `gs.to_tensor`; it moves with the
ModelSpec rework. Output memory in `Chips.merge` (C5) and sample caching (C7b)
are unchanged.
