# Native Construction and Writers Implementation Plan

**Spec:** `docs/contributing/specs/2026-11-03-native-construction-and-writers.md`

**Global constraints:** edits stay uncommitted on main; no compatibility aliases;
every changed file passes `uv run ruff check` and `uv run basedpyright`; tests
build rasters through the public steps (conftest helpers may chain them).

### Task A: Native construction

- `GeoRaster.write_grid(geobox)` and `write_crs(crs)` on the shared base.
- `GeoDataset.to_datatree(collection)`, `GeoDataArray.to_datatree(collection)`.
- `tree.gs.geobox` reads the grid nodes.
- Delete `create_dataset`, `create_dataarray`, `create_datatree`; `name_grid`
  moves to `core/raster/grid.py`.
- Tests: `write_grid` places, refuses wrong dims or sizes, links `grid_mapping`;
  `to_datatree` names the node, moves attrs, inverts `to_dataset`; conftest
  `build_dataset` and `build_tree` replace the constructors in tests.

### Task B: Native tree mapping

- Tree branches of `extent`, `temporal.select`, `warp`, `concat` build on
  `map_over_datasets` (skipping empty nodes) or `from_dict`.
- Delete `map_groups`, `replace_rasters`, `_assemble_tree`, `align(inplace=)`.

### Task C: Folder writers

- `paths.expand_paths(raster, path_format=None, *, collection=None)` with
  `default_path_format`.
- `geotiff.write_files(files, writer, ...)` is the GeoTIFF batch writer.
- `zarr.write`, `netcdf.write` take a Dataset or a DataTree, check encoding
  names, defer with `compute=False`.
- `to_cog`, `to_gtiff`, `to_catalog` take `path_format`; `path` is a folder.
- Delete `writers.py`.

### Task D: One-path `open_raster`

- Delete `combine_rasters`, `open_raster`'s sequence, `geobox`, `resampling`.

### Task E: Docs and full checks

## Progress (2026-11-03)

- Tasks A–E done, uncommitted on main. Geodata, flows, model-spec tests pass; the full
  suite still has the 37 `ModelSpec` failures listed in the nested-trees plan.
- Rulings beyond the spec:
  - `expand_paths(raster, root, path_format=None, *, collection=None)` returns
    `{UPath: Dataset}` under a folder, so no separate join step exists. A `root`
    naming a `.tif` is refused, pointing to `path_format`.
  - `geotiff.write_files(files, writer=write_cog, ...)` is the GeoTIFF batch writer;
    `zarr.write` / `netcdf.write` take a Dataset or a DataTree and check through
    `io/raster/stores.py`. `compute=False` wraps a `partial`, which keeps lazy arrays
    out of the delayed graph's dependencies.
  - `ModelSpec.ingest` takes `path_format` and `prepare` uses the default layout.
  - A tree keeps a bare root; `tree.gs.geobox` reads the grid nodes. `chips.merge`,
    `rasterize` and the STAC loaders place pixels with `write_grid`.
  - `open_raster` refuses a list with a message pointing to the catalog.
- `create_catalog` and `core/ctor/` are removed. `from_items` moved to the public
  `io/vector/stac_geoparquet` module because `ModelSpec` and the flows, outside
  `geodata`, use it. `require_item_columns` words the missing-column error once.
- `write_grid` takes a grid only; dates are set when the object is built.
