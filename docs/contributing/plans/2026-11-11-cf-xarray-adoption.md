# Slim GeoSave onto CF, cf-xarray, xarray, ODC and STAC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Goal

Make GeoSave smaller by letting each dependency own what it already defines:
CF the structure of stored rasters, cf-xarray reading it, xarray the in-memory
objects, ODC grids and loading, STAC the catalog metadata. GeoSave keeps only
what none of them states, and removes the parallel mechanisms it built instead.

## Workstreams

| # | Workstream | Outcome | State |
|---|---|---|---|
| 1 | Time cells follow CF | a label's cell is two coordinates in memory, CF bounds in files | **done** |
| 2 | Loading follows STAC assets | one node per asset on its stated grid; a grid is always an explicit choice | **done** |
| 3 | CF structure read through cf-xarray | no GeoSave reader for what `.cf` reads; no leftover in-memory bounds machinery | **done** |
| 4 | `stac` owns STAC ↔ xarray | one writer per direction; RGB and viewport reads through STAC and ODC | partly |
| 5 | One spatial naming rule | `.gs.spatial_dims` everywhere; `to_yx` removed | open |
| 6 | Sampling on xarray | `GeoSamples` takes a loose tree, gives samples sharing one time | open |

```text
STAC Items ──stac.load_datatree──► loose tree (/collection/asset) ──GeoSamples──► samples (shared time)
     ▲          (ws 2, 4)                 │  .gs.group_grid / .gs.to_dataset(grid)       (ws 6)
     └────────── stac.publish ◄───────────┘  netCDF / Zarr: CF bounds at the boundary (ws 1)
```

### 1. Time cells follow CF (done)

```text
in memory                              netCDF / Zarr
start_datetime(time) ┐   encode  ──►   time:bounds = "time_bounds"
end_datetime(time)   ┘   ◄── decode    time_bounds(time, bounds), time's units
```

- A cell ends where the next starts (CF convention), in memory and in STAC.
- Writers of cells: an Item's own period, the local solar day of a
  `solar_day` mosaic, a composite's resample bin, `.gs.write_time_bounds(freq,
  label=, dim=)`. Selection is native `.sel`.
- `utils/cf.py: encode_time_bounds / decode_time_bounds / drop_coordinate_fill`
  at the netCDF/Zarr boundary; a raster stating its cells twice is refused.
- Removed: `select_time`, `time_bounds`, `cell_bounds`, `_read_coverage`,
  `read_time_bounds`, the DataTree ancestor search; `CRS_COORDINATE` →
  `GRID_MAPPING`. Composite is `composite_ds` / `composite_da`.
- Not agreed yet: `write_time_bounds(dim=)`; skipping empty `stac_properties`
  records when merging composites.

### 2. Loading follows STAC assets (done)

- `stac.load_datatree(items)`: one node per asset on the exact grid its
  `proj:*` states; Items disagreeing on an asset's grid are refused. A
  multi-band asset holds one variable per band. `tree.gs.group_grid()` merges
  assets on one grid into `r10m`-style nodes.
- `stac.load_dataset(items, geobox)`: the grid is required; region, crs,
  resolution, anchor, `mask_and_scale` and the time filter are gone (choosing
  Items is the search's job). `catalog.gs.to_dataset(geobox)` loads one
  collection; `catalog.gs.to_datatree()` loads per asset.
- `stac/item.py: read_item_period` is the one reader of an Item's period.

### 3. CF structure read through cf-xarray (done)

- **3.1** `Quantity` → `Description` with `band.gs.description`;
  `is_categorical` is `.cf.is_flag_variable` or a GDAL colour table
  (`colorinterp="palette"`, which states colours but no CF flags). Flags
  without `flag_meanings`, or a `color_map` alone, no longer count (CF).
- **3.2** Support variables are scalar CRS carriers only. Spatial bounds and
  cell measures from foreign files stay ordinary variables for `.cf`;
  `check_coordinate_change` and every dimensioned re-attach path are gone.

### 4. `stac` owns STAC ↔ xarray (partly)

Done: loading and publication live in `geosave.geodata.stac` (`load.py`,
`publish.py`); `io` holds files and storage. Remaining, each detailed when it
starts:

- raster → STAC time has one writer (today: publication,
  `geosave_flows/tasks/labels.py`, `create_stac_item`).
- `gs.rgb` is published as the render extension on the Collection and read
  back on load.
- `read_scene_pixels` is replaced by `stac.load_dataset(item, view,
  groupby="id", chunks=None, resampling="nearest", dtype="float32",
  nodata=nan)`: equal pixels, ODC not slower (local and HTTP benchmarks); band
  names follow ODC (`red`, not `B04`).
- `stac_geoparquet` and STAC-table detection move into `stac/`, so `io` never
  imports `stac`.

### 5. One spatial naming rule (open)

`to_yx` renames ODC's `latitude`/`longitude` to `y`/`x` only because eight
places assume `("y", "x")` (`dask_mapping`, rasterize, plot, `to_tensor`,
`warp`, ...). Those read `.gs.spatial_dims`; `to_yx` goes.

### 6. Sampling on xarray (open)

`GeoSamples` takes a loose DataTree (each collection its own time) and gives
samples whose rasters share one time and its cells; `GeoCatalog.sample` loads
then calls it. This also fixes the one failing test
(`test_catalog_examples_load_chips_and_restore_pixels`: a solar-day image whose
local day crosses UTC midnight) and `GeoCatalog` counting an Item's end as
inside it. Open: several scenes in one slot (reduce or keep).

## Not in this plan

- Packing visibility on decoded variables; `write_nodata(None)` /
  `write_scale_offset` vs encoding.
- Dataset properties returning `{variable: group}` dicts (`packing`, `legend`,
  `spectral`, `nodata`).
- Splitting `Legend` into CF flags + palette; validating `Description` against
  the CF standard-name table; CF climatology bounds (§7.4).
- Time-less bands broadcast along time by composite.

## Global Constraints

- `geodata` imports no torch; no new dependencies.
- Pixels stay lazy in every changed operation; tests assert dask arrays are not computed.
- No compatibility aliases or shims (Alpha). Slimmer by removing cases, not by
  adding helpers or workarounds.
- Follow AGENTS.md Function Design: one responsibility per function, no
  structured dict inputs/outputs, no line-relocating private helpers.

---

## Detailed tasks: workstream 3

#### Task 3.1: Description and flags read through cf-xarray

**Files:**

- Rename: `src/geosave/geodata/attrs/groups/quantity.py` → `description.py`; update `attrs/__init__.py`
- Modify: `src/geosave/geodata/core/raster/dataarray.py:395-430` (`description`; `is_categorical` via `.cf.is_flag_variable` + GDAL palette)
- Modify docs: `docs/basics/attrs.md`, `docs/concepts/metadata.md` group table
- Test: `tests/geodata/attrs/groups/test_description.py`, `tests/geodata/core/raster/test_dataarray.py`

- [x] **Step 1: Write the failing tests**

```python
def test_description_reads_cf_chapter_3_attrs():
    found = scene["red"].gs.description
    assert (found.long_name, found.standard_name, found.units) == ("Red reflectance", "surface_bidirectional_reflectance", "1")

def test_flags_mark_a_band_categorical_through_cf_xarray():
    assert scene["scl"].gs.is_categorical
    assert not scene["red"].gs.is_categorical
```

- [x] **Step 2: Run** `uv run pytest tests/geodata/attrs tests/geodata/core/raster/test_dataarray.py -v` — expect FAIL.
- [x] **Step 3: Implement** the rename, the property and the `is_categorical` change.
- [x] **Step 4: Run the suite:** `uv run pytest`, `uv run pytest tests/flows -v`, `uv run ruff check .` — expect PASS. `grep -rn Quantity src tests docs/reference docs/basics` prints nothing.
- [x] **Step 5: Commit** `refactor: read description and flags through cf-xarray`

#### Task 3.2: In-memory CF bounds machinery removed (done)

Foreign spatial bounds and cell measures are not GeoSave's concern; users read
them through `.cf`. `read_support_variables` names CRS carriers only;
`check_coordinate_change` is deleted with its 5 callers; the re-attach loops
in `composite`, `write_grid`, `reproject` and `read_pixels`, the bounds
stripping in `to_dataarray` and the GeoTIFF writer's bounds refusal are gone;
`_select_pixels` only sets carriers as coordinates. Five tests that only
exercised that handling were deleted.
