# cf-xarray adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Slim GeoSave and keep its files CF-compliant: time cells stay plain `start_datetime`/`end_datetime` coordinates in memory, become CF bounds only in files, and everything else that reads CF structure goes through cf-xarray, xarray or ODC.

**Architecture:** In memory a time cell is two auxiliary coordinates on `time`'s dimensions. They survive `ds[[...]]`, travel with a DataArray, inherit through a DataTree, and map 1:1 to STAC. One encode/decode pair at the netCDF/Zarr boundary converts them to and from a CF §7.1 boundary variable. Cells are written by their owner: readers assign what their source states; composites and users state regular cells with `.gs.write_time_bounds(freq, label=)`.

**Tech Stack:** xarray 2026.4.0, cf-xarray 0.11.3, odc-stac 0.5.2, pandas 2.3.3, netCDF4, Zarr v3, pystac.

**Spec:** [CF structure reference](../reviews/2026-11-11-cf-structure-reference.md) (facts), CF 1.11 §7.1, and the 2026-11-11 design sessions. The `conventions` docstring, `GLOSSARY.md` and `docs/concepts/metadata.md` still describe in-memory `time_bounds`; Task 6 corrects them.

## State (2026-11-11, session 3)

Not committed before the 2026-11-11 checkpoint. Tasks 1–3 are done; 4–7 remain.

| Task | State | Changed from the text below |
|---|---|---|
| 1 `write_time_bounds(freq, label=)` | done | also `dim=` for a label coordinate that is not the CF time coordinate (not yet agreed) |
| 2 loader gives each label its Item's period | done | an Item's own period wins; a `solar_day` label covers its local solar day (ODC's offset, `int(lon / 15)` hours) |
| 3 composite states its bins | done | split into `transform/temporal.py: composite_ds` / `composite_da`; empty `stac_properties` records are skipped when merging |
| 4 CF bounds at the file boundary | not started | — |
| 5 remove the stale time mechanism | partly | `select_time` removed; selection is native `.sel` |
| 6 docs | partly | reference pages updated; `conventions` docstring, GLOSSARY and `metadata.md` still describe in-memory `time_bounds` |
| 7 Description and flags | not started | — |

Decided this session, outside the task list:

- Cells follow the CF convention in memory and in STAC: a cell ends where the next starts; no ±1 second conversion at the STAC boundary.
- `stac/item.py: read_item_period(item)` is the one reader of an Item's period.
- `stac.load_datatree` loads one node per STAC asset on the exact grid its `proj:*` states (Items disagreeing on an asset's grid are refused); `tree.gs.group_grid()` merges assets on one grid into `r10m`-style nodes. A multi-band asset holds one variable per band.
- STAC loading and publication live in `geosave.geodata.stac` (`load.py`, `publish.py`); `io` holds files and storage.
- `stac.load_dataset(items, geobox, ...)` always takes a grid; `region`, `crs`, `resolution`, `anchor`, `mask_and_scale` and the region time filter are removed (choosing Items is the search's job). `catalog.gs.to_dataset(grid)` loads one collection onto one grid.
- The RGB choice (`write_rgb`) is published as the render extension on the Collection.

## Structure

```text
memory (every operation)                     file (netCDF / Zarr)
time            (time) datetime64[ns]        time(time) int64   bounds="time_bounds"
start_datetime  (time) datetime64[ns]  ──►   time_bounds(time, bounds) int64
end_datetime    (time) datetime64[ns]  ◄──     units/calendar = time's (CF §7.1)
                         encode_time_bounds / decode_time_bounds
```

| Comparison (smoke 2026-11-11) | `time_bounds(time, bounds)` in memory | start/end coordinates |
|---|---|---|
| dims | adds `bounds: 2` | none added |
| `ds[["red"]]`, `ds["red"]` | cells lost | kept |
| isel / sel / concat / where | kept | kept |
| DataTree child `to_dataset(inherit="all_coords")` | kept | kept |
| units in file | must be synchronised by hand (Zarr v3) | encoder sets both |

## Who writes cells

| Source | Writes start/end with | Input |
|---|---|---|
| `io.load_dataset` / `load_datatree` | native `assign_coords` | each label's own Item |
| `stamp_item_time` (`stac/raster.py:85`) | native `assign_coords` | one Item |
| rasterio/GeoTIFF reader (`io/rasterio.py:207`) | native `assign_coords` | ACDD `time_coverage_*` tags |
| netCDF / Zarr readers | `decode_time_bounds` | the file's CF bounds (any name) |
| `Dataset.gs.composite` | `.gs.write_time_bounds(freq, label=pandas' label)` | resample bins |
| users with a regular product | `.gs.write_time_bounds("MS")` | labels + freq |

## What moves, what stays

| Concern | GeoSave today | After | Evidence |
|---|---|---|---|
| In-memory cells | start/end coords; `write_time_bounds(pairs)`, `time_bounds`, `_read_coverage`, `select_time`, paired-NaT and shared-pair cases (`base.py:52-205`) | start/end coords; `write_time_bounds(freq, label=)`; `timespan` reads coords | smoke `add_bounds.py` |
| CF bounds | second reader `cell_bounds` + `read_time_bounds`; never written | read and written only by `decode_time_bounds` / `encode_time_bounds` | smoke `bounds_codec.py` |
| Bounds in trees | ancestor search for `bounds` references (`datatree.py:365-380`) | removed: no `bounds` reference exists in memory; coords inherit natively | smoke `tree_roundtrip.py` |
| Composite cells | min/max of observed periods, NaT for empty bins (`temporal.py:73-122`) | resample bins | smoke: matches pandas for 13 freq/origin/offset cases |
| Loader periods | sorted by instant, paired by position (wrong for unsorted Items) | paired by each label's own Item | bug confirmed 2026-11-11 |
| Flags | `Legend.is_stated_in` + GDAL palette in `is_categorical` | `.cf.is_flag_variable` + GDAL palette | smoke |
| Description attrs | `Quantity` | `Description` + `DataArray.gs.description` | — |
| CRS / grid mapping | ODC | **stays with ODC**; `CRS_COORDINATE` → `GRID_MAPPING` | smoke |
| Spatial cells | GeoBox | **stays with GeoBox**; no `x_bnds`/`y_bnds` (no use case; `cf.add_bounds(["x","y"])` is exact if one appears) | smoke |
| Coordinate `_FillValue` | written on `x`/`y` | `_FillValue: None` on coordinate variables (CF §5) | smoke |

## Settled decisions

1. In memory a time cell is `start_datetime`/`end_datetime` on `time`'s dimensions (scalar for scalar time). No `time_bounds` variable and no `bounds` reference exist in memory.
2. Files carry CF bounds: `encode_time_bounds` writes `time_bounds(time, bounds)` with `time.attrs["bounds"]`; `decode_time_bounds` turns any referenced time boundary variable back into start/end. Only the netCDF and Zarr writers and readers call them, per Dataset or tree node. Neither is a `.gs` method.
3. Encoded `time` and `time_bounds` share `units`, `calendar` and `dtype`: `time`'s encoding when present, else `"nanoseconds since 1970-01-01"`, `"proleptic_gregorian"`, `int64`. `dtype` is required when the coordinates are dask arrays. xarray then omits `units`/`calendar` on the netCDF bounds variable, as CF §7.1 recommends; Zarr keeps them, equal.
4. `.gs.write_time_bounds(freq, *, label="left")` states each label's cell as `[label, label + freq]` (left) or `[label - freq, label]` (right). Labels must be on `freq`'s edges (`to_offset(freq).is_on_offset`); fixed-length freqs cannot be checked. Same behavior on Dataset and DataArray.
5. Loading writes cells only where each label is exactly one Item, from that Item; mosaicked labels get none. No sorting.
6. A composite's cell is its resample bin: `write_time_bounds(freq, label=pd.Grouper(freq=..., closed=..., label=..., origin=..., offset=...).label)`. Empty bins keep their bin; pixels stay NaN unless `interpolation=`. Input cells are replaced, never reduced. Paired NaT no longer exists.
7. Removed: `.gs.time_bounds`, `.gs.select_time`, `.gs.cell_bounds`, `_read_coverage`, `read_time_bounds`, the datatree ancestor search.
8. `.gs.timespan` reads start/end when present, else labels.
9. `Quantity` → `Description` (`long_name`, `standard_name`, `units`); `cell_methods` stays a plain attr written by composites. `is_categorical` uses `.cf.is_flag_variable` plus the GDAL palette check.

## Open before execution

- Composite structure for Dataset and DataArray: not decided (discussion 2026-11-11).

## Not in this plan

- Packing visibility on decoded variables; `write_nodata(None)` / `write_scale_offset` vs encoding.
- Dataset properties returning `{variable: group}` dicts (`packing`, `legend`, `spectral`, `nodata`).
- Splitting `Legend` into CF flags + palette; `Spectral`, `StacMetadata`, `Extent` naming; validating `Description` against the standard-name table.
- CF climatology bounds (§7.4) for multi-year seasonal composites.

## Follow-up work after this plan

| Group | Issue | Evidence |
|---|---|---|
| Sampling | `GeoCatalog.sample(freq="D")` cannot place a solar-day image whose local day crosses UTC midnight (`test_catalog_examples_load_chips_and_restore_pixels` fails) | failing test |
| | `GeoCatalog` counts an Item's end as inside it, so a June composite lands in July too (`sampling.py: assign_slots`, `select_range`) | reproduced run |
| | `GeoSamples` takes a loose tree and produces samples sharing one time and its cells; several scenes in one slot is open | outline agreed |
| STAC | raster → STAC time is written in three places (publication, `geosave_flows/tasks/labels.py`, `create_stac_item`) | code map |
| | render extension on the Collection from `gs.rgb`, read back on load | decided |
| | replace `read_scene_pixels` with `stac.load_dataset(item, view, groupby="id", chunks=None, resampling="nearest", dtype="float32", nodata=nan)`; the readers name bands differently (`B04` vs `red`) | local and HTTP benchmarks: equal pixels, ODC not slower |
| | move `stac_geoparquet` and STAC-table detection into `stac/` so `io` never imports `stac` | import map |
| Structure | remove `to_yx`; eight `("y", "x")` assumptions use `.gs.spatial_dims` | code map |
| | `_select_pixels` and the support-variable machinery; time-less bands broadcast by composite | undecided |

## Global Constraints

- `geodata` imports no torch; no new dependencies.
- Pixels stay lazy in every changed operation; tests assert dask arrays are not computed.
- No compatibility aliases or shims (Alpha).
- Written files: bounds name `time_bounds`, vertex dimension `bounds`; readers accept any name the `bounds` reference gives.
- Follow AGENTS.md Function Design: one responsibility per function, no structured dict inputs/outputs, no line-relocating private helpers. Readers assign start/end with native `assign_coords`; no shared pair-writing helper.

## Review Focus

1. Items given to `load_dataset` out of time order with `groupby="id"`: each label keeps its own Item's period.
2. A file opened by GeoSave readers keeps the `bounds` reference in `time.encoding` (`decode_coords="all"`): `decode_time_bounds` reads attrs and encoding.
3. Scalar time (`isel(time=0)`, one GeoTIFF): scalar start/end, `time_bounds(bounds)` in files, exact round trip.
4. Nanosecond labels round-trip exactly through netCDF and Zarr; dask-backed start/end encode.

---

### Task 1: Cells from a frequency

**Files:**

- Modify: `src/geosave/geodata/core/raster/base.py` (`RasterTime.write_time_bounds`, `timespan`)
- Modify: `src/geosave/geodata/stac/raster.py:89-96`, `src/geosave/geodata/io/rasterio.py:207-220`, `src/geosave/geodata/io/odc.py:321` (pairs → native `assign_coords`)
- Test: `tests/geodata/core/raster/test_base.py`

**Interfaces:**

- Produces: `.gs.write_time_bounds(freq: str, *, label: Literal["left", "right"] = "left") -> DataT` on Dataset and DataArray.

- [x] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("as_dataset", [True, False])
def test_monthly_labels_state_month_cells(as_dataset):
    raster = monthly_raster(["2025-01-01", "2025-02-01"], as_dataset)   # dask pixels
    dated = raster.gs.write_time_bounds("MS")
    assert dated.start_datetime.values.astype("datetime64[D]").tolist() == [date(2025, 1, 1), date(2025, 2, 1)]
    assert dated.end_datetime.values.astype("datetime64[D]").tolist() == [date(2025, 2, 1), date(2025, 3, 1)]
    assert dated.start_datetime.dims == ("time",)
    assert is_dask(dated)

def test_right_labels_state_the_cell_before_them():
    dated = weekly_raster(["2025-01-05", "2025-01-12"]).gs.write_time_bounds("W", label="right")
    assert dated.start_datetime.values[0] == np.datetime64("2024-12-29")

def test_scalar_time_states_scalar_cells():
    dated = monthly_raster(["2025-01-01"]).isel(time=0).gs.write_time_bounds("MS")
    assert dated.start_datetime.ndim == 0

def test_labels_off_the_frequency_are_refused():
    with pytest.raises(ValueError, match="not on a MS edge"):
        monthly_raster(["2025-01-10"]).gs.write_time_bounds("MS")

def test_timespan_reads_cells_then_labels():
    dated = monthly_raster(["2025-01-01", "2025-02-01"]).gs.write_time_bounds("MS")
    assert dated.gs.timespan == (datetime(2025, 1, 1), datetime(2025, 3, 1))
    assert dated.drop_vars(["start_datetime", "end_datetime"]).gs.timespan == (datetime(2025, 1, 1), datetime(2025, 2, 1))
```

Rewrite the pair-based tests in `test_base.py` (`test_write_time_bounds_*`, `:229-297`) to the freq form or delete them; shared-pair broadcasting and paired NaT are removed cases.

- [x] **Step 2: Run** `uv run pytest tests/geodata/core/raster/test_base.py -v` — expect FAIL.
- [x] **Step 3: Implement** `write_time_bounds(freq, label=)`; `timespan` reads `START_DATETIME`/`END_DATETIME` coords, else labels. Switch the three pair callers to `assign_coords` (scalar pair → scalar coords; loader arrays → `("time", values)`).
- [x] **Step 4: Run** `uv run pytest tests/geodata/core/raster/test_base.py tests/geodata/io -v` — expect PASS.
- [x] **Step 5: Commit** `refactor: state regular time cells from a frequency`

### Task 2: Loader pairs periods by label

**Files:**

- Modify: `src/geosave/geodata/io/odc.py:288-321` (`_write_composite_periods` → `_write_item_periods`)
- Test: `tests/geodata/io/test_odc.py`

- [x] **Step 1: Write the failing tests**

```python
def test_composite_items_keep_their_own_periods_in_any_order(observation):
    jan, feb = interval_items(observation)          # Jan 1–31, Feb 1–28, datetime=None
    loaded = io.load_dataset([feb, jan], geobox=grid, groupby="id")
    for label, start, end in zip(loaded.time.values, loaded.start_datetime.values, loaded.end_datetime.values):
        assert start == label                         # ODC labels a composite at its start
        assert end > start

def test_a_mosaicked_label_states_no_period(observation):
    loaded = io.load_dataset(two_items_one_day(observation), geobox=grid)
    assert "start_datetime" not in loaded.coords

def test_tree_periods_sit_on_the_collection_node(observation):
    tree = io.load_datatree(interval_items(observation), groupby="id")   # two grids
    assert "start_datetime" in tree["local"].to_dataset(inherit=False).coords
    assert "start_datetime" in tree.gs.datasets["local/r10m"].coords
```

- [x] **Step 2: Run** `uv run pytest tests/geodata/io/test_odc.py -v` — expect FAIL on the reversed order.
- [x] **Step 3: Implement.** Map each Item's `read_instant` to its `read_timespan`; look up each label's own instant; duplicate instants or a missing span → return `loaded` unchanged. No sorting.
- [x] **Step 4: Run** the same command — expect PASS.
- [x] **Step 5: Commit** `fix: pair each loaded label with its own Item's period`

### Task 3: Composite states its bins

**Files:**

- Modify: `src/geosave/geodata/core/raster/transform/temporal.py` (`composite`: drop observed-span and NaT logic; properties records unchanged)
- Modify: `src/geosave/geodata/core/raster/dataset.py:121-215`, `dataarray.py:72-145` (docstrings: bins, no NaT periods; accessors otherwise unchanged)
- Test: `tests/geodata/core/raster/test_temporal.py`

- [x] **Step 1: Rewrite the period tests as bin tests**

```python
# Input labels: 2025-01-10T10:30, 2025-01-20T10:30, 2025-02-15T10:30, 2025-03-01.
@pytest.mark.parametrize("options", [
    {"freq": "1MS"}, {"freq": "1ME"}, {"freq": "QE"}, {"freq": "W"},
    {"freq": "10D", "closed": "right", "label": "right"},
    {"freq": "10D", "offset": "2D"}, {"freq": "7D", "origin": "end"},
    {"freq": "36h", "offset": "6h"},
])
@pytest.mark.parametrize("as_array", [False, True])
def test_a_composite_cell_is_its_resample_bin(options, as_array):
    raster = scene["red"] if as_array else scene
    monthly = raster.gs.composite(method="mean", **options)
    expected = pandas_bins(scene.time, **options)       # pd.Grouper._get_time_bins, test-only
    assert list(zip(monthly.start_datetime.values, monthly.end_datetime.values)) == expected

def test_an_empty_bin_keeps_its_cell():
    # Jan and Mar observations, freq="MS": Feb pixels NaN, Feb cell is Feb 1 → Mar 1
def test_input_cells_are_replaced_not_reduced():
    # input with start/end; output cells equal the bins
def test_composite_stays_lazy():
```

Delete tests asserting observed spans, NaT periods and `select_time` (`test_temporal.py:173-310`, `:361`, `:376`).

- [x] **Step 2: Run** `uv run pytest tests/geodata/core/raster/test_temporal.py -v` — expect FAIL.
- [x] **Step 3: Implement.** After the reduction: `result.gs.write_time_bounds(freq, label=pd.Grouper(freq=freq, closed=closed, label=label, origin=origin, offset=offset).label)`.
- [x] **Step 4: Run** the test file — expect PASS.
- [x] **Step 5: Commit** `refactor: composites state their resample bins`

### Task 4: CF bounds at the file boundary

**Files:**

- Modify: `src/geosave/geodata/core/raster/utils/cf.py` (add `encode_time_bounds`, `decode_time_bounds`; delete `read_time_bounds`)
- Modify: `src/geosave/geodata/io/formats/netcdf.py` (`_write` ~172, `open_netcdf` ~236)
- Modify: `src/geosave/geodata/io/formats/zarr.py` (`_write_dataset` ~237, `_write_tree` ~255, `_read_in_written_order` ~345; merge with `fill_encoding`)
- Test: `tests/geodata/core/raster/test_cf_support.py`, `tests/geodata/io/formats/test_netcdf.py`, `tests/geodata/io/formats/test_zarr.py`

**Interfaces:**

```python
def encode_time_bounds(raster: xr.Dataset) -> xr.Dataset:
    """Replace start/end coordinates with a CF boundary variable before writing."""
def decode_time_bounds(raster: xr.Dataset) -> xr.Dataset:
    """Replace a CF time boundary variable with start/end coordinates after reading."""
```

Both writers also set `encoding["_FillValue"] = None` on `x`/`y`.

- [ ] **Step 1: Write the failing round-trip tests**

```python
@pytest.mark.parametrize("scalar", [False, True])
def test_netcdf_stores_cells_as_cf_bounds_and_reads_them_back(tmp_path, scalar):
    raster = dated_scene(nanosecond_labels=True)                  # dask start/end
    raster = raster.isel(time=0) if scalar else raster
    path = write_netcdf(raster, tmp_path / "a.nc")
    with netCDF4.Dataset(path) as raw:
        assert raw["time"].bounds == "time_bounds"
        assert "start_datetime" not in raw.variables
        assert getattr(raw["time_bounds"], "units", raw["time"].units) == raw["time"].units
        assert "_FillValue" not in raw["x"].ncattrs()
    back = open_netcdf(path)
    for name in ("time", "start_datetime", "end_datetime"):
        np.testing.assert_array_equal(back[name].values, raster[name].values)
    assert "time_bounds" not in back.variables and "bounds" not in back.dims

def test_zarr_stores_cells_as_cf_bounds_and_reads_them_back(tmp_path):
    # same; zarr.json of time_bounds has units/calendar equal to time's

def test_tree_cells_round_trip_on_the_collection_node(tmp_path):
    # load_datatree output → write_zarr / write_netcdf → open: child datasets carry start/end

def test_a_foreign_cf_file_decodes_to_start_and_end(tmp_path):
    foreign = scene.cf.add_bounds("time").rename(time_bounds="time_bnds", bounds="nv")
    foreign.time.attrs["bounds"] = "time_bnds"
    foreign.to_netcdf(tmp_path / "f.nc")
    back = open_netcdf(tmp_path / "f.nc")
    assert {"start_datetime", "end_datetime"} <= set(back.coords)
```

- [ ] **Step 2: Run** `uv run pytest tests/geodata/io/formats/test_netcdf.py tests/geodata/io/formats/test_zarr.py -k "bounds or cf" -v` — expect FAIL.
- [ ] **Step 3: Implement** per Settled decisions 2–3. Writers encode per Dataset or tree node; readers decode per node.
- [ ] **Step 4: Run** the format test files in full — expect PASS. Run outside the sandbox if Zarr stalls.
- [ ] **Step 5: Commit** `feat: store time cells as CF bounds in netCDF and Zarr`

### Task 5: Remove the stale time mechanism

**Files:**

- Modify: `src/geosave/geodata/core/raster/base.py` (delete `time_bounds`, `_read_coverage`; `select_time` already removed 2026-11-11)
- Modify: `src/geosave/geodata/core/raster/dataset.py:366-388` (delete `cell_bounds`, `_read_coverage`)
- Modify: `src/geosave/geodata/core/raster/datatree.py:365-380` (delete the ancestor bounds search)
- Modify: `src/geosave/geodata/io/formats/geotiff.py:589` (read start/end coords)
- Modify: `src/geosave/geodata/conventions.py` (`CRS_COORDINATE` → `GRID_MAPPING`), and its users (`datatree.py:458-507`, `dataset.py:1358-1444`, `stac/publish.py`)
- Test: `tests/geodata/core/raster/test_cf_support.py`, `test_datatree.py`, `test_dataset.py:870-929`, `tests/geodata/io/test_assets.py:79`, `test_publication.py:38`, `formats/test_geotiff.py:113,535`, `tests/geodata/test_conventions.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_publication_states_a_composite_period_from_its_bins(tmp_path):
    item = monthly_composite().gs.to_item(tmp_path)
    assert item.properties["start_datetime"].startswith("2025-01-01")
    assert item.properties["end_datetime"].startswith("2025-02-01")

def test_geotiff_period_round_trips_as_scalar_cells(tmp_path):
    cube = monthly_composite().isel(time=0)
    back = open_raster(write_cog(cube, tmp_path / "a.tif"))
    assert back.start_datetime.values == cube.start_datetime.values
```

Move remaining callers of `write_time_bounds(pairs)` / `.gs.time_bounds` in tests to the freq form or to `start_datetime`/`end_datetime`.

- [ ] **Step 2: Run** `uv run pytest tests/geodata -v` — expect FAIL where removed members are still used.
- [ ] **Step 3: Implement** the deletions and the rename.
- [ ] **Step 4: Verify nothing stale remains:** `grep -rnE "CRS_COORDINATE|select_time|cell_bounds|_read_coverage|read_time_bounds|gs\.time_bounds\b" src tests docs/reference docs/guides docs/concepts` prints nothing except the explorer's own `layer.select_time` (unrelated).
- [ ] **Step 5: Commit** `refactor: remove the stale time-cell readers`

### Task 6: Docs describe the settled structure

**Files:**

- Modify: `src/geosave/geodata/conventions.py` docstring, `GLOSSARY.md` (Raster structure), `docs/concepts/metadata.md:40-158`, `docs/reference/rasters.md`, `docs/reference/transforms.md`, `docs/reference/public-api.md`, `docs/guides/stac.md`; regenerate `docs/reference/generated/data.md` with `scripts/api_reference.py`
- Modify: `docs/contributing/reviews/2026-11-11-cf-structure-reference.md` (add the start/end comparison table above; mark in-memory `time_bounds` as the rejected option)

- [ ] **Step 1:** Show the Structure diagram and the "Who writes cells" table; `.gs.write_time_bounds("MS")` example; a selection example with native `sel` on start/end.
- [ ] **Step 2: Run** `uv run pytest tests/geodata/test_conventions.py --doctest-modules src/geosave/geodata/conventions.py` and the docs build check from `docs/contributing/documentation.md`.
- [ ] **Step 3: Commit** `docs: time cells as start/end coordinates, CF bounds in files`

### Task 7: Description and flags read through cf-xarray

**Files:**

- Rename: `src/geosave/geodata/attrs/groups/quantity.py` → `description.py`; update `attrs/__init__.py`
- Modify: `src/geosave/geodata/core/raster/dataarray.py:395-430` (`description`; `is_categorical` via `.cf.is_flag_variable` + GDAL palette)
- Modify docs: `docs/basics/attrs.md`, `docs/concepts/metadata.md` group table
- Test: `tests/geodata/attrs/groups/test_description.py`, `tests/geodata/core/raster/test_dataarray.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_description_reads_cf_chapter_3_attrs():
    found = scene["red"].gs.description
    assert (found.long_name, found.standard_name, found.units) == ("Red reflectance", "surface_bidirectional_reflectance", "1")

def test_flags_mark_a_band_categorical_through_cf_xarray():
    assert scene["scl"].gs.is_categorical
    assert not scene["red"].gs.is_categorical
```

- [ ] **Step 2: Run** `uv run pytest tests/geodata/attrs tests/geodata/core/raster/test_dataarray.py -v` — expect FAIL.
- [ ] **Step 3: Implement** the rename, the property and the `is_categorical` change.
- [ ] **Step 4: Run the suite:** `uv run pytest`, `uv run pytest tests/flows -v`, `uv run ruff check .` — expect PASS. `grep -rn Quantity src tests docs/reference docs/basics` prints nothing.
- [ ] **Step 5: Commit** `refactor: read description and flags through cf-xarray`
