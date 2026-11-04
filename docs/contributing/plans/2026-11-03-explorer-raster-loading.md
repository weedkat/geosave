# Explorer Raster Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the explorer's `read_viewport` with overview-aware `read_pixels` in geodata, a zoom-level cache on raster layers, and split catalog checkboxes into polygon and raster columns.

**Architecture:** geodata owns reading a view: a catalog row reads each asset from its overview through `open_raster(overview_level=)`, a Dataset crops and strides; both use odc-geo's `compute_reproject_roi`. The map builds one Web Mercator `GeoBox` per redraw from RangeXY and PlotSize; each raster layer caches one read per zoom level. `CatalogLayer` folds into `VectorLayer`.

**Tech Stack:** xarray, rioxarray, rasterio, odc-geo, HoloViews/GeoViews, Panel, pytest.

**Spec:** `docs/contributing/specs/2026-11-03-explorer-raster-loading.md`

## Global Constraints

- `geodata` imports no viz code; viz calls only public accessors (`.gs.read_pixels`, `.gs.to_xarray`).
- View reads never use Dask: `open_raster(..., chunks=None)`.
- Overview choice comes from `odc.geo.overlap.compute_reproject_roi(...).read_shrink`; do not import `odc.loader._reader`.
- Web Mercator zoom pixel size: `156543.03392804097 / 2**zoom` metres.
- Cache padding: half the view's width and height on each side.
- Pixels returned to the map: float32, nodata as NaN, on the requested GeoBox's grid.
- Names: functions are verb + object; variables name the value held; no comprehension with more than one `for`; loops written out.
- Every changed file passes `uv run ruff check` and `uv run basedpyright`.

## Review Focus

- A view that misses the raster entirely: `read_pixels` returns NaN on the view grid and opens no asset.
- A file without overviews: `read_pixels` reads the base image window, never fails.
- A requested variable held by a multi-band asset: only that asset opens, the right band is returned.
- A cached layer whose date changes while the view is inside the cache: it reads again.
- A frame without raster assets: no raster column, and `raster_rows` stays empty.

---

### Task 1: Read a catalog row's pixels for a view

**Files:**

- Modify: `src/geosave/geodata/io/raster/gdal.py` (add `read_overview_factors`)
- Modify: `src/geosave/geodata/io/raster/stac.py` (`_open_asset` takes `**options`; add `read_pixels`)
- Modify: `src/geosave/geodata/core/vector/item.py` (add `GeoCatalogItem.read_pixels`)
- Test: `tests/geodata/io/raster/test_stac.py`, `tests/geodata/core/vector/test_item.py`

**Interfaces:**

- Produces: `gdal.read_overview_factors(source: Location) -> list[int]` (rasterio `overviews(1)`, opened through `hf_to_s3`)
- Produces: `stac.read_pixels(item: pystac.Item, geobox: GeoBox, variables: Sequence[str] | None = None) -> xr.Dataset`
- Produces: `GeoCatalogItem.read_pixels(self, geobox: GeoBox, *, variables: Sequence[str] | None = None) -> Dataset`

- [ ] **Step 1: Write the failing tests** in `tests/geodata/io/raster/test_stac.py`, using a COG written with `to_cog` (geosave writes overviews) of a 512×512 raster and its Item from `create_file_items`:

```python
def test_read_pixels_reads_the_overview_the_view_needs(cog_item, monkeypatch):
    levels = record_overview_levels(monkeypatch)        # wraps gdal.open_raster, appends options.get("overview_level")
    view = create_view(cog_item, pixel_size=4 * native)  # GeoBox over the raster in EPSG:3857
    pixels = read_pixels(cog_item, view)
    assert pixels.odc.geobox == view
    assert levels == [1]                                 # overview factor 4 is level index 1 (factors [2, 4, ...])
    assert pixels["band"].dtype == np.float32

def test_read_pixels_reads_the_base_image_when_zoomed_in(...)      # pixel_size = native -> levels == [None]
def test_read_pixels_outside_the_raster_opens_nothing(...)         # view far away -> all NaN, levels == []
def test_read_pixels_opens_only_assets_holding_the_variables(...)  # two assets red/nir, variables=["nir"] -> one open, data_vars == ["nir"]
def test_read_pixels_turns_nodata_into_nan(...)                    # nodata pixels -> NaN
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/geodata/io/raster/test_stac.py -k read_pixels -v`
Expected: FAIL, `read_pixels` not defined.

- [ ] **Step 3: Implement.** In `stac.read_pixels`: for each asset from `item.select_data_assets` whose names from `item.read_variable_names` meet `variables`; open the asset's base image (`chunks=None`, header only) and compute the roi on its `odc.geobox`; return an all-NaN Dataset on `geobox` when `roi_src` is empty; pick the overview as the last factor `<= roi.read_shrink`; `_open_asset(asset, names, key, opened, chunks=None, overview_level=level)`; compute the roi again on that level's grid; `isel` by `roi_src`, select the variables, `.gs.to_nan()`, `astype("float32")`, `.load()`, `xr_reproject(..., geobox, resampling="nearest", dst_nodata=np.nan)`; `merge` the files. Close every opened file before returning. `GeoCatalogItem.read_pixels` passes `self.to_item()`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/geodata/io/raster/test_stac.py tests/geodata/core/vector/test_item.py -v`
Expected: PASS

- [ ] **Step 5: Commit** `feat: read a catalog row's pixels for a view from its overviews`

---

### Task 2: Read a Dataset's pixels for a view

**Files:**

- Modify: `src/geosave/geodata/core/raster/dataset.py` (add `GeoDataset.read_pixels`)
- Test: `tests/geodata/core/raster/test_dataset.py`

**Interfaces:**

- Produces: `GeoDataset.read_pixels(self, geobox: GeoBox) -> Dataset`

- [ ] **Step 1: Write the failing tests**

```python
def test_read_pixels_samples_before_computing(scene):
    # 1024² band backed by an array that records the shape of each slice it serves
    served_shapes = []
    band = dask.array.from_array(RecordingArray(values, served_shapes), chunks=256)
    pixels = scene.assign(red=(("y", "x"), band)).gs.read_pixels(view_at_8x_native)
    assert pixels.odc.geobox == view_at_8x_native
    assert sum(shape[0] * shape[1] for shape in served_shapes) <= 128 * 128
def test_read_pixels_outside_returns_nan_without_computing(scene)
def test_read_pixels_keeps_band_identity(scene)          # data_vars equal, attrs (legend) kept
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/geodata/core/raster/test_dataset.py -k read_pixels -v`
Expected: FAIL

- [ ] **Step 3: Implement** with `compute_reproject_roi(self.geobox, geobox)`: empty roi → NaN Dataset on `geobox`; else `isel` by `roi_src` with step `roi.read_shrink`, `.gs.to_nan()`, float32, `.compute()`, `xr_reproject` onto `geobox`, keeping each variable's attrs.

- [ ] **Step 4: Run the tests** — Expected: PASS

- [ ] **Step 5: Commit** `feat: read a Dataset's pixels for a view`

---

### Task 3: Web Mercator zoom levels

**Files:**

- Modify: `src/geosave/geodata/viz/explore/mercator.py`
- Test: `tests/geodata/viz/explore/test_mercator.py`

**Interfaces:**

- Produces: `ZOOM_0_PIXEL_SIZE = 156543.03392804097`
- Produces: `find_zoom_level(view: GeoBox) -> int` — finest `z` whose pixel size is at least as fine as the view's coarser axis, clamped to `0..24`
- Produces: `create_zoom_grid(bounds: BoundingBox, zoom: int) -> GeoBox` — EPSG:3857 grid over `bounds` at the zoom's pixel size

- [ ] **Step 1: Write the failing tests**

```python
def test_find_zoom_level_takes_the_finer_level():
    assert find_zoom_level(view_with_pixel_size(100.0)) == 11   # z11 = 76.4 m, z10 = 152.9 m
def test_create_zoom_grid_covers_the_bounds():
    grid = create_zoom_grid(bounds, 11)
    assert grid.resolution.x == pytest.approx(ZOOM_0_PIXEL_SIZE / 2**11)
    assert grid.boundingbox.contains(bounds)   # written out as four comparisons
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/geodata/viz/explore/test_mercator.py -v`
- [ ] **Step 3: Implement** both in `mercator.py`.
- [ ] **Step 4: Run the tests** — Expected: PASS
- [ ] **Step 5: Commit** `feat: Web Mercator zoom levels for the map`

---

### Task 4: Raster layers draw a view through a zoom-level cache

**Files:**

- Modify: `src/geosave/geodata/viz/explore/layers/layer.py` (`draw(self, view: GeoBox | None)`)
- Modify: `src/geosave/geodata/viz/explore/layers/raster.py` (delete `read_viewport` and its helpers, `render`, `_sample_*`; add `row`, cache)
- Modify: `src/geosave/geodata/viz/explore/layers/vector.py` (`draw(view)` derives `x_range`, `y_range` from `view.boundingbox`)
- Modify: `src/geosave/geodata/viz/explore/page/map.py` (`_draw` builds the view from `self.ranges` and `self.size`, default shape `(600, 900)` before PlotSize reports)
- Modify: `scripts/measure_explorer.py`
- Test: `tests/geodata/viz/explore/layers/test_raster.py` (replace the `read_viewport` tests)

**Interfaces:**

- Consumes: Task 1 `GeoCatalogItem.read_pixels`, Task 2 `GeoDataset.read_pixels`, Task 3 helpers
- Produces: `RasterLayer(raster, band=None, *, row: pd.Series | None = None, rgb=None, name=None, cmap=None, alpha=0.8)`; `row` is the catalog row whose files hold `raster`'s pixels
- Produces: `RasterLayer.read_view_pixels(self, view: GeoBox) -> xr.Dataset` — the cache's pixels resampled onto `view`
- Produces: `Layer.draw(self, view: GeoBox | None) -> hv.Overlay`

- [ ] **Step 1: Write the failing tests**

```python
def test_pan_inside_the_cache_reads_nothing(scene, monkeypatch):
    reads = count_read_pixels(monkeypatch)       # wraps GeoDataset.read_pixels
    layer = RasterLayer(scene)
    layer.draw(view)
    layer.draw(shift(view, 0.2))                 # 20% of the view: inside the half-view padding
    assert reads == 1
def test_zooming_past_a_level_reads_again(...)   # view at half the pixel size -> reads == 2
def test_changing_the_date_reads_again(monthly, ...)
def test_a_read_covers_only_the_shown_date(monthly, monkeypatch)   # the Dataset passed to read_pixels has no time dim
def test_contrast_change_reads_nothing(...)
def test_row_layer_reads_its_rows_shown_bands(catalog, monkeypatch):
    # row = catalog.iloc[1]; RasterLayer(row.gs.to_xarray(), row=row)
    # -> GeoCatalogItem.read_pixels called once, variables == ["B04"]
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/geodata/viz/explore/layers/test_raster.py -v`

- [ ] **Step 3: Implement.** `draw(view)` → `read_view_pixels(view)` → `_draw_sample` (renamed `_draw_pixels`) → `_stretch` → `_style`. The cache holds `_cached_pixels`, `_cached_zoom`, `_cached_selection` (band, rgb, time, axis positions). It reads when the selection differs, the zoom level differs, or `view.boundingbox` leaves the cached bounds. Reading: `create_zoom_grid(view.boundingbox.buffered(width / 2, height / 2), zoom)`; with `row`, `row.gs.read_pixels(grid, variables=shown_names)`; otherwise `self.select_shown_bands().gs.read_pixels(grid)`. Then `xr_reproject(cached, view, resampling="nearest", dst_nodata=np.nan)`. `_get_rgb_limits` reads its 256-pixel overview the same way, on a GeoBox over the raster's extent. A `None` view uses the layer's `bbox`.

- [ ] **Step 4: Run the viz tests**

Run: `uv run pytest tests/geodata/viz -v`
Expected: PASS

- [ ] **Step 5: Commit** `refactor: raster layers read views through a zoom-level cache`

---

### Task 5: Polygon and raster checkboxes; delete CatalogLayer

**Files:**

- Delete: `src/geosave/geodata/viz/explore/layers/catalog.py`
- Modify: `src/geosave/geodata/viz/explore/layers/vector.py` (add `footprint_rows`, `raster_rows`, `rasters`, `error`; move row opening here)
- Modify: `src/geosave/geodata/viz/explore/layers/__init__.py`, `src/geosave/geodata/viz/explore/__init__.py` (drop `CatalogLayer`)
- Modify: `src/geosave/geodata/viz/explore/explorer.py` (watch `VectorLayer.rasters`; `remove` unchecks the raster row)
- Modify: `src/geosave/geodata/viz/explore/page/table.py` (selection ↔ `footprint_rows`; boolean raster column with `tickCross` ↔ `raster_rows`)
- Modify: `src/geosave/geodata/core/vector/samples.py` (delete `explore` and `_read_sample_rasters`), `src/geosave/geodata/core/vector/catalog.py:553` (drop the `samples.explore` example), `docs/guides/catalogs-and-cuts.md:88`, `docs/concepts/catalog-windows.md:26`
- Test: `tests/geodata/viz/explore/layers/test_vector.py`, `tests/geodata/viz/explore/test_explorer.py`, `tests/geodata/core/vector/test_samples.py`, `tests/geodata/core/vector/test_catalog.py`

**Interfaces:**

- Consumes: Task 4 `RasterLayer(..., row=...)`
- Produces: `VectorLayer(frame, *, name=None, column=None, color=..., alpha=0.3)` (unchanged)
- Produces: `VectorLayer.footprint_rows: list[int]` (default every row), `raster_rows: list[int]`, `rasters: dict[int, RasterLayer]` (by row position), `has_rasters: bool` (rows carry `assets`)

- [ ] **Step 1: Write the failing tests**

```python
def test_unchecked_footprint_is_not_drawn(fields)              # footprint_rows=[0] -> one polygon drawn
def test_frame_without_assets_offers_no_raster_column(fields)  # has_rasters is False
def test_checking_a_raster_row_adds_its_layer(catalog)        # raster_rows=[0] -> list(rasters) == [0], that layer's row is catalog.iloc[0]
def test_map_click_picks_without_opening_rasters(catalog)      # identify -> picked_rows set, raster_rows unchanged
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/geodata/viz tests/geodata/core/vector -v`

- [ ] **Step 3: Implement.** Move `open_checked_rows`/`_open_new_rows` from `catalog.py` onto `VectorLayer`, watching `raster_rows`. Opening a row: `row = self.data.iloc[position]`, `RasterLayer(row.gs.to_xarray(), row=row, name=title)`, titled by collection and date as today. `render` draws `shown_rows ∩ footprint_rows`. `identify` sets only `picked_rows`. `create_layer` returns `VectorLayer` for every GeoDataFrame.

- [ ] **Step 4: Run the tests** — `uv run pytest tests/geodata -v` — Expected: PASS

- [ ] **Step 5: Commit** `refactor: separate polygon and raster checkboxes; fold CatalogLayer into VectorLayer`

---

### Task 6: Dependencies, docs and smoke test

**Files:**

- Modify: `pyproject.toml` (remove `rio-tiler>=9.4.13`), `uv.lock` via `uv sync`
- Modify: `docs/contributing/architecture.md`, `docs/reference/public-api.md`, `docs/concepts/explorer.md` (`read_pixels`, checkbox columns; no `CatalogLayer`)
- Regenerate: `docs/reference/generated/data.md` with the repo's generator, if it lists `CatalogLayer`

- [ ] **Step 1:** `uv remove rio-tiler && uv sync`
- [ ] **Step 2:** Update the docs named above.
- [ ] **Step 3: Smoke test on Sentinel-2.** Script in the scratchpad: open `S2B_48MYU_20240722_0_L2A` as a catalog row; `RasterLayer` over it; draw a zoomed-out view, pan 20%, zoom in past a level. Expected: one read, zero reads, one read; each image `(600, 900)` float32.
- [ ] **Step 4: Browser smoke.** `uv run python scripts/smoke_explorer.py` — Expected: the page loads, a checked raster row draws.
- [ ] **Step 5: Full checks.** `uv run pytest`, `uv run ruff check .`, `uv run basedpyright src/geosave/geodata` — Expected: pass.
- [ ] **Step 6: Commit** `docs: explorer reads views through read_pixels`

---

### Task 7: One map period slider at a chosen step

**Files:**

- Modify: `src/geosave/geodata/viz/explore/explorer.py` (`freq`, `period`, `_offer_periods`)
- Modify: `src/geosave/geodata/viz/explore/layers/raster.py` (`times`, `select_time`; drop the `time` param; `draw(view, time)`)
- Modify: `src/geosave/geodata/viz/explore/page/map.py` (each raster layer draws at `select_time(explorer.period)`; timeless layers always)
- Modify: `src/geosave/geodata/viz/explore/page/controls.py` (remove the per-layer date dropdown), and the page that lays out map controls (add the step selector and period slider, as `pn.ui` widgets from the params)
- Test: `tests/geodata/viz/explore/test_explorer.py`, `tests/geodata/viz/explore/layers/test_raster.py`

**Interfaces:**

- Consumes: Task 4 `RasterLayer.draw`
- Produces: `Explorer.freq: param.Selector` (`{"Year": "Y", "Month": "M", "Day": "D"}`, default `"M"`), `Explorer.period: param.Selector` of `pd.Period`
- Produces: `RasterLayer.times: list[np.datetime64]` (empty when timeless), `RasterLayer.select_time(period: pd.Period | None) -> np.datetime64 | None` (last date inside the period)
- Produces: `RasterLayer.draw(self, view: GeoBox | None, time: np.datetime64 | None = None)`; the cache key uses `time`

- [ ] **Step 1: Write the failing tests**

```python
def test_periods_are_the_months_of_switched_on_layers(three_dated_layers)   # Jan, Feb, Mar rows -> period.objects == [Jan, Feb, Mar] at "M"
def test_day_step_separates_dates_one_day_apart(...)                         # Jan 5 and Jan 6 -> two periods at "D", one at "M"
def test_hiding_a_layer_drops_its_periods(...)
def test_select_time_takes_the_last_date_in_the_period(monthly)             # two dates in Jan -> the later one
def test_timeless_layer_draws_at_every_period(...)
def test_moving_within_a_month_reads_nothing(...)                           # same selected time -> read count unchanged
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/geodata/viz -v`
- [ ] **Step 3: Implement** as the Interfaces state; `_offer_periods` uses `pd.DatetimeIndex(times).to_period(freq).unique().sort_values()` and keeps the current period when still offered, else the first.
- [ ] **Step 4: Run the tests** — Expected: PASS
- [ ] **Step 5: Commit** `feat: one map period slider at a year, month or day step`

---

### Task 8: Catalog rows open as scenes through rasterio

**Files:**

- Modify: `src/geosave/geodata/io/raster/stac.py` (`open_scene`, `read_scene_pixels`; delete `open_raster`, `open_collections`, `_describe_items`), `src/geosave/geodata/core/vector/item.py`, `src/geosave/geodata/core/vector/catalog.py`
- Test: `tests/geodata/io/raster/test_stac.py`, `tests/geodata/stac/translate/test_item.py`, `tests/geodata/io/vector/test_stac_geoparquet.py`

**Interfaces:**

- Produces: `open_scene(item: pystac.Item, **options) -> xr.Dataset` (options reach the GDAL reader); `read_scene_pixels(item, geobox, variables=None) -> xr.Dataset`
- Produces: `GeoCatalogItem.to_xarray(**options)` stamps the table's Collection; `GeoCatalog.to_xarray(**options)` concatenates each collection's scenes

- [ ] **Step 1:** tests: a scene merges its assets, closes every file once and on failure; rows on two dates stack along time; a scene whose files differ in grid fails at the merge.
- [ ] **Step 2:** implement; run `uv run pytest tests/geodata tests/model/spec/test_model.py tests/flows/test_ingest.py` (baseline 1146 passed).
