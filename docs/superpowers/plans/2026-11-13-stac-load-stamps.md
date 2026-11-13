# STAC Load Stamps and Property Coordinates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `load_dataset`'s fixed metadata helpers with a registry of named stamps, one per STAC schema, and replace the `stac_properties` dict coordinate with one coordinate per Item property marked by a `stac_property` attr.

**Architecture:** `stac/stamps.py` holds `LoadSource`, the `Stamp` type and the registry. Each STAC schema module in `stac/extensions/` registers its own stamp; core stamps (`title`, `properties`, `collection`) live in `stamps.py`. Properties are written and read only through `.gs.write_stac_property` and `.gs.stac_properties` on the shared raster accessor. `load_dataset` keeps fixed steps for what load options decide and runs stamps for what STAC documents state.

**Tech Stack:** xarray, odc-stac 0.5.2, pystac, pydantic attrs groups, pytest.

**Spec:** `docs/contributing/design/stac-load-stamps.md`

## Global Constraints

- No compatibility aliases or duplicate paths; old helpers are deleted, not wrapped (CLAUDE.md "Changes").
- A coordinate is a STAC property exactly when its attrs hold `stac_property`.
- Only scalar property values (str, int, float, bool, None) become coordinates.
- A stamp reads only STAC (`LoadSource`), never attrs another stamp wrote; a later stamp overrides an earlier one's attrs.
- `stamps=` takes names only; unknown name → `KeyError` listing registered names, raised before any pixel load.
- `stac_cfg` is removed from `StacLoadOptions`; `with_properties` stays.
- Requested numeric properties default to `float64`, strings to `object`; a request's own `dtype` wins.
- Publication skips property values that are None, NaN or `""` (a missing string reads back from Zarr/netCDF as `""`).
- Loops written out; no comprehension with more than one `for` (CLAUDE.md).
- `geodata` imports no torch; tests mirror `src/` paths.
- Run tests with `uv run pytest <path>`; lint with `uv run ruff check src tests`.

## Review Focus

1. An Item lacking a property its neighbours state, loaded with `groupby="id"`: the per-step coordinate holds a missing value there, and that Item's published properties omit the key. → Task 6 `test_properties_mark_missing_values_per_item`, Task 8 `test_publication_skips_missing_property_values`.
2. A property name equal to a dimension or data variable (`time`, `x`, `B04`): `write_stac_property` raises `ValueError` instead of overwriting. → Task 1 `test_write_refuses_names_of_dims_and_variables`.
3. Stamps on a dask-backed load read no pixels. → Task 7 `test_stamps_read_no_pixels`.
4. A requested property some Items lack: ODC fills NaN; the coordinate keeps NaN and publication omits it. → Task 7 `test_requested_property_absent_on_some_items`.
5. An integer property (`sat:relative_orbit`) publishes as `int`, not `float` or a NumPy scalar. → Task 8 `test_publication_restores_python_types`.

---

### Task 1: `StacProperty` attrs group and accessor

**Files:**
- Modify: `src/geosave/geodata/attrs/groups/stac.py`, `src/geosave/geodata/attrs/__init__.py`
- Modify: `src/geosave/geodata/core/raster/base.py` (new mixin beside `RasterTime`), `dataarray.py:106`, `dataset.py:121` (inherit it)
- Test: `tests/geodata/attrs/groups/test_stac.py`, `tests/geodata/core/raster/test_base.py`

**Interfaces:**
- Produces: `attrs.StacProperty(name: str)`, serialized key `stac_property`.
- Produces: `RasterStacProperties[DataT]` mixin with
  `write_stac_property(self, name: str, values: Any, *, item_dim: str = TIME_COORDINATE) -> DataT` and
  `stac_properties -> dict[str, Hashable]` (STAC name → coordinate name).

- [ ] **Step 1: Write failing tests**

```python
def test_stac_property_round_trips_through_attrs():
    assert attrs.StacProperty.from_attrs({"stac_property": "eo:cloud_cover"}).name == "eo:cloud_cover"
    assert attrs.StacProperty(name="platform").to_attrs() == {"stac_property": "platform"}

def test_write_scalar_and_per_step_properties(scene):          # scene: 3 time steps
    out = scene.gs.write_stac_property("platform", "Sentinel-2B")
    out = out.gs.write_stac_property("eo:cloud_cover", [0.4, 12.1, 3.0])
    assert out["platform"].dims == ()
    assert out["eo:cloud_cover"].dims == ("time",)
    assert out["eo:cloud_cover"].attrs["stac_property"] == "eo:cloud_cover"
    assert out.gs.stac_properties == {"platform": "platform", "eo:cloud_cover": "eo:cloud_cover"}

def test_unmarked_coordinates_are_not_properties(scene):
    assert "month" not in scene.assign_coords(month=("time", [2, 2, 3])).gs.stac_properties

def test_write_along_another_item_dim(scene):
    out = scene.rename(time="scene").gs.write_stac_property("dw:id", ["a", "b", "c"], item_dim="scene")
    assert out["dw:id"].dims == ("scene",)

def test_write_refuses_wrong_length_missing_dim_and_nested_values(scene):
    # ValueError: length 2 vs 3; KeyError: item_dim="scene" absent; ValueError: ["msi"] as a value

def test_write_refuses_names_of_dims_and_variables(scene):
    # ValueError for name "time", "x" and a data variable name

@pytest.mark.parametrize("operation", [isel, sel_by_property, select_variable, multiply, where,
                                       spatial_mean, concat_time, merge])
def test_marks_survive_operations_that_keep_the_coordinate(scene, operation): ...

def test_per_step_marks_go_with_time_reductions(scene):
    out = marked(scene).mean("time")
    assert "eo:cloud_cover" not in out.coords and out["platform"].attrs["stac_property"] == "platform"

def test_property_coordinates_round_trip_through_netcdf_and_zarr(scene, tmp_path): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/geodata/attrs/groups/test_stac.py tests/geodata/core/raster/test_base.py -k "stac_property or property" -v` — FAIL (`StacProperty` undefined).

- [ ] **Step 3: Implement** the group (one field `name`, alias `stac_property`, like `StacCollection`'s aliasing), and the mixin in `base.py`. `write_stac_property` writes the coordinate then stamps the group with `attrs.stamp_attrs` on the coordinate variable; `stac_properties` iterates `self._data.coords` and reads the group.

- [ ] **Step 4: Run** the same command — PASS.

- [ ] **Step 5: Commit** `feat(geodata): mark STAC property coordinates with stac_property`

---

### Task 2: Stamp registry

**Files:**
- Create: `src/geosave/geodata/stac/stamps.py`
- Test: `tests/geodata/stac/test_stamps.py`

**Interfaces:**
- Produces:
  `LoadSource(items: Sequence[pystac.Item], collection: pystac.Collection | None, groupby: str | Callable[[pystac.Item, ParsedItem, int], Any])` (frozen dataclass);
  `type Stamp = Callable[[xr.Dataset, LoadSource], xr.Dataset]`;
  `STAMPS: dict[str, Stamp]`;
  `register_stamp(name: str) -> Callable[[Stamp], Stamp]`;
  `check_stamps(names: Sequence[str]) -> None` (raises `KeyError` naming unknown and registered names);
  `run_stamps(loaded: xr.Dataset, source: LoadSource, names: Sequence[str]) -> xr.Dataset`;
  `DEFAULT_STAMPS = ("raster", "eo", "classification", "rgb", "title", "properties", "collection")`.

- [ ] **Step 1: Write failing tests**

```python
def test_stamps_run_in_order_and_later_overrides(registered):   # two test stamps writing attr "a"
    out = run_stamps(xr.Dataset(), source, ["first", "second"])
    assert out.attrs["a"] == "second"

def test_unknown_stamp_names_the_registered_ones():
    with pytest.raises(KeyError, match="registered"):
        check_stamps(["nope"])

def test_registering_a_name_twice_raises(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/geodata/stac/test_stamps.py -v` — FAIL (module missing).
- [ ] **Step 3: Implement** the module. Built-in stamps are registered by Tasks 4–6 in their modules; `stamps.py` imports those modules at its end so importing it registers them.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(geodata): registry of named load stamps`

---

### Task 3: `band.read_band` with STAC 1.1 inheritance

**Files:**
- Modify: `src/geosave/geodata/stac/band.py` (add `read_band`)
- Test: `tests/geodata/stac/test_band.py`

**Interfaces:**
- Produces: `read_band(item: pystac.Item, name: str) -> dict[str, Any]` — the STAC fields of the band ODC loads as `name`: asset-level fields first, then `raster:bands`, `eo:bands` and `bands` entries at the band's index overriding them. Keys keep their STAC spelling (`raster:scale`, `scale`, `eo:common_name`, `common_name`, …). Name resolution through `odc.stac.parse_item(item).collection.band_key(name)`.

- [ ] **Step 1: Write failing tests**

```python
def test_asset_level_raster_fields_apply_to_its_band():           # CDSE shape
    item = item_with_asset("B04_10m", {"raster:scale": 0.0001, "raster:offset": -0.1, "nodata": 0,
                                        "bands": [{"name": "B04", "eo:common_name": "red"}]})
    fields = read_band(item, "B04_10m")
    assert fields["raster:scale"] == 0.0001 and fields["raster:offset"] == -0.1

def test_a_band_entry_overrides_the_asset(): ...                  # asset raster:scale 1, band scale 2 → 2
def test_raster_v1_list_is_read_by_index(): ...                   # E84 shape, multi-band asset, index 1
def test_common_name_alias_finds_the_asset(): ...                 # read_band(pc_item, "red") reads B04
def test_unknown_band_returns_empty(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/geodata/stac/test_band.py -v` — FAIL.
- [ ] **Step 3: Implement** `read_band` beside the old functions. `stamp_band_attrs` in `load.py` still uses `read_attrs` / `read_shared_attrs` / `_read_fields`; Task 7 deletes all of them together.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(geodata): read a band's STAC fields with asset-level inheritance`

---

### Task 4: `raster` stamp with the mixed-scale refusal

**Files:**
- Modify: `src/geosave/geodata/stac/extensions/raster.py`
- Test: `tests/geodata/stac/extensions/test_raster.py`

**Interfaces:**
- Consumes: `read_band` (Task 3), `register_stamp`, `LoadSource` (Task 2).
- Produces: `stamp_raster(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset`, registered as `"raster"`. Writes `CFPacking` and `units`; writes nodata with `rio.write_nodata` as `stamp_band_attrs` does today (the declaration must match ODC's cast fill).

- [ ] **Step 1: Write failing tests**

```python
def test_raster_stamp_writes_packing_from_asset_level_fields(): ...   # → scale_factor 1e-4, add_offset -0.1
def test_raster_stamp_writes_packing_from_band_lists(): ...
@pytest.mark.parametrize("field, value", [("scale", 2.0), ("offset", 1.0)])
def test_raster_stamp_refuses_items_packing_a_band_differently(field, value):
    with pytest.raises(ValueError, match="incompatible source scale/offset"): ...
def test_raster_stamp_without_packing_leaves_values_unscaled(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/geodata/stac/extensions/test_raster.py -v` — FAIL.
- [ ] **Step 3: Implement**; move `_check_source_scales`'s logic here (compare `(scale_factor, add_offset)` across `source.items`, identity when omitted, nodata excluded).
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(geodata): raster stamp`

---

### Task 5: `eo`, `rgb`, `classification` and `title` stamps

**Files:**
- Modify: `src/geosave/geodata/stac/extensions/eo.py` (`eo`, `rgb`), `extensions/classification.py`, `src/geosave/geodata/stac/stamps.py` (`title`)
- Test: `tests/geodata/stac/extensions/test_eo.py`, `test_classification.py`, `tests/geodata/stac/test_stamps.py`

**Interfaces:**
- Produces: `stamp_eo`, `stamp_rgb`, `stamp_classification`, `stamp_title`, each `(loaded, source) -> xr.Dataset`, registered under `"eo"`, `"rgb"`, `"classification"`, `"title"`. Each writes the attrs every Item agrees on (xarray `drop_conflicts`, as `read_shared_attrs` did).

- [ ] **Step 1: Write failing tests** — move today's `test_load.py` cases into these files as stamp tests:
  `test_spectral_facts_reach_the_variable` → `stamp_eo`;
  `test_loading_states_the_rgb_composite_the_band_colours_name` and `..._no_composite_without_a_blue_band` → `stamp_rgb`;
  add `test_rgb_reads_stac_not_eo_attrs` (run `["rgb"]` alone; `Display.rgb` still written);
  `test_classification_stamp_writes_flags_and_colours`;
  `test_title_names_a_one_band_asset` / `test_title_skips_multi_band_assets`.
- [ ] **Step 2: Run** the three test files — FAIL.
- [ ] **Step 3: Implement** using the existing `eo.read_attrs`, `classification.read_attrs`, `write_rgb`.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(geodata): eo, rgb, classification and title stamps`

---

### Task 6: `properties` and `collection` stamps

**Files:**
- Modify: `src/geosave/geodata/stac/stamps.py`
- Test: `tests/geodata/stac/test_stamps.py`

**Interfaces:**
- Consumes: `write_stac_property` (Task 1), `item.select_item_properties`, `item.read_instant`.
- Produces: `stamp_properties`, `stamp_collection` registered as `"properties"`, `"collection"`. `stamp_collection` absorbs `stac/raster.py::build_stac_metadata` (moved, then deleted in Task 8).

Rule for `properties` (the spec leaves step mapping to the builder; this is the choice):

```text
every time step is exactly one Item (matched by read_instant, as _read_item_cells matches)
    → each scalar property: scalar coord if all Items agree, else (time,) coord, missing where an Item lacks it
otherwise (fused steps, timeless)
    → only scalar properties every Item shares, as scalar coords
list/dict values → skipped
```

- [ ] **Step 1: Write failing tests**

```python
def test_properties_one_item_per_step_vary_along_time(): ...      # eo:cloud_cover (time,), platform scalar
def test_properties_mark_missing_values_per_item(): ...           # Item 2 lacks s2:tile → None there
def test_properties_of_fused_steps_keep_only_shared(): ...
def test_properties_skip_nested_values(): ...                     # instruments absent
def test_properties_exclude_dates_grid_and_bands(): ...           # datetime, proj:*, eo:bands absent
def test_collection_stamp_states_collection_and_extensions(): ... # from test_a_load_carries_its_collection...
```

- [ ] **Step 2: Run** `uv run pytest tests/geodata/stac/test_stamps.py -v` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(geodata): properties and collection stamps`

---

### Task 7: Rewire `load_dataset` and `load_datatree`

**Files:**
- Modify: `src/geosave/geodata/stac/load.py`, `src/geosave/geodata/stac/__init__.py`
- Test: `tests/geodata/stac/test_load.py` (rewrite)

**Interfaces:**
- Consumes: `check_stamps`, `run_stamps`, `LoadSource`, `DEFAULT_STAMPS`, `write_stac_property`.
- Produces:
  `load_dataset(items, geobox, *, groupby="solar_day", collection=None, stamps: Sequence[str] = DEFAULT_STAMPS, mask_and_scale=False, **options) -> xr.Dataset`;
  `load_datatree(items, *, collections=(), stamps: Sequence[str] = DEFAULT_STAMPS, **options) -> xr.DataTree`;
  `geosave.geodata.stac` exports `DEFAULT_STAMPS`, `LoadSource`, `register_stamp`.

Order inside `load_dataset`: `_check_items` → `check_stamps(stamps)` → `_prepare_options` (no `stac_cfg`; property requests default `dtype` float64 for non-string requests, `object` when the caller says `{"key": k, "dtype": "object"}`; ODC names the variable by the exact key as today) → `odc.stac.load` → time axis (timeless drop, Item cells, solar-day cells; unchanged code) → requested properties: each ODC variable becomes a coordinate via `write_stac_property` → `run_stamps` → `mask_and_scale`.

`load_datatree`: forwards `stamps`; `_place_asset` writes the Item's own properties with `write_stac_property` (scalar coords on the Item node) instead of the dict.

Delete: `stamp_band_attrs`, `_band_sources`, `_check_source_scales`, `_stamp_date_properties`, the `stac_cfg` key of `StacLoadOptions`, the `intersect_mappings` import, and in `band.py` `_read_fields`, `read_attrs`, `read_shared_attrs`, `BandAttrs`, `BandFields`.

- [ ] **Step 1: Rewrite `test_load.py`**: delete cases now covered by stamp tests (Tasks 4–6); keep grid, grouping, period, tree, signing and laziness cases with `stac_properties` assertions replaced by property coordinates. Add:

```python
def test_default_stamps_write_bands_properties_and_collection(local_catalog): ...
def test_extra_stamp_runs_after_defaults(local_catalog, registered_stamp): ...
def test_unknown_stamp_raises_before_loading(local_catalog, monkeypatch):   # odc.stac.load never called
def test_stamps_read_no_pixels(local_catalog): ...                          # dask graph not computed
def test_requested_property_becomes_a_marked_coordinate(observation): ...   # eo:cloud_cover (time,) float64
def test_requested_property_absent_on_some_items(observation): ...          # NaN kept
def test_stac_cfg_is_refused(local_catalog):
    with pytest.raises(TypeError, match="unsupported load options: stac_cfg"): ...
def test_tree_item_nodes_hold_their_own_property_coordinates(observation): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/geodata/stac/test_load.py -v` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/geodata/stac -v` — PASS.
- [ ] **Step 5: Commit** `refactor(geodata): load_dataset runs named stamps; properties as coordinates`

---

### Task 8: Remove `stac_properties`; publication reads marked coordinates

**Files:**
- Delete: `src/geosave/geodata/stac/raster.py`, `tests/geodata/stac/test_raster.py`
- Modify: `src/geosave/geodata/core/conventions.py` (drop `STAC_PROPERTIES`; module docstring diagrams show property coordinates), `src/geosave/geodata/io/formats/netcdf.py:162,216`, `src/geosave/geodata/io/formats/zarr.py:237,259,352` (drop encode/decode calls), `src/geosave/geodata/stac/publish.py:54-80`
- Test: `tests/geodata/stac/test_publish.py`, `tests/geodata/core/raster/test_dataset.py`

**Interfaces:**
- Consumes: `stac_properties` (Task 1).
- Produces: `StacItem.from_datatree` builds properties from the Item node's `gs.stac_properties`, value `.item()` per coordinate, skipping None, NaN and `""`; explicit `properties=` still overrides.

- [ ] **Step 1: Write failing tests**

```python
def test_publication_publishes_only_marked_coordinates(): ...    # month absent, eo:cloud_cover present
def test_publication_skips_missing_property_values(): ...        # None, NaN, "" omitted
def test_publication_restores_python_types(): ...                # sat:relative_orbit is int 90
def test_expand_then_to_catalog_round_trips_per_step_properties(tmp_path): ...
```
Replace `stac_properties` assertions in `test_dataset.py` (expand) with property coordinates.

- [ ] **Step 2: Run** `uv run pytest tests/geodata/stac/test_publish.py tests/geodata/core/raster/test_dataset.py -v` — FAIL.
- [ ] **Step 3: Implement**; `grep -rn "STAC_PROPERTIES\|stac_properties\|encode_properties\|decode_properties" src` returns only the accessor property.
- [ ] **Step 4: Run** `uv run pytest tests/geodata -v` — PASS except `test_temporal.py` (Task 9).
- [ ] **Step 5: Commit** `refactor(geodata): drop the stac_properties coordinate`

---

### Task 9: `composite` keeps each bin's shared properties

**Files:**
- Modify: `src/geosave/geodata/core/raster/transform/temporal.py:40-103`, `src/geosave/geodata/core/raster/dataset.py:136-196` (drop the `properties` parameter), `src/geosave/geodata/utils/mapping.py` (delete `stack_mappings`; delete `intersect_mappings` if no caller remains)
- Test: `tests/geodata/core/raster/transform/test_temporal.py`

**Interfaces:**
- Produces: `composite(...)` without `properties`. For each per-step property coordinate along `dim`, each bin gets the value when all member scenes agree, else missing (None for object, NaN for float); scalar properties pass through unchanged.

- [ ] **Step 1: Write failing tests**

```python
def test_composite_keeps_a_property_the_bin_shares(): ...         # s2:tile 'A','A' → 'A'
def test_composite_marks_a_disagreeing_bin_missing(): ...         # eo:cloud_cover 1, 5 → NaN
def test_composite_keeps_scalar_properties(): ...
def test_composite_output_properties_stay_marked(): ...
```
Delete the `properties="stack"` tests.

- [ ] **Step 2: Run** `uv run pytest tests/geodata/core/raster/transform/test_temporal.py -v` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/geodata -v` — PASS.
- [ ] **Step 5: Commit** `refactor(geodata): composite keeps per-bin shared properties`

---

### Task 10: `sentinel_2` stamp

**Files:**
- Create: `src/geosave/geodata/stac/extensions/sentinel_2.py`; register import in `stamps.py`
- Test: `tests/geodata/stac/extensions/test_sentinel_2.py`

**Interfaces:**
- Produces: `stamp_sentinel_2(loaded, source) -> xr.Dataset`, registered as `"sentinel_2"`. Packing per ESA: `scale_factor=1e-4`, `add_offset=-0.1` when `float(s2:processing_baseline) >= 4.0`, else `0.0`; applied to every data variable. Docstring cites https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-pb04-00-products/35431.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("baseline, offset", [("02.12", 0.0), ("04.00", -0.1), ("05.00", -0.1)])
def test_sentinel_2_packing_follows_the_baseline(baseline, offset): ...
def test_sentinel_2_refuses_mixed_baselines():
    with pytest.raises(ValueError, match=r"02\.12.*05\.00"): ...
def test_sentinel_2_refuses_items_without_a_baseline(): ...
def test_sentinel_2_overrides_packing_the_raster_stamp_wrote(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/geodata/stac/extensions/test_sentinel_2.py -v` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(geodata): sentinel_2 stamp derives ESA packing`

---

### Task 11: Docs and the Dynamic World ingest

**Files:**
- Modify: `docs/contributing/architecture.md` (raster structure), `docs/concepts/metadata.md`, `docs/guides/stac.md`, `docs/reference/stac.md`, `docs/reference/public-api.md`, `docs/contributing/open-decisions.md` (drop the `stac_properties` mentions; add the CDSE item as resolved), `docs/contributing/design/stac-load-stamps.md` (status: implemented)
- Modify (workspace repo): `workspace/dynamicworld/scripts/ingest_dynamic_world.py` — delete `read_esa_packing`; `load_image` passes `stamps=[*stac.DEFAULT_STAMPS, "sentinel_2"]` and drops `write_packing`.

- [ ] **Step 1:** `uv run pytest tests/docs -q` if present, and `grep -rn "stac_properties\|with_properties\|stac_cfg" docs` shows only the new wording.
- [ ] **Step 2:** Rerun the ingest on one Indonesia scene into the scratchpad; the image reads back with `scale_factor 1e-4`, `add_offset -0.1` and its grid equals the label's.
- [ ] **Step 3:** `uv run pytest` (full suite) and `uv run ruff check src tests` — PASS.
- [ ] **Step 4: Commit** `docs: STAC property coordinates and load stamps` (main repo) and `ingest: sentinel_2 stamp` (workspace repo).
