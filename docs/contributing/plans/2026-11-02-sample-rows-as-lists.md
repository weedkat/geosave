# Sample Rows as Lists Implementation Plan

Status: superseded 2026-11-02 by `GeoSamples` (the item table plus each sample's row positions): no list columns, no `GeoSample` row accessor.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `catalog.gs.sample(...)` returns one row per sample, with every Item column as a list, and `samples.iloc[i].gs.to_xarray()` opens that sample as a DataTree.

**Architecture:** Sampling writes group indices (for each sample, the index of its Items), then `build_samples` lays the catalog's rows out by them into one row of lists per sample. A table whose `id` is an Arrow list gets the `GeoSamples` accessor. `GeoCatalog` stays plain stac-geoparquet, one Item per row.

**Tech Stack:** geopandas, shapely, pandas ArrowDtype, pyarrow `ListArray`, pystac, stac-geoparquet.

**Spec:** this plan's Design section, agreed in discussion on 2026-11-02. It replaces the table shape of `docs/contributing/specs/2026-10-31-samples-by-region.md` (repeated Item rows keyed by `gs:region`/`gs:window`).

## Design

```python
samples = catalog.gs.sample(pair=["s2", "s1"], tolerance="2D", window=2)
len(samples)                             # number of samples
samples.gs.explore()                     # each sample's shape
tree = samples.iloc[0].gs.to_xarray()    # DataTree ('s2', 's1', 'dem'), lazy
```

```text
catalog (GeoCatalog)                         samples (GeoSamples)
   id       collection  datetime  geometry      geometry  id                          collection         datetime              geometries
0  s2_0105  s2          Jan 5     A             0  A      [s2_0105, s1_0106,          [s2, s1, s2, s1,   [Jan 5, Jan 6, Feb 4, [▭, ▭, ▭, ▭, ▭]
1  s1_0106  s1          Jan 6     A                        s2_0204, s1_0203, dem]      dem]               Feb 3, None]
2  s2_0204  s2          Feb 4     A             1  A      [s2_0204, s1_0203,          ...
...                                                         s2_0308, s1_0309, dem]
6  dem      dem         -         A
```

```python
# GeoCatalog.sample, after its argument checks
catalog = self._data.reset_index(drop=True)   # index labels are row positions from here on
footprints = sample.find_distinct_footprints(catalog.geometry, overlap)

group_indices: GroupIndices = []
for footprint in footprints:
    # 1. Find the Items whose footprints intersect this one
    candidate_index = catalog.sindex.query(footprint, predicate="intersects")
    candidate_index = np.sort(candidate_index)
    # 2. Measure the area each candidate shares with the footprint
    candidate_geometries = catalog.geometry.values[candidate_index]
    intersections = shapely.intersection(footprint, candidate_geometries)
    intersection_areas = shapely.area(intersections)
    # 3. Keep the candidates sharing at least `overlap` of the footprint's area
    minimum_area = overlap * shapely.area(footprint)
    group_indices.append(candidate_index[intersection_areas >= minimum_area].tolist())

if pair is not None:
    group_indices = sample.cut_windows(catalog, group_indices, pair, largest_gap, window, step, freq, keep)
return sample.build_samples(catalog, group_indices)
```

Decisions, each smoke-tested against today's `sample()` on the real COG fixtures (`tests/geodata/core/vector/test_sample.py::tiles`). In five option sets the groups are the same, and every opened row is `identical` and lazy:

- **`type GroupIndices = list[list[int]]`** in `sample.py`: for each sample, the index of its Items. A `DataFrameGroupBy` can't hold these groups, because they overlap: a swath belongs to every footprint it covers, and a timeless Item to every window.
- **Distinct footprints.** Visited smallest first, a footprint is kept unless it covers at least `overlap` of one already kept, so a swath never becomes its own sample. Membership is decided against every kept footprint afterwards, so it doesn't depend on visiting order. Today's `[a, b], [c]` at `overlap=0.5` on shifted footprints becomes `[a, b], [b, c]`, because `b` covers 70% of `c`. `dissolve().explode()` was tested and rejected: a swath merges two tiles into one group.
- **`cut_windows` maps `GroupIndices -> GroupIndices`.** It splits each footprint's group into one group per window of paired Items, appends the footprint's timeless Items to each window, and warns once with totals.
- **`build_samples` uses pyarrow.** `groupby().agg(list)` and `dissolve(aggfunc=list)` raise on Arrow columns, even plain strings (`Unsupported cast from list<item: string> to utf8`). `pa.ListArray.from_arrays(offsets, column.take(index))` keeps each column's type and round-trips through Parquet.
- **`geometries` holds each Item's own geometry,** as a list of WGS84 WKB. `geometry` is the sample's shape: the intersection of its `geometries` (equal to the distinct footprint in the fixtures). The sample table is in WGS84, like the Items.
- **No `gs:region`, no `gs:window`.** The row index numbers the samples.
- **A sample row builds its Items from its own values.** `samples.iloc[i]` turns Arrow lists into Python lists. Rebuilding an Arrow table from those infers `null` for a column that's empty in a collection, and `to_xarray` then fails. Each Item is built with `_stac.to_item` from its values instead. `build_item` accepts any `datetime`, because list elements come back as `datetime`, not `pd.Timestamp`.
- **Accessors are chosen by `id`.** If it's an Arrow list: `GeoSamples` / `GeoSamplesItem`. Otherwise: `GeoCatalog` / `GeoCatalogItem`. The vector operations shared by both (`crs`, `footprint`, `plot`, `explore`, `rasterize`) move to a `GeoVector` base. `GeoCatalog` keeps `query`, `concat`, the writers and the Item operations. `GeoSamples` adds nothing, so it has nothing to refuse.
- **`create_catalog`, `_stac.from_items`, `stac/translate/*` and `stac/extensions/*` are unchanged.** They build and read single Items, and a sample row reaches them one Item at a time.

Removed: `assign_regions`, `find_covered_regions`, `pair_items`, `order_slots`, `roll_windows`, `read_times`, `copy_attrs`, `REGION_KEY`, `WINDOW_KEY`, `SAMPLE_KEYS`; `GeoCatalog.samples`; the "these rows hold N samples" check; `SAMPLE_KEYS` stripping in `GeoCatalog.to_items` and `GeoCatalogItem.to_item`; the samples check in `io/vector/stac_geoparquet.py`, together with its `io -> core.vector` import.

Breaking: the shape of `sample()` output, `gs.samples`, `gs:region`/`gs:window`, sample rows keeping the catalog's index (trace an Item by `(collection, id)`), and the shifted-footprint membership change above.

## Global Constraints

- Dependencies point `geodata <- model <- ml`. `geodata` imports no torch.
- Loops are written out in full: no comprehension with more than one `for`, no clever one-liners.
- Function names are verb + object; variables are named for the value they hold. Domain terms only: footprint, Item, group indices, sample.
- A one- or two-line pandas/shapely step is written inline with a comment, not wrapped in a function.
- Big functions get a short comment per chunk saying what it does.
- No compatibility aliases, no second sampling path.
- Opening stays lazy: every variable opened from a sample has Dask chunks.
- Run `uv run basedpyright <changed files>` and `uv run ruff check .`. No commits unless the user asks.

## Review Focus

1. **Many Items each missing a different edge of the footprint:** the intersection shrinks below the footprint, and in theory becomes empty. Expected: a sample's `geometry` is never empty; `build_samples` falls back to the group's smallest geometry when the intersection is empty.
2. **Accessor caching:** pandas caches `.gs` on the frame. Expected: a sample table built by assignment (`frame["id"] = lists`) after `.gs` was first read still gets `GeoSamples` on a fresh selection (`frame.copy().gs`).
3. **A sample table read with `gpd.read_parquet` without `types_mapper`:** list values come back as numpy arrays. Expected: `GeoSamplesItem.to_items` treats `np.ndarray` like `list`.
4. **`GeoCatalog.concat` of two sample tables, or a sample table with a non-default index:** `ModelSpec.prepare` names folders by index label. Expected: `prepare` samples its own catalog, so labels are 0…n−1. Test that two samples get two folders.
5. **A footprint whose Items leave no complete window:** it contributes no sample, and the `DroppedRowsWarning` total counts it. Expected: no empty row, and the `ValueError` only when no footprint has a window.

---

### Task 1: Open a list of Items as one tree

**Files:**

- Modify: `src/geosave/geodata/io/raster/stac.py` (add `open_collections`)
- Modify: `src/geosave/geodata/io/vector/_stac.py` (`build_item`)
- Modify: `src/geosave/geodata/core/vector/catalog.py` (`GeoCatalog.to_xarray`)
- Test: `tests/geodata/io/raster/test_stac.py`, `tests/geodata/io/vector/test_stac.py` (create where absent)

**Interfaces:**

- Produces: `open_collections(items: Sequence[pystac.Item], *, collections: Mapping[str, pystac.Collection]) -> xr.DataTree`. It groups Items by `collection_id` in first-appearance order, opens each group with `open_raster(group, collection=collections.get(name))`, and hands every closer to the tree, as `openers.open_tree` does. It raises `ValueError` when an Item names no collection.
- Produces: `build_item` writes any `datetime` property value (`pd.Timestamp` included) as ISO text.

- [ ] **Step 1: Write failing tests**

```python
def test_a_property_date_given_as_datetime_becomes_iso_text():
    fields = {"type": "Feature", "id": "a", "geometry": None, "bbox": None, "links": [],
              "assets": {}, "stac_extensions": [],
              "properties": {"datetime": datetime(2024, 1, 5, tzinfo=UTC)}}
    assert build_item(fields).datetime == datetime(2024, 1, 5, tzinfo=UTC)

def test_collections_open_as_groups_in_first_appearance_order(items):
    tree = open_collections(items.gs.to_items(), collections=items.gs.collections)
    assert tree.gs.groups == tuple(dict.fromkeys(items["collection"]))
    assert tree["s1"].dataset.vv.chunks is not None

def test_an_item_without_a_collection_cannot_name_its_group(items):
    entry = items.gs.to_items()[0]
    entry.collection_id = None
    with pytest.raises(ValueError, match="collection"):
        open_collections([entry], collections={})
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/io/raster/test_stac.py tests/geodata/io/vector/test_stac.py -q`
Expected: `ImportError` for `open_collections`; a `TypeError` from `isoparse` for the date test.

- [ ] **Step 3: Implement both; `GeoCatalog.to_xarray` becomes `open_collections(self.to_items(), collections=self.collections)`**

The Items are built once for the whole table. The per-collection `groupby`, the per-group `to_items` and the sample-count check go.

- [ ] **Step 4: Run them and confirm they pass**

Run: `uv run pytest tests/geodata/io tests/geodata/core/vector/test_catalog.py -q`
Expected: pass, including `test_catalog_tree_closes_all_collection_rasters_once` (it monkeypatches `raster_stac.open_raster`, which `open_collections` looks up when called).

---

### Task 2: Sample by distinct footprints and windows into group indices

**Files:**

- Rewrite: `src/geosave/geodata/core/vector/sample.py`
- Modify: `src/geosave/geodata/core/vector/catalog.py` (`sample`; delete `samples`)
- Test: `tests/geodata/core/vector/test_sample.py`, `tests/geodata/core/vector/test_catalog.py:400-466`, `tests/geodata/core/vector/conftest.py` (`read_sample_ids` reads `samples["id"].tolist()`)

**Interfaces:**

- Produces, in `sample.py`:
  - `type GroupIndices = list[list[int]]`
  - `find_distinct_footprints(footprints: gpd.GeoSeries, overlap: float) -> gpd.GeoSeries`: footprints, smallest first, each covering less than `overlap` of every one kept before it. Areas come from `shapely.area(footprints.values)` (no geographic-CRS warning).
  - `cut_windows(catalog: gpd.GeoDataFrame, group_indices: GroupIndices, pair: Sequence[str], tolerance: pd.Timedelta, window: int, step: int, freq: str | None, keep: Keep) -> GroupIndices`: one group per window of paired Items, with each footprint's timeless Items appended. It warns `DroppedRowsWarning` once for unpaired Items and once for broken windows, with totals across all footprints. It raises `ValueError(f"no footprint holds {window} consecutive pairs of {list(pair)}")` when no window is found.
  - `build_samples(catalog: gpd.GeoDataFrame, group_indices: GroupIndices) -> gpd.GeoDataFrame`: the docstring and body agreed in discussion. The Item columns become `pd.ArrowDtype` lists; `geometry` is renamed `geometries` (WGS84 WKB); the active `geometry` is `shapely.intersection_all` of each group's geometries, falling back to the group's smallest geometry when that is empty (Review Focus 1); CRS is WGS84; the result carries a deep copy of `catalog.attrs`.
- Produces: `GeoCatalog.sample(...)` with today's parameters, returning `build_samples(...)`, as in the Design sketch.

`cut_windows` keeps today's pairing rules, written out in labelled chunks: `merge_asof` nearest within `tolerance` per footprint, `resample(freq)` slots filled by `keep`, runs of `window` slots every `step`, a run holding an empty slot breaks. It works on item index labels instead of frames.

- [ ] **Step 1: Rewrite the sample tests to the new shape**

Keep each behavior test (pairing, `step`, `freq`/`keep`, empty bucket, monthly windows, option checks, overlap). Only change how a sample is read: `read_samples` iterates `samples["id"]`; `len(samples)` replaces `ngroups`. Delete: `test_sample_rows_keep_the_index_of_the_table_they_came_from`, `test_reading_several_samples_at_once_raises`, `test_sample_keys_are_no_item_fields`, `test_a_window_is_keyed_by_its_last_step`. Add:

```python
def test_a_sample_is_one_row_whose_item_columns_are_lists(tiles):
    with pytest.warns(DroppedRowsWarning):
        samples = tiles.gs.sample(pair=["s2", "s1"], tolerance="2D", window=2)
    row = samples.iloc[0]
    assert len(row["id"]) == len(row["collection"]) == len(row["assets"]) == len(row["geometries"])
    assert row["collection"] == ["s2", "s1", "s2", "s1", "dem"]
    assert samples.crs == "EPSG:4326"

def test_a_sample_shape_is_where_its_items_overlap(tiles):
    samples = tiles.gs.sample()
    assert samples.geometry.iloc[0].equals(tiles.geometry.iloc[0])   # tile A
    assert not samples.geometry.is_empty.any()

def test_footprints_without_pair_are_one_sample_each(tiles):
    assert [len(ids) for ids in tiles.gs.sample()["id"]] == [10, 1]

def test_an_item_joins_every_footprint_it_covers_enough_of(catalog):
    # three footprints shifted 30% apart, overlap=0.5: the middle one joins both ends
    ...
    assert shifted.gs.sample(overlap=0.5)["id"].tolist() == [["a", "b"], ["b", "c"]]

def test_sample_an_empty_catalog(catalog):
    samples = catalog.iloc[:0].gs.sample()
    assert len(samples) == 0

def test_a_footprint_without_a_window_gives_no_sample(tiles):
    with pytest.warns(DroppedRowsWarning):
        samples = tiles.gs.sample(pair=["s2"], window=2)
    assert all(len(ids) > 1 for ids in samples["id"])    # tile B's single scene left no row

def test_a_sample_table_round_trips_through_parquet(tiles, tmp_path):
    samples = tiles.gs.sample()
    samples.to_parquet(tmp_path / "samples.parquet")
    back = gpd.read_parquet(tmp_path / "samples.parquet", to_pandas_kwargs={"types_mapper": pd.ArrowDtype})
    assert back["assets"].dtype == samples["assets"].dtype
```

Rewrite `test_a_region_is_a_place_that_summarizes_its_rows` as `samples.assign(items=samples["id"].list.len()).gs.explore(column="items")` returning an `Explorer`.

- [ ] **Step 2: Run them and confirm they fail on the old shape**

Run: `uv run pytest tests/geodata/core/vector -q`

- [ ] **Step 3: Rewrite `sample.py` and `GeoCatalog.sample`; delete `GeoCatalog.samples`**

- [ ] **Step 4: Run them and confirm they pass** (the row-opening tests come in Task 3)

Run: `uv run pytest tests/geodata/core/vector/test_sample.py tests/geodata/core/vector/test_catalog.py -q`

---

### Task 3: `GeoSamples` and `GeoSamplesItem` accessors

**Files:**

- Create: `src/geosave/geodata/core/vector/base.py` (`GeoVector`)
- Create: `src/geosave/geodata/core/vector/samples.py` (`GeoSamples`, `GeoSamplesItem`)
- Modify: `src/geosave/geodata/core/vector/catalog.py` (`GeoCatalog(GeoVector)`)
- Modify: `src/geosave/geodata/core/vector/item.py` (drop the `SAMPLE_KEYS` stripping)
- Modify: `src/geosave/geodata/core/register.py`
- Modify: `src/geosave/geodata/io/vector/stac_geoparquet.py` (delete the samples check and its import)
- Test: `tests/geodata/core/vector/test_samples.py` (create), `tests/geodata/core/test_register.py`, `tests/geodata/core/vector/test_item.py`

**Interfaces:**

- Consumes: `open_collections` (Task 1); `build_samples` output (Task 2).
- Produces: `GeoVector(data)`, with `crs`, `footprint`, `plot`, `explore`, `rasterize` moved unchanged from `GeoCatalog`, and the GeoDataFrame validation from `GeoCatalog.__init__`.
- Produces: `GeoSamples(GeoVector)`, with no members of its own.
- Produces: `GeoSamplesItem(row)`:
  - `to_items() -> list[pystac.Item]`: one Item per list position, from `_stac.to_item(pd.Series(fields))` with `fields["geometry"] = shapely.from_wkb(row["geometries"][k])`. `list` and `np.ndarray` values are both read (Review Focus 3).
  - `to_xarray() -> DataTree`: `open_collections(self.to_items(), collections=...)`, with the Collections read from `row.attrs["stac-geoparquet"]`.
- Produces: in `register.py`, the dataframe and series `gs` registrations call a function choosing the class by whether `id` is an Arrow list (frame) or a list/array (row). The function names say what they choose, e.g. `select_frame_accessor(data) -> GeoCatalog | GeoSamples`.

- [ ] **Step 1: Write failing tests**

```python
def test_a_sample_table_has_the_samples_accessor(tiles):
    samples = tiles.gs.sample()
    assert type(samples.gs) is GeoSamples
    assert type(samples.iloc[0].gs) is GeoSamplesItem
    assert type(tiles.gs) is GeoCatalog

def test_a_sample_row_opens_lazily_one_group_per_collection(tiles):
    with pytest.warns(DroppedRowsWarning):
        samples = tiles.gs.sample(pair=["s2", "s1"], tolerance="2D", window=2)
    tree = samples.iloc[0].gs.to_xarray()
    assert tree.gs.groups == ("s2", "s1", "dem")
    assert tree["s1"].dataset.gs.resolution.x == 20
    assert "time" not in tree["dem"].dataset.dims
    ...  # every variable has chunks

def test_a_sample_row_opens_what_its_items_open(tiles):
    row = tiles.gs.sample(pair=["s2"], window=2).iloc[0]
    expected = tiles[tiles["id"].isin(row["id"])].gs.to_xarray()
    tree = row.gs.to_xarray()
    for name in tree.gs.groups:
        assert tree[name].dataset.identical(expected[name].dataset)

def test_a_sample_with_an_all_empty_column_opens(tiles):
    row = tiles[tiles["collection"] == "s2"].gs.sample(pair=["s2"], window=2).iloc[0]
    assert row["start_datetime"] == [None, None]
    assert row.gs.to_xarray()["s2"].dataset.sizes["time"] == 2

def test_items_of_a_reprojected_catalog_keep_their_wgs84_bbox(tiles):
    row = tiles.to_crs(3857).gs.sample().iloc[0]
    expected = tiles[tiles["id"] == row["id"][0]].gs.to_items()[0].bbox
    assert row.gs.to_items()[0].bbox == pytest.approx(expected)

def test_a_selected_sample_row_still_opens(tiles):
    picked = tiles.gs.sample(pair=["s2"], window=2).iloc[[2, 0]]
    assert picked.attrs["stac-geoparquet"] == tiles.attrs["stac-geoparquet"]
    assert picked.iloc[0].gs.to_xarray().gs.groups == ("s2", "dem")

def test_a_sample_row_read_back_without_arrow_types_opens(tiles, tmp_path):
    tiles.gs.sample().to_parquet(tmp_path / "s.parquet")
    row = gpd.read_parquet(tmp_path / "s.parquet").iloc[1]
    row.attrs = tiles.attrs
    assert row.gs.to_items()[0].id == "s2_320000_20240308"

def test_samples_have_no_item_operations(tiles):
    samples = tiles.gs.sample()
    assert not hasattr(samples.gs, "to_items")
    assert not hasattr(samples.gs, "to_geoparquet")
```

Move `test_a_sample_reads_lazily_one_group_per_collection` from `test_sample.py` here (rewritten above). Delete `test_a_sample_row_reads_as_its_item` from `test_item.py`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/core/vector/test_samples.py tests/geodata/core/test_register.py -q`

- [ ] **Step 3: Extract `GeoVector`, add `GeoSamples`/`GeoSamplesItem`, change the registration, and remove the leftover sample code in `item.py` and `stac_geoparquet.py`**

- [ ] **Step 4: Run all geodata tests and basedpyright**

Run: `uv run pytest tests/geodata -q && uv run basedpyright src/geosave/geodata/core src/geosave/geodata/io`
Expected: pass. `grep -rn "SAMPLE_KEYS\|REGION_KEY\|WINDOW_KEY\|gs:region\|gs:window" src tests` returns nothing.

---

### Task 4: ModelSpec reads sample rows

**Files:**

- Modify: `src/geosave/model/spec/model.py` (`prepare` loop; `ingest`/`prepare` docstring examples)
- Test: `tests/model/spec/test_model.py`

**Interfaces:**

- Consumes: `GeoSamplesItem.to_xarray() -> DataTree` (Task 3); `DataTree.gs.align(target=...)`; `DataTree.gs.timespan`.
- Produces: `prepare` names each sample's folder and Item id prefix by its index label (`str(index)` from `iterrows()`), because `row["id"]` is now a list.

- [ ] **Step 1: Update the six model tests that fail on `main` today**

`samples.gs[i]` → `samples.iloc[i]`; `len(samples.gs)` → `len(samples)`; `samples["id"].is_unique` → `samples.explode("id")["id"].is_unique`. Add an assertion that two samples write two folders (Review Focus 4).

- [ ] **Step 2: Run them and confirm they fail inside `prepare`**

Run: `uv run pytest tests/model/spec/test_model.py -q`

- [ ] **Step 3: Rewrite the `prepare` loop**

```python
for index, row in raw.gs.sample().iterrows():
    sample_id = str(index)
    tree = row.gs.to_xarray()
    if target is not None:
        tree = tree.gs.align(target=target)
    rasters = tree.gs.rasters
    ...                       # unchanged, with datetime=tree.gs.timespan
```

- [ ] **Step 4: Run model and flows tests**

Run: `uv run pytest tests/model tests/flows -q`
Expected: pass. The baseline on `main` before Task 1 is 6 failures in `tests/model/spec/test_model.py`.

---

### Task 5: Docs and decision record

**Files:**

- Modify: `docs/contributing/architecture.md` (data flow paragraph; `core/vector` rules: GeoVector, GeoCatalog, GeoSamples, sample.py)
- Modify: `docs/guides/catalogs-and-cuts.md`, `docs/reference/catalog.md`, `docs/concepts/data-flow.md`, `docs/concepts/catalog-windows.md`, `docs/quickstart.md`, `docs/guides/stac.md`, `docs/guides/model-spec.md`
- Modify: `docs/contributing/specs/2026-10-31-samples-by-region.md` (Status: table shape superseded by this plan)
- Modify: `docs/contributing/open-decisions.md` (add: persisting sample tables. Plain GeoParquet keeps the list types, but asset hrefs stay absolute, unlike the item table's relative ones.)

- [ ] **Step 1: Replace each `gs.samples` / `get_group` / repeated-row description with the Design section's calls**

Check: `grep -rn "gs\.samples\|get_group\|samples\.gs\[\|gs:region\|gs:window" docs src README.md` returns nothing outside dated plans and specs.

- [ ] **Step 2: Run the full checks**

Run: `uv run pytest -q && uv run ruff check . && uv run pre-commit run --all-files`
Expected: pass, apart from failures that the baseline run (`main` before Task 1) also shows.
