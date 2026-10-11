# STAC Surface Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give users one consistent geodata surface. They `open_vector` catalogs, get item tables (GeoDataFrames) from every STAC builder and search, write them with `to_geoparquet`, and never handle pystac Items or the `stac.table`/`stac.item` modules.

**Architecture:**
- **Accessors.** `GeoVector` stays a pure vector accessor. `GeoCatalog(GeoVector)` adds STAC rules, and one `gs` accessor factory picks the class by the frame's columns.
- **Private modules.** `stac.item` becomes `stac._item`, and `stac.table` becomes `stac._table`.
- **Public builders.** `create_items`/`create_item` return item tables, and both clients' `search` returns an item table. `StacSource` reads pystac Items through a separate internal `search_items`.
- **Band names.** Every asset names its bands in `eo:bands`, so odc-stac names multi-band variables.

**Tech Stack:** pandas/geopandas accessors, pystac 1.14 (`EOExtension`), stac-geoparquet, rustac, odc-stac 0.5.2, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-cuts-windows-design.md` (sections "Accessors", "Public API", and the Band rows of "Decisions"). Plan 2 implements the cuts, ml and model sections and the removals.

## Global Constraints

- **Commands.** Run tests with `uv run pytest` and lint with `uv run ruff check .`.
- **Baseline before starting:** `1943 passed, 36 deselected, 3 xfailed`.
- **No compatibility aliases.** Don't keep old names, re-exports or duplicate paths. Every caller moves to the new name in the same task.
- **Renames are made file by file** with the Edit tool. Never use sed/regex bulk replacement.
- **Don't edit `*.ipynb`** (project rule). Notebook tests are `slow`/`integration` and deselected by default. Record any notebook that still says `read_vector` in the handoff.
- **Don't edit historical documents** under `docs/superpowers/plans/` and `docs/superpowers/specs/` other than the 2026-10-10 spec.
- **Don't commit.** The working tree holds unrelated user changes in the same files, so leave all changes uncommitted for the user.
- **Dependency direction.** `geodata` imports no torch. `core/vector.py` may import `stac` only inside a function body, which is what `tests/test_layering.py` checks.
- **Docstrings** are Google style and concise. Examples use `import geosave_engine as gs` or the `gs` accessor.

## Review Focus

1. **An item table loses a STAC column through column selection** (`rows[["id", "geometry"]]`). Expected: `.gs` is a plain `GeoVector` and writes plain GeoParquet. This is tested in Task 5.
2. **An empty search result.** Expected: an empty item table, not an exception from `from_items`. This is tested in Task 6.
3. **A plain DataFrame or GeoDataFrame without a geometry column**, read through `.gs`. Expected: still an `AttributeError`, so pandas reports that no accessor exists. This is tested in Task 5.
4. **A non-spectral raster such as a label or DEM.** Expected: it now writes `eo:bands` names, and reading those Items back adds no spurious `Spectral` attrs. This is tested in Task 2.
5. **`create_items` given a `.zarr` path or a mapping without `id`.** Expected: a `ValueError` that names the fix. This is tested in Task 4.

---

### Task 1: Rename `read_vector` to `open_vector`

**Files:**
- Modify (definition and exports):
  - `src/geosave_engine/geodata/io/readers.py`
  - `src/geosave_engine/geodata/io/__init__.py`
  - `src/geosave_engine/geodata/__init__.py`
  - `src/geosave_engine/__init__.py`
- Modify (callers and docs):
  - `src/geosave_engine/geodata/core/vector.py`
  - `src/geosave_engine/geodata/cuts/__init__.py`
  - `src/geosave_engine/ml/segmentation/supervised/data.py`
  - `src/geosave_engine/model/spec/model.py`
  - `src/geosave_engine/workflow/configs/anchor.py`
  - `src/geosave_engine/templates/workspaces/segmentation/README.md`
  - `docs/guides/architecture.md`
  - `README.md` and `AGENTS.md`, if they mention it
- Modify (tests):
  - `tests/cli/core/test_workspace.py`
  - `tests/examples/test_notebooks.py`
  - `tests/geodata/io/test_readers.py`
  - `tests/geodata/io/test_reference.py`
  - `tests/geodata/io/test_remote.py`
  - `tests/geodata/io/vector/test_geojson.py`
  - `tests/geodata/io/vector/test_geopackage.py`
  - `tests/geodata/io/vector/test_geoparquet.py`
  - `tests/geodata/stac/test_catalog.py`
  - `tests/geodata/stac/test_table.py`
  - `tests/ml/segmentation/supervised/test_data.py`
  - `tests/model/spec/test_prepare.py`

**Interfaces:**
- Produces: `geosave_engine.open_vector(source, **options) -> GeoDataFrame`, also exported from `geosave_engine.geodata` and `geosave_engine.geodata.io`. `read_vector` no longer exists.

- [ ] **Step 1: Write the failing test**

Add to `tests/geodata/io/test_readers.py`:

```python
def test_vectors_open_through_open_vector_and_read_vector_is_gone(tmp_path) -> None:
    import geosave_engine as gs

    plots = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    path = plots.gs.to_geoparquet(tmp_path / "plots.parquet")

    assert gs.open_vector(path)["name"].tolist() == ["a"]
    assert not hasattr(gs.geodata, "read_vector")
```

Add `import geopandas as gpd` and `from shapely.geometry import Point` at the top of the file if they are missing.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/geodata/io/test_readers.py::test_vectors_open_through_open_vector_and_read_vector_is_gone -v`
Expected: FAIL with `AttributeError: ... has no attribute 'open_vector'`.

- [ ] **Step 3: Rename the definition**

In `src/geosave_engine/geodata/io/readers.py`, rename `def read_vector(` to `def open_vector(`. Change the docstring summary to `"""Open a vector file through the reader its suffix names.`, and change the example to `>>> open_vector("plantations.geojson").gs.crs.to_epsg()`.

- [ ] **Step 4: Rename every other occurrence, file by file**

List the files that still mention the old name:

```bash
grep -rlw read_vector src tests README.md AGENTS.md docs/guides
```

Open each file and replace every `read_vector` with `open_vector` using the Edit tool, one file at a time. This covers imports, `__all__` entries, calls, docstrings and prose. The four `__init__.py` files must export `open_vector` in place of `read_vector`, with no alias.

- [ ] **Step 5: Verify that nothing is left**

Run: `grep -rnw read_vector src tests README.md AGENTS.md docs/guides`
Expected: no output.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest -q`
Expected: `1944 passed` (the baseline plus 1), 36 deselected, 3 xfailed.

Run: `uv run ruff check .`
Expected: `All checks passed!`

---

### Task 2: Name every band in `eo:bands`

**Files:**
- Modify: `src/geosave_engine/geodata/stac/extensions/eo.py` (the `write` function and the module docstring)
- Test: `tests/geodata/stac/extensions/test_eo.py`, `tests/geodata/stac/test_item.py`

**Interfaces:**
- Produces: every asset built by `create_items`/`create_item` carries `eo:bands` with one `name` per variable, in file order.

- [ ] **Step 1: Write the failing tests**

In `tests/geodata/stac/extensions/test_eo.py`, replace `test_a_raster_with_no_spectral_facts_writes_nothing` with:

```python
def test_a_raster_with_no_spectral_facts_still_names_its_bands() -> None:
    dem = build_raster()[["nir"]].astype("float32").rename(nir="height")
    asset = _asset()

    eo.write(asset, dem)

    (height,) = EOExtension.ext(asset).bands
    assert (height.name, height.common_name, height.center_wavelength) == (
        "height",
        None,
        None,
    )
    assert asset.owner.stac_extensions == [EOExtension.get_schema_uri()]
```

Change the module docstring of the test file to `"""The EO module names every band, adding spectral facts where a band has them."""`.

Add to `tests/geodata/stac/test_item.py`:

```python
def test_odc_stac_names_a_multiband_asset_after_its_bands(tmp_path) -> None:
    import odc.stac

    scene = build_raster()
    (built,) = stac.create_items(
        scene.gs.to_cog(tmp_path / "forest"), datetime=datetime(2025, 6, 1, tzinfo=UTC)
    )

    loaded = odc.stac.load([built], geobox=scene.gs.geobox, chunks={})

    assert list(loaded.data_vars) == ["red", "nir"]
```

If `test_item.py` doesn't already import `build_raster`, `datetime` and `UTC`, add `from tests.geodata.conftest import build_raster` and `from datetime import UTC, datetime`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/stac/extensions/test_eo.py tests/geodata/stac/test_item.py::test_odc_stac_names_a_multiband_asset_after_its_bands -v`
Expected:
- The eo test fails because `EOExtension.ext` raises `ExtensionNotImplemented`.
- The odc test fails with `['image.1', 'image.2'] != ['red', 'nir']`.

- [ ] **Step 3: Implement**

Replace the module docstring and `write` in `src/geosave_engine/geodata/stac/extensions/eo.py`:

```python
"""EO extension: what each band is called and where in the spectrum it sits."""
```

```python
def write(asset: pystac.Asset, raster: xr.Dataset) -> None:
    """Name one EO band per variable, adding spectral facts where it has them.

    Tools that read a catalog without opening its files, such as odc-stac,
    name a multi-band asset's variables from these names.

    Args:
        asset: Asset already added to its Item.
        raster: Raster as `open_raster` opens it.

    Examples:
        >>> write(asset, scene)
        >>> [band.name for band in EOExtension.ext(asset).bands]
        ['red', 'nir']
    """
    header = raster.gs.attrs
    bands = []
    for name in raster.gs.variables:
        variable = header.data_vars[name]
        facts = variable.get(Spectral)
        semantics = variable.get(CFVariable)
        bands.append(
            Band.create(
                name=name,
                common_name=None if facts is None else facts.common_name,
                center_wavelength=None if facts is None else facts.center_wavelength,
                full_width_half_max=None
                if facts is None
                else facts.full_width_half_max,
                description=None if semantics is None else semantics.long_name,
            )
        )
    EOExtension.ext(asset, add_if_missing=True).apply(bands)
```

- [ ] **Step 4: Run the targeted tests**

Run: `uv run pytest tests/geodata/stac -q`
Expected: the two new tests pass.

Some existing tests assert an exact `stac_extensions` list or an empty `extra_fields` for non-spectral assets. Update each one to include `EOExtension.get_schema_uri()`, or to expect the `eo:bands` names. The behaviour change is intended: names are now always written. Re-run until the folder passes.

- [ ] **Step 5: Confirm reading names adds no attrs**

`eo.read` on a band that states only its name returns `[]`, which `test_a_band_stating_only_its_name_reads_nothing` covers. Run that test to confirm it still passes:

Run: `uv run pytest tests/geodata/stac/extensions/test_eo.py -v`
Expected: all tests pass.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, and ruff passes.

---

### Task 3: Make `stac.item`/`stac.table` private, publish `get_collections`, and make the table client private

**Files:**
- Rename: `src/geosave_engine/geodata/stac/item.py` → `src/geosave_engine/geodata/stac/_item.py` (`git mv`)
- Rename: `src/geosave_engine/geodata/stac/table.py` → `src/geosave_engine/geodata/stac/_table.py` (`git mv`)
- Modify:
  - `src/geosave_engine/geodata/stac/__init__.py`
  - `src/geosave_engine/geodata/stac/client.py`
  - `src/geosave_engine/geodata/io/readers.py`
  - `src/geosave_engine/geodata/core/vector.py`
  - `src/geosave_engine/geodata/cuts/stacks.py`
  - `src/geosave_engine/ml/segmentation/supervised/data.py`
  - `src/geosave_engine/model/spec/model.py`
  - `src/geosave_engine/workflow/tasks/labels.py`
  - any other file the grep in Step 4 lists
- Test: every test file importing `geosave_engine.geodata.stac.table`/`.item`, or calling `read_collections`/`StacTableClient`

**Interfaces:**
- Produces:
  - `geosave_engine.geodata.stac.get_collections(path, *, storage_options=None) -> dict[str, pystac.Collection]`. This is the renamed `read_collections`.
  - `geosave_engine.geodata.stac.client._StacTableClient`, the renamed `StacTableClient`. It is no longer exported.
- Produces (internal, for Plan 2): `stac._table.from_items`, `to_items`, `write`, `write_catalog`, `read`, `read_catalog`, `resolve`, `parse_collections`, `load`, plus `stac._item._items`, `_open`, `_identity`, `default_key`.
- `gs.stac` exports exactly:
  - `create_collection`, `create_item`, `create_items`, `create_stack_items` (`create_stack_items` until Plan 2 removes it);
  - `get_collections`;
  - `StacClient`, `StacProvider`, `StacQuery`, `StacSource`, `StacSourceConfig`.

- [ ] **Step 1: Write the failing test**

Add to `tests/geodata/stac/test_table.py`:

```python
def test_the_stac_namespace_exposes_builders_and_clients_only() -> None:
    import geosave_engine as gs

    assert sorted(gs.stac.__all__) == [
        "StacClient",
        "StacProvider",
        "StacQuery",
        "StacSource",
        "StacSourceConfig",
        "create_collection",
        "create_item",
        "create_items",
        "create_stack_items",
        "get_collections",
    ]
    assert not hasattr(gs.stac, "table")
    assert not hasattr(gs.stac, "StacTableClient")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/geodata/stac/test_table.py::test_the_stac_namespace_exposes_builders_and_clients_only -v`
Expected: FAIL. `__all__` contains `item`, `table` and `StacTableClient`.

- [ ] **Step 3: Move the modules and rename**

```bash
git mv src/geosave_engine/geodata/stac/item.py src/geosave_engine/geodata/stac/_item.py
git mv src/geosave_engine/geodata/stac/table.py src/geosave_engine/geodata/stac/_table.py
```

In `_table.py`, rename `def read_collections(` to `def get_collections(`, including the recursive call inside it and its docstring example. In `client.py`, rename `class StacTableClient` to `class _StacTableClient` and update the reference in `StacClient.open`.

Replace `src/geosave_engine/geodata/stac/__init__.py` with:

```python
"""Build, read and search STAC: item tables in, item tables out."""

from ._item import create_collection, create_item, create_items, create_stack_items
from ._table import get_collections
from .client import StacClient, StacProvider
from .query import StacQuery
from .source import StacSource, StacSourceConfig

__all__ = [
    "StacClient",
    "StacProvider",
    "StacQuery",
    "StacSource",
    "StacSourceConfig",
    "create_collection",
    "create_item",
    "create_items",
    "create_stack_items",
    "get_collections",
]
```

- [ ] **Step 4: Update every import, file by file**

List the files that still use the old names:

```bash
grep -rnE "stac import (item|table)|stac\.(item|table)\b|from \. import (item|table)|from \.(item|table) import|read_collections|StacTableClient" src tests
```

For each file, edit with the Edit tool:
- `from geosave_engine.geodata.stac import table` → `from geosave_engine.geodata.stac import _table as table`. Inside `src/geosave_engine/geodata/stac/` use `from . import _table as table`. The local name `table` keeps the call sites unchanged.
- `from geosave_engine.geodata.stac import item` → `... import _item as item`.
- `stac.table.<name>` attribute access in tests → `from geosave_engine.geodata.stac import _table`, then `_table.<name>`.
- `read_collections` → `get_collections`.
- `StacTableClient` → `_StacTableClient`, imported from `geosave_engine.geodata.stac.client`.

Re-run the grep. Expected: only `_table`/`_item` forms remain.

- [ ] **Step 5: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, including the new one, and ruff passes.

---

### Task 4: `create_items` and `create_item` return item tables

**Files:**
- Modify: `src/geosave_engine/geodata/stac/_item.py` (`create_item`, `create_items`, and new helpers `_files` and `_group_items`)
- Modify (callers):
  - `src/geosave_engine/workflow/tasks/labels.py`
  - docstrings in `src/geosave_engine/geodata/stac/_table.py` and `src/geosave_engine/geodata/core/vector.py`
- Test:
  - `tests/geodata/stac/test_item.py`
  - callers in `tests/geodata/core/test_raster.py`
  - `tests/geodata/stac/test_table_client.py`
  - `tests/geodata/stac/test_dataflow.py`
  - `tests/geodata/stac/test_table.py`
  - `tests/geodata/io/test_remote.py`
  - `tests/geodata/cuts/test_stacks.py`

**Interfaces:**
- Consumes: `_table.from_items(items) -> GeoDataFrame` (Task 3).
- Produces:
  - `create_items(paths, *, id=None, collection=None, datetime=None, **options) -> GeoDataFrame`. `paths` is one path, a sequence of paths, or `Mapping[str, path | Sequence[path]]`. A mapping makes each key the collection of its rows, requires `id` (the stack's name), prefixes every id with `f"{id}/"`, and dates timeless groups by the span the dated groups cover.
  - `create_item(assets, *, id, collection=None, datetime=None, **options) -> GeoDataFrame`, a one-row table.
  - `create_stack_items` is unchanged; Plan 2 removes it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/stac/test_item.py`:

```python
def test_create_items_returns_an_item_table(scene, tmp_path) -> None:
    rows = stac.create_items(
        scene.gs.to_cog(tmp_path / "forest"), datetime=datetime(2025, 6, 1, tzinfo=UTC)
    )

    assert isinstance(rows, gpd.GeoDataFrame)
    assert rows["stac_version"].notna().all()
    assert rows["assets"].iloc[0]["image"]["href"].endswith(".tif")


def test_a_stack_mapping_makes_each_group_a_collection(tmp_path) -> None:
    image = build_raster(times=2)
    label = (image // 1000).astype("uint8")[["red"]].rename(red="label")
    saved = stack({"optical": image, "label": label.isel(time=0, drop=True)}).gs.to_cog(
        tmp_path / "s0"
    )

    rows = stac.create_items(saved, id="s0")

    assert rows["collection"].tolist() == ["optical", "optical", "label"]
    assert all(item_id.startswith("s0/") for item_id in rows["id"])
    timeless = rows[rows["collection"] == "label"].iloc[0]
    assert timeless["start_datetime"] is not None


def test_a_stack_mapping_needs_the_stack_name(tmp_path) -> None:
    saved = stack({"optical": build_raster(times=2)}).gs.to_cog(tmp_path / "s0")

    with pytest.raises(ValueError, match="id="):
        stac.create_items(saved)


def test_a_datacube_store_is_not_catalogued(scene, tmp_path) -> None:
    store = scene.gs.to_zarr(tmp_path / "cube.zarr")

    with pytest.raises(ValueError, match="to_cog"):
        stac.create_items(store, datetime=datetime(2025, 6, 1, tzinfo=UTC))


def test_create_item_returns_one_row(tmp_path) -> None:
    path = build_raster().gs.to_cog(tmp_path / "label")

    rows = stac.create_item(
        {"label": path[0] if isinstance(path, tuple) else path},
        id="s0",
        datetime=datetime(2025, 6, 1, tzinfo=UTC),
    )

    assert rows["id"].tolist() == ["s0"]
    assert list(rows["assets"].iloc[0]) == ["label"]
```

Add any missing imports: `import geopandas as gpd`, `from geosave_engine.geodata import stack`, `from tests.geodata.conftest import build_raster`. `scene` is the existing fixture in this file. If the file has none, use `build_raster(times=2)` inline.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/stac/test_item.py -k "item_table or mapping or datacube or one_row" -v`
Expected: FAIL. Today's functions return `tuple[pystac.Item, ...]` or a `pystac.Item`, and have no mapping or zarr checks.

- [ ] **Step 3: Implement in `_item.py`**

Add the imports `import os`, `from pathlib import Path`, `from typing import TYPE_CHECKING`, and `from . import _table`. Under `if TYPE_CHECKING:`, add `from geosave_engine.geodata import GeoDataFrame`.

Add next to `_MEDIA_TYPES`:

```python
# Stores holding a whole datacube, which a STAC Item, one scene, does not describe.
_CUBE_SUFFIXES = (".zarr", ".nc", ".nc4", ".cdf")
```

Replace the bodies and return types of `create_item` and `create_items`, keeping their Args docs. Update the Returns and Examples text to say "Item table" and use `rows["id"].tolist()`. Then add `_files` and `_group_items`:

```python
def create_item(
    assets: Mapping[str, str | PathLike[str]],
    *,
    id: str,
    collection: pystac.Collection | None = None,
    datetime: ItemTime | None = None,
    **options: Any,
) -> GeoDataFrame:
    with ExitStack() as opened:
        named = {key: _open(opened, path, options) for key, path in assets.items()}
        item = _item(named, id=id, collection=_identity(collection), when=datetime)
    return _table.from_items([item])


def create_items(
    paths: str
    | PathLike[str]
    | Sequence[str | PathLike[str]]
    | Mapping[str, str | PathLike[str] | Sequence[str | PathLike[str]]],
    *,
    id: str | None = None,
    collection: pystac.Collection | None = None,
    datetime: ItemTime | None = None,
    **options: Any,
) -> GeoDataFrame:
    if isinstance(paths, Mapping):
        if id is None:
            raise ValueError(
                "a {group: paths} mapping is one saved stack; name it with id=, "
                "which prefixes every Item id"
            )
        if collection is not None:
            raise ValueError(
                "a {group: paths} mapping names each group's collection; drop "
                "collection="
            )
        return _table.from_items(_group_items(paths, name=id, when=datetime, options=options))
    files = _files(paths)
    with ExitStack() as opened:
        saved = dict(_open(opened, path, options) for path in files)
        built = _items(saved, id=id, collection=_identity(collection), when=datetime)
    return _table.from_items(built)


def _files(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
) -> list[str | PathLike[str]]:
    """List the scene files a writer returned, refusing datacube stores."""
    files = [paths] if isinstance(paths, (str, PathLike)) else list(paths)
    if not files:
        raise ValueError("create_items needs the paths a writer returned")
    cubes = [
        str(path)
        for path in files
        if PurePosixPath(str(path).rstrip("/")).suffix.lower() in _CUBE_SUFFIXES
    ]
    if cubes:
        raise ValueError(
            f"{cubes} are datacube stores, which a STAC Item, one scene, does "
            f"not describe; save the scenes with to_cog to catalogue them"
        )
    return files


def _group_items(
    paths: Mapping[str, str | PathLike[str] | Sequence[str | PathLike[str]]],
    *,
    name: str,
    when: ItemTime | None,
    options: Mapping[str, Any],
) -> list[pystac.Item]:
    """Build every group's Items, timeless groups dated by what the dated cover."""
    with ExitStack() as opened:
        groups: dict[str, Saved] = {
            group: dict(_open(opened, path, options) for path in _files(saved))
            for group, saved in paths.items()
        }
        spans = [
            raster.gs.timespan
            for files in groups.values()
            for raster in files.values()
            if raster.gs.timespan is not None
        ]
        fallback = when
        if fallback is None and spans:
            fallback = (min(start for start, _ in spans), max(end for _, end in spans))

        items: list[pystac.Item] = []
        for group, files in groups.items():
            dated = any(raster.gs.timespan is not None for raster in files.values())
            items.extend(
                _items(
                    files,
                    id=None,
                    collection=group,
                    when=None if dated else fallback,
                    prefix=f"{name}/",
                )
            )
        return items
```

Leave `create_stack_items` as it is. It still returns pystac Items for Plan 2's current consumers.

- [ ] **Step 4: Update `workflow/tasks/labels.py`**

Replace the directory branch's building:

```python
        rows = []
        for sample_id, label_path in find_labels(path, pattern).items():
            with open_raster(label_path) as label:
                if label.gs.timespan is None:
                    raise ValueError(f"Label raster has no time: {label_path}")
            rows.append(create_item({"label": label_path}, id=sample_id))
        return GeoVector.concat(rows)
```

Add `GeoVector` to the `from geosave_engine.geodata import ...` line. Drop `table` from the stac import if it's now unused; `table.read` below still needs it, so keep it as `_table as table`.

- [ ] **Step 5: Update test callers that expect pystac Items, file by file**

List them:

```bash
grep -rnF -e "create_items(" -e "create_item(" tests
```

In each test that uses the result as pystac Items (iterating, `.validate()`, `.assets`, `.collection_id`, unpacking `(item,) = ...`), wrap it as `_table.to_items(stac.create_items(...))`, importing `from geosave_engine.geodata.stac import _table`. Where a test passes the result to `table.from_items(...)`, drop the `from_items` call, since the result is already a table. Tests passing `scene.gs.to_zarr(...)` to `create_items` (for example `test_item.py:265`) now expect the new `ValueError`: rewrite them as `pytest.raises(ValueError, match="to_cog")`, or delete them if `test_a_datacube_store_is_not_catalogued` covers the same case.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, and ruff passes.

---

### Task 5: `GeoCatalog` accessor; `to_stac_parquet` folds into `to_geoparquet`

**Files:**
- Create: `src/geosave_engine/geodata/stac/catalog.py`
- Modify: `src/geosave_engine/geodata/core/vector.py`:
  - drop `@register_dataframe_accessor` from the class;
  - add `ITEM_COLUMNS`, `is_item_table` and the `_gs` factory;
  - remove `to_stac_parquet`.
- Modify:
  - `src/geosave_engine/geodata/io/readers.py`: use `is_item_table`.
  - `src/geosave_engine/geodata/stac/_table.py`: use `is_item_table` in `write`, `write_catalog` and `resolve`, and drop its own `ITEM_COLUMNS`.
- Modify (callers): every `to_stac_parquet` call in src and tests.
- Test: `tests/geodata/stac/test_catalog.py`

**Interfaces:**
- Consumes: `_table.write`, `_table.write_catalog` (Task 3).
- Produces:
  - `geosave_engine.geodata.core.vector.is_item_table(data: pd.DataFrame) -> bool`. It is True when the columns `type`, `stac_version`, `id` and `assets` are all present.
  - `GeoCatalog(GeoVector)` in `geosave_engine.geodata.stac.catalog`:
    - `to_geoparquet(path, *, collections=None, overwrite=False, storage_options=None) -> Path | str`. A path without a suffix writes a catalog folder, one file per Collection.
    - `to_geopackage(...)` and `to_geojson(...)` raise `ValueError`.
  - `GeoVector.to_stac_parquet` no longer exists.

- [ ] **Step 1: Write the failing tests**

Append to `tests/geodata/stac/test_catalog.py`:

```python
import geopandas as gpd
from shapely.geometry import Point

from geosave_engine.geodata import GeoVector
from geosave_engine.geodata.stac.catalog import GeoCatalog
from geosave_engine.geodata.stac import _table


def test_gs_is_a_catalog_on_an_item_table_and_a_vector_otherwise(rows) -> None:
    plots = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:4326")

    assert type(rows.gs) is GeoCatalog
    assert type(plots.gs) is GeoVector
    assert type(rows[["id", "geometry"]].gs) is GeoVector


def test_a_catalog_writes_stac_geoparquet_that_validates(rows, tmp_path) -> None:
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    for item in _table.to_items(open_vector(path)):
        item.validate()


def test_a_catalog_refuses_formats_that_cannot_hold_assets(rows, tmp_path) -> None:
    with pytest.raises(ValueError, match="GeoParquet"):
        rows.gs.to_geopackage(tmp_path / "catalog.gpkg")
    with pytest.raises(ValueError, match="GeoParquet"):
        rows.gs.to_geojson(tmp_path / "catalog.geojson")


def test_a_plain_frame_still_has_no_gs_accessor() -> None:
    import pandas as pd

    with pytest.raises(AttributeError):
        pd.DataFrame({"a": [1]}).gs
```

`rows` is the existing fixture in `test_catalog.py`. Change `open_vector` in the import line at the top of the file if Task 1 hasn't already.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/stac/test_catalog.py -v`
Expected: FAIL with `ModuleNotFoundError: geosave_engine.geodata.stac.catalog`.

- [ ] **Step 3: Add the factory to `core/vector.py`**

Remove the `@pd.api.extensions.register_dataframe_accessor("gs")` line above `class GeoVector:`. Then delete the whole `to_stac_parquet` method. Because the class no longer mentions item tables or `to_stack` writing, change its docstring sentence "`to_stac_parquet` and `to_stack` read an item table, ..." to "`to_stack` reads an item table, a frame of STAC Items." Then append at module end:

```python
# Columns every STAC Item row carries; their presence makes a frame an item table.
ITEM_COLUMNS = ("type", "stac_version", "id", "assets")


def is_item_table(data: pd.DataFrame) -> bool:
    """Return whether a frame's rows are STAC Items.

    Args:
        data: Any DataFrame.

    Returns:
        True when the columns every STAC Item row carries are all present.
    """
    return all(column in data.columns for column in ITEM_COLUMNS)


@pd.api.extensions.register_dataframe_accessor("gs")
def _gs(data: pd.DataFrame) -> GeoVector:
    """Read a frame through `GeoCatalog` when its rows are STAC Items."""
    if isinstance(data, gpd.GeoDataFrame) and is_item_table(data):
        # Deferred: only an item table needs the STAC dependencies.
        from geosave_engine.geodata.stac.catalog import GeoCatalog

        return GeoCatalog(data)
    return GeoVector(data)
```

- [ ] **Step 4: Create `stac/catalog.py`**

```python
"""The `gs` accessor of an item table: a vector whose rows are STAC Items."""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, NoReturn

from geosave_engine.geodata.core.vector import GeoVector

from . import _table

if TYPE_CHECKING:
    import pystac

    from geosave_engine.geodata.io.storage import StorageOptions


class GeoCatalog(GeoVector):
    """GeoSave operations on an item table, read through `gs`.

    An item table is a GeoDataFrame of STAC Items, as `open_vector` reads a
    stac-geoparquet file and `stac.create_items` builds one. It is a vector,
    so `query` and `rasterize` apply. It is written only as stac-geoparquet,
    the format that keeps its assets.

    Examples:
        >>> rows = gs.open_vector("data/catalog.parquet")
        >>> rows.gs.to_geoparquet("data/copy.parquet")
        PosixPath('data/copy.parquet')
    """

    def to_geoparquet(
        self,
        path: str | PathLike[str],
        *,
        collections: Iterable[pystac.Collection] | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Any,
    ) -> Path | str:
        """Write this item table as stac-geoparquet.

        A path ending in `.parquet` or `.geoparquet` writes one file. A path
        with no suffix writes a catalog: a folder with one file for each
        Collection, which `open_vector` reads back as one table. Asset hrefs
        are stored relative to the file.

        Args:
            path: Output file, or a local catalog folder.
            collections: Collections the rows belong to, stored with the
                extent of their rows. None stores none.
            overwrite: Replace existing files when true.
            storage_options: Options for the filesystem a URL names.
            **options: Not accepted; stac-geoparquet fixes its own encoding.

        Returns:
            The file path or URL written, or the catalog folder.

        Raises:
            FileExistsError: A file exists and `overwrite` is false.
            ValueError: A catalog folder is given `storage_options`, a row
                names a collection absent from `collections`, or options are
                passed.

        Examples:
            >>> rows.gs.to_geoparquet("data/forest.parquet", collections=[forest])
            PosixPath('data/forest.parquet')
        """
        if options:
            raise ValueError(
                f"stac-geoparquet fixes its own encoding; {sorted(options)} do not apply"
            )
        if not PurePath(str(path)).suffix:
            if storage_options is not None:
                raise ValueError(
                    "a catalog folder is local; storage_options apply to one file"
                )
            return _table.write_catalog(
                self._data, path, collections=collections, overwrite=overwrite
            )
        return _table.write(
            self._data,
            path,
            collections=collections,
            overwrite=overwrite,
            storage_options=storage_options,
        )

    def to_geopackage(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refuse: GeoPackage cannot hold an item table's assets."""
        raise ValueError(
            "an item table is STAC; GeoPackage cannot hold its assets, write "
            "GeoParquet with to_geoparquet"
        )

    def to_geojson(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refuse: GeoJSON features here would lose the STAC table encoding."""
        raise ValueError(
            "an item table is STAC; write it as GeoParquet with to_geoparquet"
        )
```

- [ ] **Step 5: Use `is_item_table` in readers and `_table`**

- **`readers.py` `open_vector`:** replace `if "stac_version" not in frame or "assets" not in frame:` with `if not is_item_table(frame):`, importing `from geosave_engine.geodata.core.vector import is_item_table` inside the function, next to the existing deferred import.
- **`_table.py`:**
  - Delete `ITEM_COLUMNS = (...)`.
  - Add `from geosave_engine.geodata.core.vector import is_item_table`.
  - In `write` and `write_catalog`, replace `if any(column not in rows for column in ITEM_COLUMNS):` with `if not is_item_table(rows):`, and change their messages to `"to_geoparquet writes an item table as stac-geoparquet; this frame has no STAC Item columns"`.
  - In `resolve`, use `or not is_item_table(rows)`.

- [ ] **Step 6: Move `to_stac_parquet` callers, file by file**

List them:

```bash
grep -rn "to_stac_parquet" src tests docs/guides README.md AGENTS.md
```

Replace each `.gs.to_stac_parquet(` with `.gs.to_geoparquet(`, and each prose mention in docstrings or docs with `to_geoparquet`. Re-run the grep. Expected: no output.

- [ ] **Step 7: Describe every collection on write**

A row naming a collection that `collections=` doesn't describe gets a minimal Collection: its id, a description equal to its id, and the extent of its rows. This replaces today's `ValueError` in `write` and `write_catalog`. Test, added to `tests/geodata/stac/test_catalog.py`:

```python
def test_collections_rows_name_are_described_when_not_given(rows, tmp_path) -> None:
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    described = gs.stac.get_collections(path)

    assert sorted(described) == sorted(rows["collection"].dropna().unique())
    assert all(c.extent.spatial.bboxes for c in described.values())
```

Import `geosave_engine as gs` in the test file. Run the test and confirm it fails with `[] != [...]`.

Then, in `_table.write`, build `described` from the given collections. For every `collection_id` named by the rows and absent from `described`, add `pystac.Collection(collection_id, collection_id, pystac.Extent.from_items(members))`, where `members` are the stored Items with that id, then proceed as today. Delete the `"is not among the collections given"` raise from both `write` and `write_catalog`, and any tests asserting it. In `write_catalog`, pass `collections=[described[name]] if name in described else None` per file.

- [ ] **Step 8: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, including `tests/test_layering.py`, and ruff passes.

---

### Task 6: `search` returns an item table; `StacSource` reads Items through `search_items`

**Files:**
- Modify: `src/geosave_engine/geodata/stac/client.py`. On both clients, rename the current `search` to `search_items` and add a new `search`.
- Modify: `src/geosave_engine/geodata/stac/_table.py`, adding `empty()`.
- Modify: `src/geosave_engine/geodata/stac/source.py`. The `SearchClient` protocol method becomes `search_items`, and so do its two call sites (`sample_item`, `load`).
- Test:
  - `tests/geodata/stac/test_table_client.py`
  - `tests/geodata/stac/test_client.py`
  - the fakes in `tests/conftest.py` (the `EmptyCatalog.search` and `LocalCatalog.search` methods) and in `tests/geodata/stac/test_source.py` (`FakeClient.search`)

**Interfaces:**
- Produces:
  - `StacClient.search(query) -> GeoDataFrame` and `_StacTableClient.search(query) -> GeoDataFrame`. Each returns an item table, and an empty result is an empty item table.
  - `StacClient.search_items(query) -> list[pystac.Item]`, and the same on `_StacTableClient`. These are internal, used by `StacSource`.
  - `_table.empty() -> GeoDataFrame`: zero rows, WGS84, with `ITEM_COLUMNS` and `geometry`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/stac/test_table_client.py`:

```python
def test_search_returns_an_item_table(tmp_path) -> None:
    forest = stac.create_collection("forest", description="Forest samples")
    rows = stac.create_items(
        build_raster(times=2).gs.to_cog(tmp_path / "forest"), collection=forest
    )
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet", collections=[forest])
    client = StacClient.open(path)

    found = client.search(StacQuery(collections=["forest"]))
    none = client.search(StacQuery(collections=["absent"]))

    assert found["id"].tolist() == rows["id"].tolist()
    assert type(found.gs).__name__ == "GeoCatalog"
    assert none.empty and type(none.gs).__name__ == "GeoCatalog"
```

Add the imports `from geosave_engine.geodata.stac import StacClient, StacQuery` and `from tests.geodata.conftest import build_raster` if they are missing.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/geodata/stac/test_table_client.py::test_search_returns_an_item_table -v`
Expected: FAIL with `AttributeError: 'list' object has no attribute ...`.

- [ ] **Step 3: Implement**

In `_table.py`, add:

```python
def empty() -> GeoDataFrame:
    """Return an item table with no rows, as a search matching nothing gives."""
    from geosave_engine.geodata.core.vector import ITEM_COLUMNS

    return cast(
        "GeoDataFrame",
        gpd.GeoDataFrame(
            columns=[*ITEM_COLUMNS, "geometry"], geometry="geometry", crs="EPSG:4326"
        ),
    )
```

In `client.py`:
- On **both** classes, rename the existing `def search(` to `def search_items(`. Change its docstring summary to `"""Run one search to completion as pystac Items, for loaders."""`.
- Add to each class:

```python
    def search(self, query: StacQuery | dict[str, Any]) -> GeoDataFrame:
        """Run one search to completion as an item table.

        Args:
            query: Query object, or raw search parameters.

        Returns:
            One row per matching Item, read through `gs` as a catalog. Empty
            where nothing matches.

        Examples:
            >>> client.search(StacQuery(collections=["forest"]))["id"].tolist()
            ['forest_20250601T000000']
        """
        found = self.search_items(query)
        return table.from_items(found) if found else table.empty()
```

Add `from geosave_engine.geodata import GeoDataFrame` under a `TYPE_CHECKING` block.

In `source.py`:
- Change the `SearchClient` protocol method to `def search_items(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]: ...`.
- Replace the two `self.client.search(` calls with `self.client.search_items(`.

- [ ] **Step 4: Update fakes and existing client tests**

- In `tests/conftest.py`, rename the `search` methods of `EmptyCatalog` and `LocalCatalog` to `search_items`. Leave the HTTP handler's `search(self, params)` alone; it is the fake STAC API server.
- In `tests/geodata/stac/test_source.py`, rename `FakeClient.search` to `search_items`.
- In `tests/geodata/stac/test_table_client.py` and `tests/geodata/stac/test_client.py`, change existing calls that treat the result as pystac Items (`item.id for item in client.search(...)`, `== []`, tuple unpacking) to `client.search_items(...)`.

- [ ] **Step 5: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, and ruff passes.

---

## Handoff

Report:
1. **What changed.**
2. **Checks run:** the final `uv run pytest -q` counts and ruff.
3. **Breaking changes:**
   - `read_vector`;
   - the `stac.item` and `stac.table` modules;
   - `StacTableClient` export;
   - `read_collections`;
   - `to_stac_parquet`;
   - `create_items`/`create_item` return types;
   - `search` return type;
   - non-spectral assets now carry `eo:bands`.
4. **Notebooks still mentioning `read_vector`:** `examples/01_imagery.ipynb`, `examples/02_prepare.ipynb`. They are left untouched by rule.

Plan 2 (cuts windows, `GeoWindow`, merge from COGs, ml/model consumers, and the removals) is written after this plan lands, against these final names.
