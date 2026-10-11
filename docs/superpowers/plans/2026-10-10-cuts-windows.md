# Cuts Windows Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Writers state what they wrote, catalogues are built from those records without guessing, `cuts` turns an item table into self-contained window rows that read themselves (`row.gs.to_stack()`), and predictions go back to georeferenced COGs that merge and catalogue like any other data.

**Architecture:**
- **Writer records.** `to_cog` returns `tuple[CogFile, ...]`, one typed record per file (`href`, `bands`, `datetime`, `collection`).
- **Reading.** A single reader, `io.raster.cogs.read(files, geobox=...)`, opens `CogFile` records, whether they come from a writer or from catalogue Items.
- **Cuts.** `cuts.stacks`/`frames`/`chips` map table to table. A row stores its member Items as JSON in `sources`, and the `GeoSample` Series accessor reads it.
- **Consumers.** `ModelSpec.prepare`, the segmentation `Dataset`, `Module`'s evaluation merge and `ChipWriter` move onto window rows.
- **Removals.** Zarr items, `geosave:stack`, the datacube extension and `StacMetadata` are removed.

**Tech Stack:** xarray, geopandas, pystac 1.14, stac-geoparquet, odc-geo, tiler, orjson, Lightning, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-cuts-windows-design.md`. Plan 1 (`docs/superpowers/plans/2026-10-10-stac-surface.md`) is implemented.

## Global Constraints

- **Commands.** Run tests with `uv run pytest` and lint with `uv run ruff check .`.
- **Baseline:** `1969 passed, 37 deselected, 3 xfailed`.
- **Green checkpoints.** Tasks 1–4 each end with the full suite green. Tasks 5–11 rebuild cuts and its consumers together, and only the tests each task names must pass during them. Task 12 removes the old paths and restores a full green suite. Say this explicitly in the ledger for each of Tasks 5–11.
- **No compatibility aliases** and no duplicate execution paths after Task 12. Transitional code from Tasks 1–11 is listed in Task 12 and deleted there.
- **Renames are made file by file** with the Edit tool. Never use sed/regex bulk replacement.
- **Don't edit `*.ipynb`.** Don't edit historical files under `docs/superpowers/` other than the 2026-10-10 spec.
- **Don't commit.** The working tree holds the user's uncommitted work, and another session edits `io/raster/cogs.py`, `core/raster.py` and tests concurrently. Re-read a file before editing it, and never revert changes you didn't make.
- **Layering.** `geodata` imports no torch. `io` imports no `stac` code. `core/vector.py` imports `stac` only inside functions.
- **Docstrings** are Google style and concise. Use the `gs` accessor in examples.

## Review Focus

1. **A chip at the parent window's corner with `mode="reflect"`.** Expected: real pixels inside the parent and reflected pixels outside it, on the chip's own grid (Task 5 `test_a_reflect_halo_fills_only_past_the_parent`).
2. **Two items of one collection covering the anchor at the same instant** (overlapping Sentinel-2 tiles). Expected: one kept and a `DuplicateInstantsWarning` (Task 5).
3. **An item that only intersects the anchor without covering it.** Expected: it does not join, and `stacks` warns naming it, so a near-miss isn't dropped silently (Task 5).
4. **A misspelled `on=` collection.** Expected: a `ValueError` naming the collections present, not an empty table (Task 5).
5. **A windows table saved to GeoParquet and read back in another process or folder.** Expected: rows read the same pixels, because hrefs are absolute and `sources` is JSON text (Task 7).

---

## Phase A: writer records, one reader, STAC cleanup (suite green after each task)

### Task 1: `to_cog` returns `CogFile` records; `{group}` becomes `{collection}`

**Files:**
- Modify: `src/geosave_engine/geodata/io/raster/cogs.py`:
  - rename the internal `CogFile` NamedTuple to `_Planned`;
  - add the public `CogFile` dataclass;
  - `layout(..., collection=)`, with the built-in field `collection` replacing `group`;
  - `write`/`write_files` return `tuple[CogFile, ...]`.
- Modify: `src/geosave_engine/geodata/core/raster.py` (`GeoRaster.to_cog` return type and docs), `src/geosave_engine/geodata/core/array.py` (`to_cog` docs), `src/geosave_engine/geodata/core/stack.py` (`GeoStack.to_cog` returns a flat `tuple[CogFile, ...]`, template field `{collection}`, messages).
- Modify: `src/geosave_engine/geodata/io/__init__.py`, `src/geosave_engine/geodata/io/raster/__init__.py` (export `CogFile`).
- Test: `tests/geodata/io/raster/test_cogs.py`, plus every test that unpacks a `to_cog` result (`grep -rln "to_cog(" tests`).

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class CogFile:
    href: str                         # absolute local path or URL as written
    bands: tuple[str, ...]            # variables it holds, in band order
    datetime: pd.Timestamp | None     # instant; None when undated
    collection: str | None = None     # stack layer it was written from
```

  `GeoRaster.to_cog(...)`, `GeoArray.to_cog(...)` and `GeoStack.to_cog(...)` each return `tuple[CogFile, ...]`, in time-then-band order (a stack's layers in layer order). A stack's template field is `{collection}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/geodata/io/raster/test_cogs.py`:

```python
def test_to_cog_returns_one_record_per_file(tmp_path) -> None:
    from geosave_engine.geodata.io import CogFile

    scene = build_raster(times=2)

    files = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}_{{band}}.tif")

    assert all(isinstance(file, CogFile) for file in files)
    assert [f.bands for f in files] == [("red",), ("nir",), ("red",), ("nir",)]
    assert len({f.href for f in files}) == 4
    assert files[0].datetime == pd.Timestamp(scene.time.values[0])
    assert files[0].collection is None


def test_an_unsplit_file_lists_every_band_in_order(tmp_path) -> None:
    (file,) = build_raster().gs.to_cog(f"{tmp_path}/scene.tif")

    assert file.bands == ("red", "nir")
    assert file.datetime is None
    assert Path(file.href).is_absolute()


def test_a_stack_names_each_file_by_its_collection(tmp_path) -> None:
    image = build_raster(times=1)
    label = image.isel(time=0, drop=True)[["red"]].rename(red="label")
    sample = stack({"optical": image, "label": label})

    files = sample.gs.to_cog(
        {"optical": f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif",
         "label": f"{tmp_path}/{{collection}}.tif"}
    )

    assert [f.collection for f in files] == ["optical", "label"]
    assert Path(files[0].href).name.startswith("optical_")
    assert Path(files[1].href).name == "label.tif"
```

Add any missing imports: `from pathlib import Path`, `import pandas as pd`, `from geosave_engine.geodata import stack`, `from tests.geodata.conftest import build_raster`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/io/raster/test_cogs.py -k "record or every_band or collection" -v`
Expected: FAIL, with `ImportError: cannot import name 'CogFile' from 'geosave_engine.geodata.io'` or an `AttributeError` on `.bands`.

- [ ] **Step 3: Implement in `cogs.py`**

- **Internal record.** Rename the NamedTuple `CogFile` to `_Planned` (same fields `path`, `raster`, `time`) everywhere in the module. Add `collection: str | None` to it.
- **Built-in fields.** Change `BUILT_IN = ("time", "band", "group")` to `BUILT_IN = ("time", "band", "collection")`. In the module docstring, change "`group`, a stack's group" to "`collection`, a stack's layer".
- **`layout`.** Rename the parameter `group` to `collection`. Inside, set `built_in["collection"] = collection`, and return `_Planned(path, rasters[path][names], times_of[path], collection)`.
- **Public record.** Add, after the imports:

```python
@dataclass(frozen=True)
class CogFile:
    """One Cloud Optimized GeoTIFF a write produced.

    Args:
        href: Absolute local path or URL of the file.
        bands: Variables the file holds, in band order.
        datetime: Instant the file holds, or None when undated.
        collection: Stack layer the file was written from, or None for a lone
            raster.

    Examples:
        >>> scene.gs.to_cog("out/s_{time:%Y%m%d}.tif")[0]
        CogFile(out/s_20250601.tif, bands=['red', 'nir'], 2025-06-01, -)
    """

    href: str
    bands: tuple[str, ...]
    datetime: pd.Timestamp | None
    collection: str | None = None

    def __repr__(self) -> str:
        when = "undated" if self.datetime is None else f"{self.datetime:%Y-%m-%d}"
        return (
            f"CogFile({self.href}, bands={list(self.bands)}, {when}, "
            f"{self.collection or '-'})"
        )
```

  Add `from dataclasses import dataclass` and `from ..storage import absolute_location`.
- **`write_files`.** Change the return type to `tuple[CogFile, ...]`, and build each record:

```python
        written.append(
            CogFile(
                href=absolute_location(written_path),
                bands=tuple(str(name) for name in file.raster.data_vars),
                datetime=file.time,
                collection=file.collection,
            )
        )
```

  Update the docstrings of `write` and `write_files`: "Returns: One record per file written, in time then variable order."

- [ ] **Step 4: Implement in the accessors**

- **`core/raster.py` `GeoRaster.to_cog`.** Return type `tuple[CogFile, ...]` under `TYPE_CHECKING` (`from geosave_engine.geodata.io.raster.cogs import CogFile`). In the docstring Returns, say "One `CogFile` per file written". The example shows a `CogFile` repr.
- **`core/array.py` `GeoArray.to_cog`.** Same return type; its example becomes `>>> ds["ndvi"].gs.to_cog("ndvi_{time:%Y%m%d}.tif")[0].bands` → `('ndvi',)`.
- **`core/stack.py` `GeoStack.to_cog`.**
  - Return type `tuple[CogFile, ...]`.
  - Call `cogs.layout(raster, templates[name], collection=name, fields=fields)`.
  - Every message and docstring mention of `{group}` becomes `{collection}`.
  - The duplicate-file error says `add {collection} to the path, as out/{collection}.tif`.
  - Return the concatenation, layer by layer:

```python
        return tuple(
            file
            for files in planned.values()
            for file in cogs.write_files(
                files,
                map_scale=map_scale,
                overwrite=overwrite,
                storage_options=storage_options,
                **options,
            )
        )
```

  - The Returns doc becomes "One `CogFile` per file, layers in order, each naming its layer as `collection`." Remove "which `open_stack` reads back"; Task 2 restores reading.
- **Exports.** Export `CogFile` from `geosave_engine.geodata.io.raster` and `geosave_engine.geodata.io`, alphabetically in `__all__`.

- [ ] **Step 5: Move every `to_cog` caller, file by file**

List them: `grep -rn "to_cog(" src tests --include=*.py`. In each caller:
- **`(path,) = x.gs.to_cog(...)`, `paths[0]`, and similar.** Read `.href` from the record: `(file,) = ...; path = file.href`.
- **Stack results used as a `{group: paths}` dict** (`saved["optical"]`, `.items()`). Use `[f for f in files if f.collection == "optical"]`.
- **Results passed to `stac.create_items` or `create_stack_items`.** Leave them for Task 3; in this task, pass `[f.href for f in files]` so they keep working.
- **Results passed to `open_stack`/`open_raster`.** Pass `[f.href for f in files]` for now; Task 2 makes `CogFile` sequences readable directly.
- **Templates using `{group}`.** Change them to `{collection}` (`grep -rn "{group}" src tests docs/guides`).

`create_stack_items(paths, name=...)` takes `{group: paths}`. Where a test feeds it a stack's `to_cog` result, build that mapping inline:

```python
by_layer = {}
for f in files:
    by_layer.setdefault(f.collection, []).append(f.href)
```

The same applies to `tests/geodata/cuts/conftest.py` `items`. Task 3 replaces it.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, and ruff passes.

---

### Task 2: one reader for `CogFile` records; single-path `open_raster`

**Files:**
- Modify: `src/geosave_engine/geodata/io/raster/cogs.py` (add `read`).
- Modify: `src/geosave_engine/geodata/io/readers.py`:
  - `open_raster` accepts a path or a `Sequence[CogFile]`;
  - `open_stack` accepts a store directory, a `.safe` product, or a `Sequence[CogFile]`;
  - remove the multi-path branch, the folder-of-COGs branch, `_combine_rasters`, and `open_stack`'s mapping branch and COG-folder groups.
- Test: `tests/geodata/io/raster/test_cogs.py`, `tests/geodata/io/test_readers.py` (delete or rewrite the multi-path and folder tests), `tests/geodata/core/test_stack.py`, `tests/geodata/io/raster/test_zarr_stack.py` (keep its Zarr cases).

**Interfaces:**
- Consumes: `CogFile` (Task 1).
- Produces:
  - **`cogs.read(files, *, geobox=None, resampling="nearest", **options) -> Dataset`.** Files sharing a `datetime` are merged with `merge_bands`, after each is reprojected onto `geobox` when one is given. When more than one distinct `datetime` is present, the instants join along `time` with `concat_time`, in chronological order. The result keeps each file's own attrs under the metadata merge rules, and it is lazy.
  - **`open_raster(source: str | PathLike | Sequence[CogFile], **options) -> Dataset`.** A sequence goes to `cogs.read`.
  - **`open_stack(source: str | PathLike | Sequence[CogFile], **options) -> DataTree`.** A sequence groups by `collection` (in first-seen order) and reads each with `cogs.read`. A `None` collection raises `ValueError("these files name no collection; open_raster reads a lone raster")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/geodata/io/raster/test_cogs.py`:

```python
def test_records_read_back_as_the_raster_written(tmp_path) -> None:
    from geosave_engine.geodata import open_raster

    scene = build_raster(times=2)
    files = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}_{{band}}.tif")

    restored = open_raster(files)

    assert list(restored.data_vars) == ["red", "nir"]
    assert restored.sizes["time"] == 2
    np.testing.assert_array_equal(restored.red.values, scene.red.values)


def test_a_stack_reads_back_from_its_records(tmp_path) -> None:
    from geosave_engine.geodata import open_stack

    image = build_raster(times=1)
    label = image.isel(time=0, drop=True)[["red"]].rename(red="label")
    files = stack({"optical": image, "label": label}).gs.to_cog(
        {"optical": f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif",
         "label": f"{tmp_path}/{{collection}}.tif"}
    )

    restored = open_stack(files)

    assert restored.gs.groups == ("optical", "label")
    assert "time" not in restored["label"].dims


def test_records_read_onto_a_grid_stay_lazy(tmp_path) -> None:
    import dask

    from geosave_engine.geodata.io.raster import cogs

    scene = build_raster(times=2)
    files = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")
    grid = scene.gs.geobox

    read = cogs.read(files, geobox=grid, chunks={})

    assert dask.is_dask_collection(read.red)
    assert read.gs.geobox == grid
```

Add `import numpy as np` if it is missing.

Append to `tests/geodata/io/test_readers.py`:

```python
def test_open_raster_takes_one_path_not_a_list(tmp_path) -> None:
    (file,) = build_raster().gs.to_cog(f"{tmp_path}/scene.tif")

    with pytest.raises(TypeError, match="CogFile"):
        open_raster([file.href, file.href])
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/io/raster/test_cogs.py tests/geodata/io/test_readers.py -k "records or one_path" -v`
Expected: FAIL. `cogs.read` doesn't exist, and `open_raster` of `CogFile`s raises inside `PurePath(...)`.

- [ ] **Step 3: Implement `cogs.read`**

```python
def read(
    files: Sequence[CogFile],
    *,
    geobox: GeoBox | None = None,
    resampling: str = "nearest",
    **options: Any,
) -> Dataset:
    """Open the records one write produced, or one Item lists, as one raster.

    Files of one instant are merged band by band, after each is placed on
    `geobox` when one is given; instants join along `time` in order.

    Args:
        files: Records of one raster: one layer, any instants and bands.
        geobox: Grid every file is reprojected onto. None keeps the files'
            own grid, which they must share.
        resampling: Kernel `reproject` applies where a file is reprojected.
        **options: Read options forwarded to the GeoTIFF reader.

    Returns:
        Lazy Dataset, with variables in band order and `time` present only
        when the files span more than one instant.

    Raises:
        ValueError: No file is given, or files of one instant repeat a band.
    """
    # Deferred: transform imports core, which imports io.
    from geosave_engine.geodata.transform.concat import concat_time
    from geosave_engine.geodata.transform.merge import merge_bands
    from geosave_engine.geodata.transform.warp import reproject

    from .gdal import read as read_tiff

    if not files:
        raise ValueError("read needs at least one CogFile")
    instants: dict[pd.Timestamp | None, list[CogFile]] = {}
    for file in files:
        instants.setdefault(file.datetime, []).append(file)

    scenes = []
    for when in sorted(instants, key=lambda t: (t is None, t)):
        bands = []
        for file in instants[when]:
            raster = read_tiff(file.href, **{"chunks": {}, **options})
            if geobox is not None:
                raster = reproject(raster, geobox, resampling=resampling)
            bands.append(raster)
        scene = merge_bands(bands) if len(bands) > 1 else bands[0]
        if when is not None and len(instants) > 1 and TIME_COORDINATE not in scene.dims:
            scene = scene.expand_dims({TIME_COORDINATE: [when]})
        scenes.append(scene)
    return cast("Dataset", concat_time(scenes) if len(scenes) > 1 else scenes[0])
```

Import `GeoBox` under `TYPE_CHECKING` (`from odc.geo.geobox import GeoBox`). If `gdal.read` already restores a scalar `time` coordinate from the file's tags, `expand_dims` must not add a duplicate. In that case, replace the `expand_dims` line with `scene = scene.expand_dims(TIME_COORDINATE)` when `TIME_COORDINATE in scene.coords and TIME_COORDINATE not in scene.dims`, and assert the file's own instant equals `when`. Smoke-check both branches with the Step 1 tests before choosing.

- [ ] **Step 4: Rewire `readers.py`**

- **`open_raster`.**
  - Type `source: str | PathLike[str] | Sequence[CogFile]`.
  - Replace the multi-source block with:

```python
    if not isinstance(source, (str, PathLike)):
        files = list(source)
        if not all(isinstance(file, CogFile) for file in files):
            raise TypeError(
                "open_raster reads one path, or the CogFile records to_cog "
                "returned; open other files one at a time"
            )
        return cogs.read(files, **options)
```

  - Delete the "folder of COGs" branch and `_combine_rasters`. Keep `.zarr`, NetCDF, GDAL and `.safe` single-path reading.
  - Docstring: "Open one raster file or store, or the records one `to_cog` wrote."
- **`open_stack`.**
  - Type `source: str | PathLike[str] | Sequence[CogFile]`.
  - Replace the `Mapping` branch with a `CogFile` sequence branch: group by `file.collection` in first-seen order, raise the `ValueError` above for `None`, and `stack({name: cogs.read(group, **options) for ...})`.
  - In the directory branch, accept only `.zarr`/NetCDF entries and `.safe` (drop `.tif` files and sub-folders as groups).
  - The docstring examples use `open_stack(sample.gs.to_cog(...))` and `open_stack("prepared/s1")` for a folder of stores.
- **Imports.** Import `from .raster import cogs` and `from .raster.cogs import CogFile`.

- [ ] **Step 5: Update old reader tests, file by file**

- **`tests/geodata/io/test_readers.py`:**
  - Delete the tests that pass lists of paths or folders of `.tif` to `open_raster`/`open_stack` (around lines 40–140 and 200–265, which `grep -n "open_raster(\[\|open_raster(paths\|open_stack(folder" tests/geodata/io/test_readers.py` lists).
  - Re-express any behaviour still in the spec through `CogFile` records: band merge per instant, chronological time, attrs merge with `DroppedAttrsWarning`.
- **`tests/geodata/core/test_stack.py`.** `open_stack(paths)` takes the `to_cog` records directly.
- **`tests/geodata/io/test_backend.py:68`.** If it opens a COG folder through the xarray backend, rewrite it to save with `to_zarr` and open that store; the backend keeps Zarr stacks.

- [ ] **Step 6: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, and ruff passes. Then run `uv run pytest tests/test_layering.py -q`. Expected: pass, since `io` imports `transform` only inside `read`.

---

### Task 3: `create_items` builds Items from `CogFile` records

**Files:**
- Modify: `src/geosave_engine/geodata/stac/_item.py`:
  - `create_items(files: Iterable[CogFile], *, id=None, collection=None, datetime=None, **options)`;
  - remove `_files`, `_group_items` and the timespan grouping in `_items`;
  - `create_stack_items(files, *, name)` becomes `create_items(files, id=name)` plus the `geosave:stack` stamp, kept until Task 12;
  - a new internal helper `cog_files(item: Mapping) -> list[CogFile]` for Task 6.
- Modify: `src/geosave_engine/geodata/stac/extensions/eo.py` and `raster.py`. The band order written follows the file's own band order, as today. No change is expected; verify.
- Test: `tests/geodata/stac/test_item.py`, plus the callers of `create_items`/`create_stack_items` in tests and src (`grep -rn "create_items(\|create_stack_items(" src tests`).

**Interfaces:**
- Consumes: `CogFile` (Task 1).
- Produces:
  - **`create_items(files, *, id=None, collection=None, datetime=None, **options) -> GeoDataFrame`.**
    - Files sharing `(collection, datetime)` form one Item, with Items in first-seen order.
    - The asset key is the band name when a file holds one band, and `"image"` when it holds several. Two files of one Item with the same key raise a `ValueError`.
    - `collection=` (a `pystac.Collection`) applies only when every file's `collection` is None; otherwise it raises `ValueError("these files already name their collections ...")`.
    - `id`: None names an Item after its file stem (one file) or after the files' shared folder (several). Otherwise it is a `GeoAnchor.format` template, or, when the files name collections, a prefix `f"{id}/"`.
    - Undated files take `datetime=`, else the span the dated files cover (widened by TimeSpec through `gs.timespan`), else they raise `pass datetime=`.
    - `.zarr`/`.nc` hrefs can't occur, because `CogFile`s are TIFFs.
  - **`create_stack_items(files: Iterable[CogFile], *, name: str) -> tuple[pystac.Item, ...]`.** Transitional; removed in Task 12.
  - **`cog_files(item: Mapping[str, Any]) -> list[CogFile]`.** For one Item dict, one record per data asset: `href` from the asset; `bands` from the asset's `eo:bands` names (Plan 1 writes them); `datetime` from `properties["datetime"]`, else `start_datetime`; `collection` from `item["collection"]`.

- [ ] **Step 1: Write the failing tests**

Replace the Plan 1 tests in `tests/geodata/stac/test_item.py` that build from path lists:
- `test_create_items_returns_an_item_table`
- `test_a_stack_mapping_makes_each_group_a_collection`
- `test_a_stack_mapping_needs_the_stack_name`
- `test_a_stack_of_datacube_stores_is_not_catalogued`
- `test_a_datacube_store_is_not_catalogued`

Use these:

```python
def test_records_become_one_item_per_scene(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}_{{band}}.tif")

    rows = stac.create_items(files)

    assert len(rows) == 2
    item = table.to_items(rows)[0]
    assert sorted(item.assets) == ["nir", "red"]
    assert [b.name for b in EOExtension.ext(item.assets["red"]).bands] == ["red"]


def test_a_multiband_file_is_one_image_asset_with_bands_in_order(scene, tmp_path) -> None:
    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}.tif")

    item = table.to_items(stac.create_items(files))[0]

    assert list(item.assets) == ["image"]
    names = [b.name for b in EOExtension.ext(item.assets["image"]).bands]
    assert names == ["red", "nir"]


def test_a_stacks_layers_become_collections(scene, label, tmp_path) -> None:
    files = stack({"optical": scene, "label": label}).gs.to_cog(
        {"optical": f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif",
         "label": f"{tmp_path}/{{collection}}.tif"}
    )

    rows = stac.create_items(files, id="s0")

    assert rows["collection"].tolist() == ["optical", "optical", "label"]
    assert rows["id"].tolist()[2] == "s0/label"
    assert str(rows.iloc[2]["start_datetime"])[:10] == "2025-06-01"


def test_a_collection_cannot_override_named_layers(scene, label, tmp_path) -> None:
    files = stack({"optical": scene}).gs.to_cog(f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif")

    with pytest.raises(ValueError, match="already name"):
        stac.create_items(files, collection=stac.create_collection("x", description="x"))


def test_records_of_an_item_round_trip(scene, tmp_path) -> None:
    from geosave_engine.geodata.stac._item import cog_files

    files = scene.gs.to_cog(f"{tmp_path}/forest_{{time:%Y%m%d}}_{{band}}.tif")
    item = table.to_items(stac.create_items(files))[0]

    back = cog_files(item.to_dict())

    assert sorted(f.bands for f in back) == [("nir",), ("red",)]
    assert {f.datetime for f in back} == {files[0].datetime}
```

`EOExtension` is already imported (Plan 1). `table` is `_table`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/stac/test_item.py -k "records or multiband or layers or override" -v`
Expected: FAIL. `create_items` iterates `CogFile`s as paths, and `cog_files` doesn't exist.

- [ ] **Step 3: Implement in `_item.py`**

Replace `create_items`, `_files`, `_group_items` and the scene-grouping part of `_items` with:

```python
def create_items(
    files: Iterable[CogFile],
    *,
    id: str | None = None,
    collection: pystac.Collection | None = None,
    datetime: ItemTime | None = None,
    **options: Any,
) -> GeoDataFrame:
    """Build the item table of the files one or more `to_cog` calls wrote.

    Files sharing a collection and an instant are one Item, one Asset each:
    the band's name for a single-band file, `"image"` for several bands.

    Args:
        files: Records `to_cog` returned.
        id: `GeoAnchor.format` template filled from each Item's raster, or,
            for files naming their collections, a prefix every id takes.
            None names each Item after its file, or the folder its files share.
        collection: Collection of files that name none.
        datetime: Time for undated files. None gives them the span the dated
            files cover.
        **options: Read options passed to `open_raster` for every file.

    Returns:
        Item table, one row per Item, in the order the files first name them.

    Raises:
        ValueError: No file is given; `collection` is given for files naming
            their own; two files of one Item share an asset key; ids repeat;
            or a file is undated, none is dated and `datetime` is None.

    Examples:
        >>> rows = create_items(sample.gs.to_cog("s0/{collection}_{time:%Y%m%d}.tif"), id="s0")
        >>> rows["collection"].tolist()
        ['optical', 'optical', 'label']
    """
    records = list(files)
    if not records:
        raise ValueError("create_items needs the CogFile records to_cog returned")
    named = any(file.collection is not None for file in records)
    if named and collection is not None:
        raise ValueError(
            "these files already name their collections, the layers they were "
            "written from; drop collection="
        )

    scenes: dict[tuple[str | None, pd.Timestamp | None], list[CogFile]] = {}
    for file in records:
        scenes.setdefault((file.collection, file.datetime), []).append(file)

    with ExitStack() as opened:
        rasters = {file.href: _open(opened, file.href, options)[1] for file in records}
        spans = [r.gs.timespan for r in rasters.values() if r.gs.timespan is not None]
        fallback = datetime
        if fallback is None and spans:
            fallback = (min(s for s, _ in spans), max(e for _, e in spans))

        items = []
        for (layer, when), members in scenes.items():
            assets: dict[str, tuple[str, xr.Dataset]] = {}
            for file in members:
                key = file.bands[0] if len(file.bands) == 1 else "image"
                if key in assets:
                    raise ValueError(
                        f"{file.href} and {assets[key][0]} would both be the asset "
                        f"{key!r} of one Item; split them with {{band}} or {{time}}"
                    )
                assets[key] = (file.href, rasters[file.href])
            first = PurePosixPath(members[0].href)
            stem = first.stem if len(members) == 1 else first.parent.name
            if id is None:
                item_id = stem
            elif named:
                item_id = f"{id}/{stem}"
            else:
                item_id = next(iter(assets.values()))[1].gs.anchor.format(id)
            items.append(
                _item(
                    assets,
                    id=item_id,
                    collection=layer if named else _identity(collection),
                    when=None if when is not None else fallback,
                )
            )
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"id {id!r} gives several Items the same id {ids}; add a time field "
            f"such as {{start:%Y%m%d}}"
        )
    return _table.from_items(items)


def cog_files(item: Mapping[str, Any]) -> list[CogFile]:
    """Spell the data assets of one Item dict as the records a write produced.

    Args:
        item: One STAC Item as a dict, its asset hrefs absolute.

    Returns:
        One record per data asset, bands named as the asset's `eo:bands`
        lists them, dated by the Item's `datetime` or `start_datetime`.
    """
    properties = item.get("properties", {})
    stamp = properties.get("datetime") or properties.get("start_datetime")
    when = None if stamp is None else pd.Timestamp(stamp).tz_convert(None)
    records = []
    for key, asset in item["assets"].items():
        roles = asset.get("roles")
        if roles is not None and "data" not in roles:
            continue
        names = [band["name"] for band in asset.get("eo:bands", [])] or [key]
        records.append(
            CogFile(
                href=asset["href"],
                bands=tuple(names),
                datetime=when,
                collection=item.get("collection"),
            )
        )
    return records
```

Rewrite `create_stack_items(files, *, name)` as:

```python
    rows = create_items(files, id=name, datetime=datetime, **options)
    items = _table.to_items(rows)
    for each in items:
        GeosaveExtension.ext(each, add_if_missing=True).stack = name
    return tuple(items)
```

Imports to add:
- `from collections.abc import Iterable`
- `from geosave_engine.geodata.io.raster.cogs import CogFile`

Imports to remove: whichever ones become unused; ruff reports them.

`create_item(assets: Mapping[str, path], *, id, ...)` stays. Its keys name the assets explicitly, as `workflow/tasks/labels.py` needs.

Because `_items` loses its timespan grouping, delete it if nothing else calls it after this change. Keep `_item`, `_open`, `_identity`, `_covered`, `default_key` and `_media_type`. Check with `grep -n "_items(" src/geosave_engine/geodata/stac/_item.py`.

- [ ] **Step 4: Move callers, file by file**

List them: `grep -rn "create_items(\|create_stack_items(" src tests --include=*.py`.
- **Calls passing hrefs or `{group: paths}` built in Task 1.** Pass the `to_cog` records directly.
- **`create_stack_items` callers.** Pass the stack's `to_cog` records and `name=`.
- **`ModelSpec.prepare`.** It still writes Zarr with `create_stack_items` until Task 10, so `create_stack_items` would break there. Handle this by keeping prepare's Zarr branch working through a temporary `_store_items(paths: Mapping[str, str], name)` in `_item.py` that wraps the old store logic:

```python
def _store_items(
    paths: Mapping[str, str | PathLike[str]], *, name: str, datetime: ItemTime | None
) -> tuple[pystac.Item, ...]:
    """Transitional: Items of a stack saved to Zarr stores, removed in Task 12."""
    with ExitStack() as opened:
        groups = {group: dict([_open(opened, path, {})]) for group, path in paths.items()}
        spans = [r.gs.timespan for f in groups.values() for r in f.values() if r.gs.timespan]
        fallback = datetime or (
            (min(s for s, _ in spans), max(e for _, e in spans)) if spans else None
        )
        items = []
        for group, files in groups.items():
            ((href, raster),) = files.items()
            item = _item(
                {default_key(raster): (href, raster)},
                id=f"{name}/{group}",
                collection=group,
                when=None if raster.gs.timespan is not None else fallback,
            )
            GeosaveExtension.ext(item, add_if_missing=True).stack = name
            items.append(item)
        return tuple(items)
```

  `prepare` calls `_store_items(stores, name=name, datetime=when)` in place of `create_stack_items(stores, ...)`. Ledger this as transitional.

- [ ] **Step 5: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, and ruff passes.

---

### Task 4: `StacSource` drops `StacMetadata`; `search(anchor)` and `collection`

**Files:**
- Delete: `src/geosave_engine/geodata/attrs/models/stac.py`. Remove `StacMetadata` and `StacItem` from `attrs/models/__init__.py` (including the `"stac_metadata": StacMetadata` registry entry) and from `attrs/__init__.py`.
- Modify: `src/geosave_engine/geodata/attrs/headers/stac.py` (`create_header`):
  - drop the `item_properties`, `asset_fields` and `groupby` parameters and the `stac_items`/`stac_groupby` root keys;
  - delete `_item_assets`/`_selected` if they become unused.
- Modify: `src/geosave_engine/geodata/stac/source.py`:
  - `StacSourceConfig` loses `item_properties` and `asset_fields`;
  - `to_load_kwargs` stops excluding them;
  - add `StacSource.search(anchor) -> GeoDataFrame` and a `StacSource.collection` property;
  - `load` uses the header without provenance;
  - the docstrings of `list_item_properties`/`list_asset_fields` point at `with_properties` and `search(anchor)["assets"]`.
- Modify: `src/geosave_engine/model/spec/stac.py`, if its load serializer mentions the removed fields.
- Test:
  - `tests/geodata/stac/test_source.py`
  - `tests/geodata/attrs/headers/test_stac.py`
  - `tests/conftest.py` (fakes passing `item_properties`)
  - any test asserting `StacMetadata` (`grep -rn "StacMetadata\|stac_items\|stac_groupby\|item_properties\|asset_fields" tests src`)

**Interfaces:**
- Produces:
  - `StacSource.search(anchor: GeoAnchor) -> GeoDataFrame`: the item table of exactly the Items `load(anchor)` reads.
  - `StacSource.collection -> pystac.Collection`: from the client, cached by the client.
  - `StacSourceConfig` fields: `bands`, `groupby`, `chunks`, `resampling`, `dtype`, `nodata`, `fail_on_error`, `with_properties`, `stac_cfg`, `pool`, `progress`, `patch_url`, `preserve_original_order`.

- [ ] **Step 1: Write the failing test**

Add to `tests/geodata/stac/test_source.py`, using the local fixture that `test_dataflow.py` already uses (`local_source`, defined in `tests/geodata/stac/conftest.py` or `tests/conftest.py`; find it with `grep -rn "def local_source" tests`):

```python
def test_a_source_searches_the_items_it_loads_and_keeps_no_copy(local_source) -> None:
    source, anchor = local_source

    rows = source.search(anchor)
    loaded = source.load(anchor)

    assert type(rows.gs).__name__ == "GeoCatalog"
    assert source.collection.id == rows["collection"].iloc[0]
    assert "stac_items" not in loaded.attrs
    assert "stac_groupby" not in loaded.attrs
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/geodata/stac/test_source.py::test_a_source_searches_the_items_it_loads_and_keeps_no_copy -v`
Expected: FAIL with `AttributeError: 'StacSource' object has no attribute 'search'`.

- [ ] **Step 3: Implement**

In `source.py`:

```python
    @property
    def collection(self) -> pystac.Collection:
        """Return the Collection this source loads, as its catalog publishes it."""
        return self.client.collection(self._collection)

    def search(self, anchor: GeoAnchor) -> GeoDataFrame:
        """List the Items `load(anchor)` reads, as an item table.

        Args:
            anchor: Grid and datetime window to search.

        Returns:
            One row per matching Item, read through `gs` as a catalog.
        """
        from . import _table

        found = self.client.search_items(self._search_query(anchor))
        return _table.from_items(found) if found else _table.empty()
```

The class stores the collection id under some attribute name. Use the existing attribute; check `__init__`. If it's `self.collection` today, rename it to `self._collection` throughout the class, because the property now takes the public name.

`load` keeps its search and odc call, then builds the header with `create_header(matched, self.collection, data, stac_cfg=self.config.stac_cfg)`.

In `headers/stac.py`: delete the `rows = [...]` block and the `"stac_groupby"`/`"stac_items"` root keys, and remove the now-unused parameters and helpers.

In `StacSourceConfig`: delete the two fields and their Args lines. In `to_load_kwargs`, use `self.model_dump(exclude_none=True)`.

- [ ] **Step 4: Remove the attrs model and update the tests**

Delete `attrs/models/stac.py` and its exports. Then, file by file, remove assertions about `StacMetadata`, `stac_items`, `item_properties` or `asset_fields`. Where a test checks a per-scene property, rewrite it through `with_properties` coordinates, as `test_dataflow.py` already does.

- [ ] **Step 5: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, and ruff passes.

---

## Phase B: cuts windows and consumers (only named tests pass until Task 12)

Shared fixture for Tasks 5–9. Replace `tests/geodata/cuts/conftest.py`'s `items` and `opened` fixtures with the following. Keep `grid`, `_series` and `sample`.

```python
@pytest.fixture
def items(sample, tmp_path):
    """The sample written as COGs and catalogued, read back from its table."""
    files = sample.gs.to_cog(stack_templates(sample, tmp_path / "s0"))
    rows = stac.create_items(files, id="s0")
    return open_vector(rows.gs.to_geoparquet(tmp_path / "items.parquet"))
```

Imports: `from geosave_engine.geodata import open_vector, raster, stac, stack` and `from tests.geodata.conftest import stack_templates`.

`stack_templates` returns `{collection: template}`, so it keeps working once Task 1 renames the field. If it spells `{group}`, change it to `{collection}`.

### Task 5: `cuts.stacks(items, on=)` and `GeoSample.to_stack`

**Files:**
- Rewrite: `src/geosave_engine/geodata/cuts/stacks.py`.
- Create: `src/geosave_engine/geodata/cuts/sample.py` (the `GeoSample` Series accessor).
- Modify: `src/geosave_engine/geodata/cuts/__init__.py`. Import `sample` for its registration side effect; the exports are `chips`, `frames`, `layout`, `merge`, `stacks`. `select*` stay exported until Task 12.
- Modify: `src/geosave_engine/geodata/warnings.py`. Add `DuplicateInstantsWarning` and `PartialCoverWarning`, as the existing warning classes there are declared.
- Test: rewrite `tests/geodata/cuts/test_stacks.py`; create `tests/geodata/cuts/test_sample.py`.

**Interfaces:**
- Consumes: `cog_files` (Task 3), `cogs.read` (Task 2), `transform.crop.crop`, `stac._table.to_items`.
- Produces: `stacks(items: gpd.GeoDataFrame, *, on: str) -> GeoDataFrame`, one row per item of collection `on`, in table order, with a `RangeIndex` and these columns:

```text
anchor           str          the anchor item's id
parent           None
sources          str          orjson of {collection: [stripped Item dict, ...]}
proj:code        str          anchor CRS, as "EPSG:32633" or WKT2
proj:transform   list[float]  6 affine terms of the anchor grid
proj:shape       list[int]    [height, width]
start_datetime   Timestamp | None   span of the dated members (naive UTC)
end_datetime     Timestamp | None
geometry         Polygon      anchor footprint, EPSG:4326
<anchor props>   ...          the anchor Item's own properties, minus STAC fields
```

  The STAC fields excluded from `<anchor props>` are `type`, `stac_version`, `stac_extensions`, `id`, `collection`, `assets`, `links`, `bbox`, `geometry`, `datetime`, `start_datetime` and `end_datetime`, plus `proj:*`.

  A stripped Item dict keeps `type`, `stac_version`, `id`, `collection`, `geometry`, `bbox`, `properties` (only `datetime`, `start_datetime`, `end_datetime`) and `assets` (whole; hrefs absolute).

  Membership: an item of another collection joins when its footprint covers the anchor's footprint shrunk by `1e-7` degrees (to absorb float noise in reprojected footprints). Anchor items never join each other's windows.

  Warnings:
  - `DuplicateInstantsWarning`: two items of one collection join at the same instant. The first in table order is kept.
  - `PartialCoverWarning`: an item intersects an anchor without covering it. One warning per call, naming the count and a few ids.

  Raises `ValueError` when `on` names no collection in the table, listing the collections present.
- Produces: `GeoSample` Series accessor `gs` on any row with `sources`:
  - `to_stack(*, resampling: Mapping[str, str] | None = None) -> DataTree`;
  - other Series raise `AttributeError("gs reads a row of a cuts table ...")`.

- [ ] **Step 1: Write the failing tests**

`tests/geodata/cuts/test_stacks.py` (replace its contents):

```python
"""An item table lists one window per anchor item, joined by what covers it."""

from __future__ import annotations

import orjson
import pytest

from geosave_engine.geodata import cuts
from geosave_engine.geodata.warnings import DuplicateInstantsWarning


def test_each_anchor_item_is_one_window_on_its_grid(items, grid) -> None:
    (window,) = cuts.stacks(items, on="label").to_dict("records")

    assert window["anchor"] == "s0/label"
    assert window["parent"] is None
    assert window["proj:shape"] == [grid.shape[0], grid.shape[1]]
    assert window["proj:transform"] == pytest.approx(list(grid.transform)[:6])
    sources = orjson.loads(window["sources"])
    assert sorted(sources) == ["dem", "label", "s1", "s2"]
    assert len(sources["s2"]) == 5


def test_sources_keep_absolute_hrefs_and_only_what_loading_needs(items) -> None:
    sources = orjson.loads(cuts.stacks(items, on="label").iloc[0]["sources"])

    s2 = sources["s2"][0]
    assert set(s2) == {"type", "stac_version", "id", "collection", "geometry", "bbox", "properties", "assets"}
    assert set(s2["properties"]) <= {"datetime", "start_datetime", "end_datetime"}
    assert all(asset["href"].startswith("/") for asset in s2["assets"].values())


def test_an_unknown_anchor_collection_refuses(items) -> None:
    with pytest.raises(ValueError, match="label"):
        cuts.stacks(items, on="labels")


def test_a_repeated_instant_keeps_the_first_and_warns(items) -> None:
    import geopandas as gpd

    doubled = gpd.GeoDataFrame(
        __import__("pandas").concat([items, items[items["collection"] == "s1"].head(1)]),
        geometry="geometry",
        crs=items.crs,
    )
    doubled.iloc[-1, doubled.columns.get_loc("id")] = "s0/s1_copy"

    with pytest.warns(DuplicateInstantsWarning):
        window = cuts.stacks(doubled, on="label").iloc[0]

    assert len(orjson.loads(window["sources"])["s1"]) == 3


def test_an_item_covering_only_part_of_the_anchor_does_not_join(items) -> None:
    from shapely.affinity import translate

    from geosave_engine.geodata.warnings import PartialCoverWarning

    shifted = items.copy()
    s1 = shifted["collection"] == "s1"
    shifted.loc[s1, "geometry"] = shifted.loc[s1, "geometry"].apply(
        lambda geom: translate(geom, xoff=0.001)
    )

    with pytest.warns(PartialCoverWarning):
        window = cuts.stacks(shifted, on="label").iloc[0]

    assert "s1" not in orjson.loads(window["sources"])


def test_anchor_properties_ride_on_the_window(items) -> None:
    labelled = items.copy()
    labelled["annotator"] = None
    labelled.loc[labelled["collection"] == "label", "annotator"] = "ana"

    window = cuts.stacks(labelled, on="label").iloc[0]

    assert window["annotator"] == "ana"
```

`tests/geodata/cuts/test_sample.py`:

```python
"""A window row reads itself as an aligned lazy stack."""

from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import pytest

from geosave_engine.geodata import cuts


def test_a_window_reads_every_collection_on_its_grid(items, sample, grid) -> None:
    window = cuts.stacks(items, on="label").iloc[0]

    tree = window.gs.to_stack()

    assert sorted(tree.gs.groups) == ["dem", "label", "s1", "s2"]
    for raster in tree.gs.rasters.values():
        assert raster.gs.geobox == grid
    assert dask.is_dask_collection(tree["s2"].dataset.red)
    np.testing.assert_array_equal(
        tree["label"].dataset.label.values, sample["label"].dataset.label.values
    )


def test_a_row_without_sources_has_no_sample_accessor() -> None:
    with pytest.raises(AttributeError, match="cuts table"):
        pd.Series({"a": 1}).gs
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/cuts/test_stacks.py tests/geodata/cuts/test_sample.py -v`
Expected: FAIL. `stacks()` doesn't accept `on=`, and pandas Series has no `gs`.

- [ ] **Step 3: Implement `stacks.py`**

```python
"""One window per anchor item, joined by the items that cover it."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import geopandas as gpd
import orjson
import pandas as pd
import pystac
from affine import Affine
from odc.geo.geobox import GeoBox
from pystac.extensions.projection import ProjectionExtension

from geosave_engine import __path__ as _package_paths
from geosave_engine.geodata.stac import _table
from geosave_engine.geodata.warnings import DuplicateInstantsWarning, PartialCoverWarning

if TYPE_CHECKING:
    from geosave_engine.geodata import GeoDataFrame

# Footprints are WGS84 polygons rebuilt from each grid; a shrink this small
# absorbs float noise without letting a real gap pass.
_TOLERANCE = 1e-7

_STAC_FIELDS = {
    "type", "stac_version", "stac_extensions", "id", "collection", "assets",
    "links", "bbox", "geometry", "datetime", "start_datetime", "end_datetime",
}
_TIME_FIELDS = ("datetime", "start_datetime", "end_datetime")


def stacks(items: gpd.GeoDataFrame, *, on: str) -> GeoDataFrame:
    """List one window per item of collection `on`, on that item's grid.

    Every item of another collection whose footprint covers the anchor's
    joins it, so a window holds what was observed over all of it.

    Args:
        items: Item table, local or searched, of any collections and dates.
        on: Collection whose items are the windows, as labels are.

    Returns:
        One row per anchor item, in table order; see the module for columns.

    Raises:
        ValueError: `on` names no collection in the table, or an anchor's
            asset states no grid.

    Warns:
        DuplicateInstantsWarning: Two items of one collection cover an anchor
            at one instant; the first is kept.
        PartialCoverWarning: Items intersect an anchor without covering it.

    Examples:
        >>> windows = stacks(items, on="label")
        >>> windows.iloc[0].gs.to_stack().gs.groups
        ('s2', 's1', 'dem', 'label')
    """
    present = sorted(items["collection"].dropna().unique())
    if on not in present:
        raise ValueError(
            f"no item belongs to the collection {on!r}; the table holds {present}"
        )
    anchors = items[items["collection"] == on]
    others = items[items["collection"] != on]
    shrunk = gpd.GeoDataFrame(
        {"anchor_row": range(len(anchors))},
        geometry=anchors.geometry.buffer(-_TOLERANCE).values,
        crs=items.crs,
    )
    covering = gpd.sjoin(others, shrunk, predicate="covers", how="inner")
    touching = gpd.sjoin(others, shrunk, predicate="intersects", how="inner")
    partial = touching.index.difference(covering.index)
    if len(partial):
        warnings.warn(
            f"{len(partial)} items intersect a window without covering it, so they "
            f"do not join it: {others.loc[partial[:5], 'id'].tolist()}",
            PartialCoverWarning,
            skip_file_prefixes=tuple(_package_paths),
        )

    rows = []
    for position, (_, anchor) in enumerate(anchors.iterrows()):
        (item,) = _table.to_items(anchors.iloc[[position]])
        members = {on: [_stripped(item)]}
        joined = others.loc[covering.index[covering["anchor_row"] == position]]
        for each in _table.to_items(joined) if len(joined) else []:
            kept = members.setdefault(cast("str", each.collection_id), [])
            when = _instant(each)
            if when is not None and any(_instant_of(d) == when for d in kept):
                warnings.warn(
                    f"{each.collection_id!r} has two items at {when} over "
                    f"{item.id!r}; kept {kept[0]['id']!r}, dropped {each.id!r}",
                    DuplicateInstantsWarning,
                    skip_file_prefixes=tuple(_package_paths),
                )
                continue
            kept.append(_stripped(each))
        grid = _grid(item)
        dated = [t for ds in members.values() for d in ds if (t := _instant_of(d))]
        rows.append(
            {
                "anchor": item.id,
                "parent": None,
                "sources": orjson.dumps(members).decode(),
                "proj:code": str(grid.crs),
                "proj:transform": list(grid.transform)[:6],
                "proj:shape": [grid.shape[0], grid.shape[1]],
                "start_datetime": min(dated) if dated else None,
                "end_datetime": max(dated) if dated else None,
                "geometry": anchor.geometry,
                **{k: v for k, v in anchor.items() if k not in _STAC_FIELDS and not str(k).startswith("proj:")},
            }
        )
    return cast("GeoDataFrame", gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326"))


def _stripped(item: pystac.Item) -> dict[str, Any]:
    """Keep what reading an Item needs, its hrefs absolute."""
    fields = item.to_dict(include_self_link=False)
    return {
        "type": fields["type"],
        "stac_version": fields["stac_version"],
        "id": fields["id"],
        "collection": fields.get("collection"),
        "geometry": fields["geometry"],
        "bbox": fields.get("bbox"),
        "properties": {k: v for k, v in fields["properties"].items() if k in _TIME_FIELDS and v is not None},
        "assets": fields["assets"],
    }


def _instant(item: pystac.Item) -> pd.Timestamp | None:
    """Return an Item's instant: `datetime`, else `start_datetime`."""
    stamp = item.datetime or item.common_metadata.start_datetime
    return None if stamp is None else pd.Timestamp(stamp).tz_convert(None)


def _instant_of(item: dict[str, Any]) -> pd.Timestamp | None:
    properties = item["properties"]
    stamp = properties.get("datetime") or properties.get("start_datetime")
    return None if stamp is None else pd.Timestamp(stamp).tz_convert(None)


def _grid(item: pystac.Item) -> GeoBox:
    """Return the grid the anchor's first data asset states."""
    for asset in item.assets.values():
        stated = ProjectionExtension.ext(asset)
        if stated.shape is not None and stated.transform is not None:
            crs = stated.code if stated.code is not None else stated.wkt2
            return GeoBox(tuple(stated.shape), Affine(*stated.transform[:6]), crs)
    raise ValueError(f"{item.id} states no grid to cut windows on")
```

Remove `_instant` if `_instant_of(_stripped(each))` serves; keep a single helper. Hrefs are absolute because `open_vector` and `search` resolve them (Plan 1).

- [ ] **Step 4: Implement `sample.py`**

```python
"""The `gs` accessor of one cuts row: read the window it names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import orjson
import pandas as pd
from affine import Affine
from odc.geo.geobox import GeoBox

if TYPE_CHECKING:
    from geosave_engine.geodata import DataTree


@pd.api.extensions.register_series_accessor("gs")
class GeoSample:
    """Read one row of `cuts.stacks`, `frames` or `chips` as a stack.

    Args:
        row: One row of a cuts table.

    Raises:
        AttributeError: The row states no `sources`, so it is no cuts row.

    Examples:
        >>> windows.iloc[0].gs.to_stack().gs.groups
        ('s2', 'label')
    """

    def __init__(self, row: pd.Series) -> None:
        """Bind the accessor to one row."""
        # pandas reads AttributeError as "this object has no such accessor".
        if "sources" not in row.index:
            raise AttributeError(
                "gs reads a row of a cuts table, which states its sources; "
                "cut one with cuts.stacks"
            )
        self._row = row

    def to_stack(self, *, resampling: Mapping[str, str] | None = None) -> DataTree:
        """Read every collection of this window onto its grid, lazily.

        Args:
            resampling: Kernel per collection where its files are reprojected.
                A collection not named takes `"nearest"`, safe for classes.

        Returns:
            Stack with one group per collection, every group on this row's
            grid; pixels past the window it was cut from are filled as the
            row's `mode` states.
        """
        from geosave_engine.geodata.core.stack import stack
        from geosave_engine.geodata.io.raster import cogs
        from geosave_engine.geodata.stac._item import cog_files
        from geosave_engine.geodata.transform.crop import crop

        row = self._row
        height, width = (int(edge) for edge in row["proj:shape"])
        grid = GeoBox((height, width), Affine(*row["proj:transform"]), row["proj:code"])
        pad = row.get("pad")
        pad = [[0, 0], [0, 0]] if pad is None else [[int(a), int(b)] for a, b in pad]
        (top, bottom), (left, right) = pad
        inner = grid[top : height - bottom, left : width - right]

        kernels = dict(resampling or {})
        groups = {}
        for name, members in orjson.loads(row["sources"]).items():
            files = [file for item in members for file in cog_files(item)]
            groups[name] = cogs.read(
                files, geobox=inner, resampling=kernels.get(name, "nearest")
            )
        tree = stack(groups)
        if pad == [[0, 0], [0, 0]]:
            return tree
        return cast(
            "DataTree",
            crop(
                tree,
                (-top, -left),
                (height, width),
                padding=[tuple(pad[0]), tuple(pad[1])],
                mode=row.get("mode") or "constant",
            ),
        )
```

In `cuts/__init__.py`, add `from . import sample as _sample  # registers the row accessor`, and update the module docstring to the new flow: `stacks` → `frames` → `chips`, a row reads itself with `row.gs.to_stack()`, and `merge` puts chip outputs back.

- [ ] **Step 5: Run the named tests**

Run: `uv run pytest tests/geodata/cuts/test_stacks.py tests/geodata/cuts/test_sample.py -v`
Expected: all pass. Ledger: "Phase B: full suite intentionally red until Task 12 (old cuts consumers)."

---

### Task 6: `cuts.frames` over window members

**Files:**
- Modify: `src/geosave_engine/geodata/cuts/frames.py`. Keep `_spans` and `_joint_axis` unchanged; rewrite the per-window loop over `sources`.
- Test: `tests/geodata/cuts/test_frames.py`, `tests/geodata/cuts/test_frames_labels.py`. Keep every `_joint_axis`/`_spans` unit test; rewrite the tests that build windows from `times` dicts to go through `cuts.stacks(items, on="label")`.

**Interfaces:**
- Consumes: `stacks` rows (Task 5).
- Produces: `frames(windows, length, *, tolerance, stride=None, mode="strict") -> GeoDataFrame`.
  - One row per frame, windows in order, with a fresh `RangeIndex`.
  - `parent` is the input row's position.
  - `sources` keeps, per collection, only the items at the frame's instants. A collection with exactly one item (static) keeps it in every frame.
  - `start_datetime`/`end_datetime` give the frame's own span.
  - Every other column is copied.

- [ ] **Step 1: Write the failing test**

Add to `tests/geodata/cuts/test_frames.py`:

```python
def test_frames_keep_only_the_items_of_their_instants(items) -> None:
    import orjson

    windows = cuts.stacks(items, on="label")

    cut = cuts.frames(windows, 2, tolerance="10D")

    assert cut["parent"].tolist() == [0, 0]
    first = orjson.loads(cut.iloc[0]["sources"])
    assert [d["properties"]["datetime"][:10] for d in first["s2"]] == ["2024-03-08", "2024-03-28"]
    assert len(first["label"]) == 1 and len(first["dem"]) == 1
    assert str(cut.iloc[0]["start_datetime"])[:10] == "2024-03-05"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/geodata/cuts/test_frames.py::test_frames_keep_only_the_items_of_their_instants -v`
Expected: FAIL with `KeyError: 'times'`.

- [ ] **Step 3: Implement**

In `frames`, replace the per-window body that reads `window["times"]` with:

```python
    cut = []
    skips = 0
    for position, (_, window) in enumerate(windows.iterrows()):
        members = orjson.loads(window["sources"])
        dated = {
            name: sorted(items, key=_instant_of)
            for name, items in members.items()
            if len(items) > 1
        }
        if not dated:
            raise ValueError(
                f"no collection of window {position} holds more than one item, so "
                f"it holds no sequence to cut into frames"
            )
        series = {
            name: pd.DatetimeIndex([_instant_of(item) for item in items])
            for name, items in dated.items()
        }
        table = _joint_axis(series, reach)
        complete = (table.to_numpy() != -1).all(axis=1)
        for span in _spans(table.index.to_numpy(), length, stride):
            rows = table.iloc[span]
            if not complete[span].all():
                ...  # unchanged strict/drop handling
            kept = {
                name: (
                    [dated[name][i] for i in dict.fromkeys(rows[name].to_numpy())]
                    if name in dated
                    else items
                )
                for name, items in members.items()
            }
            stamps = [_instant_of(item) for name in dated for item in kept[name]]
            cut.append(
                {
                    **window.to_dict(),
                    "parent": position,
                    "sources": orjson.dumps(kept).decode(),
                    "start_datetime": min(stamps),
                    "end_datetime": max(stamps),
                }
            )
```

The `...` stands for the existing strict-raise and drop-skip block, kept exactly as it is. Import `_instant_of` from `.stacks`, and `orjson`. Change the final `GeoDataFrame(...)` to `.reset_index(drop=True)` so the index is fresh.

Delete every reference to `window["id"]`; messages say `window {position}`. Update the docstring example to use `orjson.loads(cut.iloc[0]["sources"])["s2"]`.

- [ ] **Step 4: Run the named tests**

Run: `uv run pytest tests/geodata/cuts/test_frames.py tests/geodata/cuts/test_frames_labels.py -v`
Expected: all pass, after rewriting the window-construction tests as described in Files.

---

### Task 7: `cuts.chips` with integer parents and per-chip `pad`; windows survive GeoParquet

**Files:**
- Modify: `src/geosave_engine/geodata/cuts/chips.py`.
- Test: `tests/geodata/cuts/test_chips.py`, plus a persistence test in `tests/geodata/cuts/test_sample.py`.

**Interfaces:**
- Consumes: `stacks`/`frames` rows.
- Produces: `chips(windows, size, *, overlap=0, mode="reflect") -> GeoDataFrame`.
  - One row per chip, with a fresh `RangeIndex`.
  - `parent` is the input row's position, and `chip` is the tile id in the layout.
  - The chip's own grid: `proj:transform` and `proj:shape` (halo included), plus `geometry`.
  - `halo`, the layout padding around the parent, kept for `merge`.
  - `overlap`.
  - `mode`.
  - `pad`: `[[top, bottom], [left, right]]`, the pixels of this chip past the parent's edge.
  - `sources` and the other columns are copied unchanged.
  - `layout` is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/geodata/cuts/test_chips.py`:

```python
def test_chips_state_their_grid_parent_and_edge_padding(items, grid) -> None:
    windows = cuts.stacks(items, on="label")

    cut = cuts.chips(windows, 32, overlap=8)

    assert set(cut["parent"]) == {0}
    assert cut.index.tolist() == list(range(len(cut)))
    first = cut.iloc[0]
    assert first["proj:shape"] == [32, 32]
    assert first["pad"][0][0] > 0 and first["pad"][1][0] > 0
    last = cut.iloc[-1]
    assert last["pad"][0][1] > 0 and last["pad"][1][1] > 0
```

Add to `tests/geodata/cuts/test_sample.py`:

```python
def test_a_reflect_halo_fills_only_past_the_parent(items, sample) -> None:
    chip = cuts.chips(cuts.stacks(items, on="label"), 32, overlap=8).iloc[0]

    label = chip.gs.to_stack()["label"].dataset.label.values

    (top, _), (left, _) = chip["pad"]
    source = sample["label"].dataset.label.values
    np.testing.assert_array_equal(label[top:, left:], source[: 32 - top, : 32 - left])
    np.testing.assert_array_equal(label[top - 1, left:], source[1, : 32 - left])


def test_a_saved_windows_table_reads_the_same_pixels(items, tmp_path) -> None:
    from geosave_engine.geodata import open_vector

    cut = cuts.chips(cuts.stacks(items, on="label"), 32)
    path = cut.gs.to_geoparquet(tmp_path / "windows.parquet")

    back = open_vector(path)

    a = cut.iloc[3].gs.to_stack()["s2"].dataset.red.values
    b = back.iloc[3].gs.to_stack()["s2"].dataset.red.values
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/geodata/cuts/test_chips.py::test_chips_state_their_grid_parent_and_edge_padding tests/geodata/cuts/test_sample.py -k "halo or saved" -v`
Expected: FAIL, because `chips` reads `window["height"]` and `window["id"]`.

- [ ] **Step 3: Implement**

In `chips`, read the parent grid from `proj:*`:

```python
    for position, (_, window) in enumerate(windows.iterrows()):
        height, width = (int(edge) for edge in window["proj:shape"])
        grid = GeoBox((height, width), Affine(*window["proj:transform"]), window["proj:code"])
        tiler, halo = layout((height, width), side, overlap=overlap, mode=mode)
        for number in range(len(tiler)):
            near, far = tiler.get_tile_bbox(number)
            row = int(near[0]) - halo[0][0]
            column = int(near[1]) - halo[1][0]
            h, w = (int(edge) for edge in far - near)
            chip = grid.translate_pix(column, row).crop((h, w))
            pad = [
                [max(0, -row), max(0, row + h - height)],
                [max(0, -column), max(0, column + w - width)],
            ]
            cut.append(
                {
                    **window.to_dict(),
                    "parent": position,
                    "chip": number,
                    "proj:transform": list(chip.transform)[:6],
                    "proj:shape": [h, w],
                    "halo": [list(widths) for widths in halo],
                    "overlap": overlap,
                    "mode": mode,
                    "pad": pad,
                    "geometry": chip.extent.to_crs("EPSG:4326").geom,
                }
            )
```

Delete the `id`, `row_off`, `col_off`, `height`, `width` and unreferenced-pixel branches. The columns list becomes `[*windows.columns, "chip", "halo", "overlap", "mode", "pad"]`, deduplicated. End with `.reset_index(drop=True)`.

`translate_pix(column, row).crop((h, w))` must give a chip that starts before the grid when `row`/`column` are negative. Verify it in the halo test; if odc-geo clips there, build it as `GeoBox((h, w), grid.transform * Affine.translation(column, row), grid.crs)`.

`pad` and `halo` are nested lists. Before they reach GeoParquet, check that `open_vector` returns them as nested lists or arrays, which `GeoSample` already normalizes with `int(a), int(b)`. If pyarrow returns them as numpy arrays of arrays, the `[[int(a), int(b)] for a, b in pad]` normalization covers it.

- [ ] **Step 4: Run the named tests**

Run: `uv run pytest tests/geodata/cuts/test_chips.py tests/geodata/cuts/test_sample.py -v`
Expected: all pass. Rewrite or delete old chips tests that pass hand-built windows with `height`/`width`/`id`; any such case must go through `cuts.stacks(items, on="label")`.

---

### Task 8: `cuts.merge` from saved chip COGs

**Files:**
- Modify: `src/geosave_engine/geodata/cuts/merge.py`.
- Test: rewrite `tests/geodata/cuts/test_merge.py`.

**Interfaces:**
- Consumes: `chips` rows (Task 7), `CogFile` and `open_raster` (Tasks 1–2), `create_items` (Task 3).
- Produces: `merge(parents, chips, predictions, *, taper=None) -> dict[int, Dataset]`.
  - `predictions` is an item table whose rows carry a `chip` column (the chip row's position) and one data asset each: a multi-band COG of one prediction, `C` bands.
  - The result maps each parent position that has chips to a Dataset on that parent's grid, holding one variable named after the prediction's bands' common prefix, or `"output"`, with dims `("band", "y", "x")`.
  - The merge is overlap-add with `tiler.Merger`, rebuilt from `layout(parent shape, chip shape, overlap, mode)`.
  - Raises `ValueError` when a parent's chip has no prediction.

- [ ] **Step 1: Write the failing test**

`tests/geodata/cuts/test_merge.py`:

```python
"""Chip outputs saved as COGs merge back onto the grid they were cut from."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import cuts, raster, stac


def _predict(chips, tmp_path, *, drop=()):
    files, positions = [], []
    for position, (_, chip) in enumerate(chips.iterrows()):
        if position in drop:
            continue
        h, w = chip["proj:shape"]
        grid = GeoBox((h, w), Affine(*chip["proj:transform"]), chip["proj:code"])
        # A prediction equal to the pixel's own x coordinate merges back exactly.
        xs = np.tile(np.asarray(grid.coords["x"].values, "float32"), (h, 1))
        out = raster({"logits_0": (("y", "x"), xs), "logits_1": (("y", "x"), -xs)}, grid)
        (file,) = out.gs.to_cog(f"{tmp_path}/chip-{position}.tif")
        files.append(file)
        positions.append(position)
    rows = stac.create_items(files, collection=stac.create_collection("chips", description="c"))
    rows["chip"] = positions
    return rows


@pytest.mark.parametrize("taper", [None, "hann"])
def test_saved_chips_merge_onto_the_parent_grid(items, grid, tmp_path, taper) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32, overlap=8)

    merged = cuts.merge(windows, chips, _predict(chips, tmp_path), taper=taper)

    (result,) = merged.values()
    assert result.gs.geobox == grid
    values = result["logits"].values
    expected = np.tile(np.asarray(grid.coords["x"].values, "float32"), (grid.shape[0], 1))
    np.testing.assert_allclose(values[0], expected, rtol=1e-5)
    np.testing.assert_allclose(values[1], -expected, rtol=1e-5)


def test_a_missing_chip_refuses(items, tmp_path) -> None:
    windows = cuts.stacks(items, on="label")
    chips = cuts.chips(windows, 32)

    with pytest.raises(ValueError, match="no output"):
        cuts.merge(windows, chips, _predict(chips, tmp_path, drop={0}))
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/geodata/cuts/test_merge.py -v`
Expected: FAIL, because `merge` expects an `outputs` DataArray.

- [ ] **Step 3: Implement**

```python
def merge(
    parents: gpd.GeoDataFrame,
    chips: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
    *,
    taper: str | None = None,
) -> dict[int, Dataset]:
    """Merge saved chip outputs onto the grid of each window they were cut from.

    The layout a cut used is rebuilt from the parent's shape and the chips'
    settings, so chips may have been predicted in any order and by several
    writers.

    Args:
        parents: The windows the chips were cut from, as given to `chips`.
        chips: The chips, as `chips` returned them.
        predictions: Item table of the saved outputs, one per chip, with a
            `chip` column holding the chip's row position.
        taper: How overlapping chips are weighed, as a window name such as
            `"hann"`. None weighs every chip alike.

    Returns:
        {
            <parent position>: Dataset on the parent's grid, holding the
                merged output as `(band, y, x)` under the bands' shared name,
        }

    Raises:
        ValueError: A parent's chip has no saved output.
    """
    from geosave_engine.geodata.io.readers import open_raster
    from geosave_engine.geodata.stac._item import cog_files
    from geosave_engine.geodata.stac._table import to_items

    by_chip = {
        int(position): cog_files(item.to_dict())
        for position, item in zip(predictions["chip"], to_items(predictions), strict=True)
    }
    merged: dict[int, Dataset] = {}
    for parent, members in chips.groupby("parent", sort=True):
        absent = [int(p) for p in members.index if int(p) not in by_chip]
        if absent:
            raise ValueError(
                f"window {parent} cannot be merged: chips {absent} have no output; "
                f"predict them first"
            )
        window = parents.iloc[int(parent)]
        height, width = (int(e) for e in window["proj:shape"])
        first = members.iloc[0]
        tiler, halo = layout(
            (height, width),
            tuple(int(e) for e in first["proj:shape"]),
            overlap=int(first["overlap"]),
            mode=first["mode"],
        )
        merger = None
        name = "output"
        for position, chip in members.iterrows():
            output = open_raster(by_chip[int(position)])
            bands = list(output.data_vars)
            name = _shared_name(bands)
            values = np.stack([output[band].values for band in bands])
            if merger is None:
                merger = Merger(tiler, logits=len(bands), window=taper, save_visits=False)
            merger.add(int(chip["chip"]), values)
        grid = GeoBox((height, width), Affine(*window["proj:transform"]), window["proj:code"])
        data = merger.merge(extra_padding=halo)
        merged[int(parent)] = raster({name: (("band", *SPATIAL_DIMENSIONS), data)}, grid)
    return merged


def _shared_name(bands: list[str]) -> str:
    """Name an output after its bands' shared prefix, as `logits` for `logits_0..n`."""
    prefix = os.path.commonprefix(bands).rstrip("_")
    return prefix or "output"
```

Imports: `os`, `numpy as np`. Remove the `xarray` import if it's unused.

- [ ] **Step 4: Run the named tests**

Run: `uv run pytest tests/geodata/cuts/test_merge.py -v`
Expected: all pass.

---

### Task 9: segmentation `Dataset` and `Module` evaluation on window rows

**Files:**
- Modify: `src/geosave_engine/ml/segmentation/supervised/data.py`:
  - `Dataset(manifest, spec, *, target, on=None, ignore_index)`;
  - delete `_Parents`, `parents` and the pickling hooks;
  - `self.frames` and `self.windows` replace `self._frames` and `self.reference`;
  - `DataModule` passes `on=target`.
- Modify: `src/geosave_engine/ml/segmentation/supervised/module.py` (`_merge` and `_Evaluation` keyed by integer parent; labels read through `frames.iloc[parent].gs.to_stack()`).
- Test:
  - `tests/ml/segmentation/supervised/test_data.py`
  - `tests/ml/segmentation/supervised/test_module.py`
  - `tests/ml/test_inputs.py`: the `Samples` helper and `_samples` move to window rows built from COGs written in `tmp_path`.

**Interfaces:**
- Consumes: `stacks`, `frames`, `chips` (Tasks 5–7); `ModelSpec.frames.cut`, `ModelSpec.chips.cut`, `ModelSpec.chips.layout` (forwarding, unchanged).
- Produces:
  - **`Dataset.windows: GeoDataFrame`.** The chip rows with a `RangeIndex`, carrying the anchor's properties as columns, so `model_inputs(spec, rasters, row)` reads row context as before.
  - **`Dataset.frames: GeoDataFrame`.** The frame rows (or the stack rows when the spec has no frames).
  - **`Dataset[i]`** returns `(model_inputs, target, i, valid)`, where `i: int` is the chip row's position.
  - **`Module.predict_step`** returns `(logits, ids)`, with `ids` the integer positions from the batch.

- [ ] **Step 1: Write the failing test**

Add to `tests/ml/segmentation/supervised/test_data.py` a test that builds a prepared catalog the new way: write a stack of `image` and `label` with `to_cog(stack_templates(...))` into `tmp_path`, call `create_items(..., id="s0")` and `to_geoparquet`, then construct `Dataset(open_vector(path), spec, target="label")`. Assert:

```python
    inputs, target, index, valid = dataset[0]
    assert index == 0
    assert target.shape == valid.shape == dataset.spec.chips.shape
    assert list(dataset.windows["parent"].unique()) == [0]
```

Base the spec on the existing `test_data.py` spec helper (find it with `grep -n "def _spec\|ModelSpec.model_validate" tests/ml/segmentation/supervised/test_data.py`).

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/ml/segmentation/supervised/test_data.py -k "windows or index" -v`
Expected: FAIL, because the manifest has no `geosave:stack` column.

- [ ] **Step 3: Implement `Dataset`**

```python
    def __init__(self, manifest, spec, *, target="label", on=None, ignore_index=255):
        if spec.chips is None:
            raise ValueError("the model spec declares no chips to cut samples into")
        if not spec.inputs:
            raise ValueError("the model spec declares no inputs for the model")
        self.spec, self.target, self.ignore_index = spec, target, ignore_index
        groups = (*spec.input_rasters, target)
        wanted = manifest[manifest["collection"].isin(groups)]
        whole = cuts.stacks(wanted, on=on or target)
        for position, sources in enumerate(whole["sources"]):
            present = set(orjson.loads(sources))
            if missing := [g for g in groups if g not in present]:
                raise ValueError(
                    f"window {position} ({whole.iloc[position]['anchor']}) has no "
                    f"{missing}; its collections are {sorted(present)}. A raw "
                    f"catalog is prepared for a model with spec.prepare"
                )
        self.frames = whole if spec.frames is None else spec.frames.cut(whole)
        self.windows = spec.chips.cut(self.frames)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        row = self.windows.iloc[index]
        rasters = row.gs.to_stack().gs.rasters
        target = _read_target(rasters[self.target], self.ignore_index)
        inputs = model_inputs(self.spec, rasters, row)
        valid = torch.ones(target.shape, dtype=torch.bool)
        for name in self.spec.pixel_inputs:
            pixels = inputs[name]
            valid &= torch.isfinite(pixels).reshape(-1, *target.shape[-2:]).all(dim=0)
        return inputs, target, index, valid
```

Delete `_Parents`, `parents`, `__getstate__`, the `STACK_PROP` import and the `table` import. Keep `_read_target` and `model_inputs`.

- [ ] **Step 4: Implement `Module._merge`**

Replace the dataset lookups:
- `_Evaluation(remaining=set(range(len(dataset.windows))))`.
- `rows = dataset.windows.iloc[[int(i) for i in ids]]`.
- `parent_id = int(row.parent)`.
- The layout comes from `dataset.spec.chips.layout(tuple(int(e) for e in dataset.frames.iloc[parent_id]["proj:shape"]))`.
- Merger add is unchanged (`int(row.chip)`).
- Completion check: `expected = dataset.windows.index[dataset.windows["parent"] == parent_id]`.
- Labels:

```python
            parent = dataset.frames.iloc[parent_id].gs.to_stack()
            labels = parent[dataset.target].dataset
            image = next(iter(parent.gs.rasters.values()))
```

The rest (grid check, `_read_target`, metrics) is unchanged. `_Evaluation` type annotations become `set[int]`, `dict[int, Merger]` and `dict[int, list[tuple[int, int]]]`. `predict_step`'s docstring says the ids are window positions.

- [ ] **Step 5: Move `tests/ml/test_inputs.py` and `test_module.py` helpers**

`_samples(parents, shape, ...)` takes in-memory rasters today. Rewrite it to:
1. write each parent as COGs under a `tmp_path` it receives (`stack_templates` for stacks, `cog_template` for a lone raster, named `image`);
2. add a `label` layer, the first band cast to `uint8`, so `stacks(on="label")` has an anchor;
3. `create_items(..., id=key)`, then `cuts.chips(cuts.stacks(rows, on="label"), shape, overlap=..., mode=...)`.

`Samples.read(position)` becomes `self.windows.iloc[position].gs.to_stack()` with the `label` group dropped, and `Samples.reference` is renamed `windows`.

Tests asserting string ids (`"scene-a/chip-0"`) assert integer positions instead. In-memory unreferenced-pixel cases (no CRS) are deleted, because windows are georeferenced only (spec).

- [ ] **Step 6: Run the named tests**

Run: `uv run pytest tests/ml -q`
Expected: all pass except the `ChipWriter` test, which Task 10 rewrites.

---

### Task 10: `ChipWriter` writes chip COGs and an item table; `ModelSpec.prepare(on=)` writes COGs

**Files:**
- Modify: `src/geosave_engine/ml/segmentation/callbacks.py`, `ChipWriter`:
  - on each batch, write one COG per chip on the chip row's grid;
  - at predict end, write `chips-<rank>.parquet`;
  - `read(output_dir) -> GeoDataFrame` concatenates the rank tables.
- Modify: `src/geosave_engine/model/spec/model.py`, `prepare(raw, output, *, on, keep=())`:
  - iterate `cuts.stacks(raw, on=on)`;
  - read each row with `to_stack()`;
  - preprocess;
  - write `stack({...}).gs.to_cog({collection: template})` under `draft/<anchor>/`;
  - `create_items(files, id=<anchor>)`;
  - `write_catalog`;
  - delete `_store_items` usage and the `STACK_PROP` checks.
- Modify: `src/geosave_engine/model/spec/cuts.py`. Update the docstring examples (no `"s0/frame-0"` ids); the forwarding is unchanged.
- Test: `tests/ml/segmentation/supervised/test_module.py` (`test_predicted_chips_are_saved_then_merged_on_the_source_grid`, `test_a_chip_writer_refuses_chips_left_by_an_earlier_run`), `tests/model/spec/test_prepare.py`.

**Interfaces:**
- Consumes: `Dataset.windows`/`frames` (Task 9), `cuts.merge` (Task 8), `create_items` (Task 3).
- Produces:
  - **`ChipWriter(output_dir, *, name="logits")`.** Each chip `i` is written to `output_dir/chip-<i>.tif`, holding bands `f"{name}_{c}"`.
  - **`ChipWriter.read(output_dir) -> GeoDataFrame`.** An item table with a `chip` column, ready for `cuts.merge(dataset.frames, dataset.windows, ChipWriter.read(dir))`.
  - **`ModelSpec.prepare(raw, output, *, on: str, keep: Sequence[str] = ())`**, where `on` is the anchor collection.

- [ ] **Step 1: Write the failing tests**

Rewrite `test_predicted_chips_are_saved_then_merged_on_the_source_grid` so that it:
- builds samples with the Task 9 `_samples(..., tmp_path)`;
- predicts with `ChipWriter(tmp_path / "chips")`;
- then asserts:

```python
    predictions = ChipWriter.read(tmp_path / "chips")
    assert sorted(predictions["chip"]) == list(range(len(samples.windows)))
    merged = cuts.merge(samples.frames, samples.windows, predictions, taper="hann")
    (result,) = merged.values()
    assert result.gs.geobox == scene.gs.geobox
    torch.testing.assert_close(
        torch.from_numpy(result["logits"].values).float(), scene.gs.to_tensor(), rtol=1e-5, atol=1e-6
    )
```

`test_a_chip_writer_refuses_chips_left_by_an_earlier_run` creates `tmp_path / "chips" / "chips-0.parquet"` instead of a `.zarr`.

In `tests/model/spec/test_prepare.py`, change every `spec.prepare(raw, out, keep=["label"])` to `spec.prepare(raw, out, on="label", keep=["label"])`. Build `raw` with `to_cog` + `create_items(..., id=...)` in place of `create_stack_items`, and assert that the prepared catalog's assets end in `.tif`:

```python
    rows = open_vector(_spec().prepare(raw, tmp_path / "prepared", on="label", keep=["label"]))
    assert all(a["href"].endswith(".tif") for assets in rows["assets"] for a in assets.values() if a)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/ml/segmentation/supervised/test_module.py -k "chip" tests/model/spec/test_prepare.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `ChipWriter`**

```python
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        """Write each chip's output as a COG on the chip's own grid."""
        outputs, ids = prediction
        windows = _windows(trainer, dataloader_idx)
        values = outputs.detach().float().cpu().numpy()
        for index, value in zip(ids, values, strict=True):
            row = windows.iloc[int(index)]
            h, w = (int(e) for e in row["proj:shape"])
            grid = GeoBox((h, w), Affine(*row["proj:transform"]), row["proj:code"])
            planes = value if value.ndim == 3 else value[None]
            bands = {f"{self.name}_{c}": (("y", "x"), plane) for c, plane in enumerate(planes)}
            (file,) = raster(bands, grid).gs.to_cog(f"{self.output_dir}/chip-{int(index)}.tif")
            self._files.append(file)
            self._chips.append(int(index))

    def on_predict_epoch_end(self, trainer, pl_module) -> None:
        """Catalogue this process's chips as one item table."""
        rows = create_items(self._files, collection=create_collection("chips", description="Chip outputs"))
        rows["chip"] = self._chips
        rows.gs.to_geoparquet(self._table(trainer))

    @staticmethod
    def read(output_dir: str | Path) -> GeoDataFrame:
        """Read every process's chip table as one item table."""
        parts = sorted(Path(output_dir).glob("chips-*.parquet"))
        if not parts:
            raise FileNotFoundError(f"{output_dir} holds no saved chips")
        return GeoVector.concat(open_vector(part) for part in parts)
```

- **`_table`.** `_table(trainer)` returns `self.output_dir / f"chips-{trainer.global_rank}.parquet"`. `on_predict_start` refuses if that file exists, and resets `self._files = []` and `self._chips = []`.
- **`_windows`.** `_windows(trainer, i)` returns the dataset's `windows` from `trainer.predict_dataloaders`, which is a list or a single loader, so handle both as `Module._merge` does.
- **Imports, inside the module:** `GeoBox`, `Affine`, `raster`, `GeoVector`, `open_vector`, and `create_items`/`create_collection` from `geosave_engine.geodata.stac`.
- **The `chip` column has to survive `to_geoparquet`.** stac-geoparquet stores extra columns as Item properties, so `open_vector` returns it. Assert that in the test.

- [ ] **Step 4: Implement `prepare(on=)`**

In the loop, replace `rows.gs.to_stack()`, `to_zarr` and `create_stack_items` with:

```python
            windows = cuts.stacks(raw[raw["collection"].isin(needed)], on=on)
            for _, window in windows.iterrows():
                name = window["anchor"]
                rasters = window.gs.to_stack().gs.rasters
                results = self.preprocess(rasters)
                prepared = stack(
                    {
                        **{group: results[group] for group in self.input_rasters},
                        **{group: rasters[group] for group in keep},
                    }
                )
                templates = {
                    group: f"{draft}/{name}/{group}/{group}_{{time:%Y%m%dT%H%M%S}}.tif"
                    if "time" in raster.dims
                    else f"{draft}/{name}/{group}.tif"
                    for group, raster in prepared.gs.rasters.items()
                }
                files = prepared.gs.to_cog(templates)
                items.append(create_items(files, id=name, datetime=(window["start_datetime"], window["end_datetime"])))
            catalog = GeoVector.concat(items)
            catalog.gs.to_geoparquet(draft, collections=collections)
```

The missing-collection checks move to the per-window `sources` check, as in Task 9's `Dataset`. Delete the `STACK_PROP` import and checks, `_TIME_COLUMNS`, if it's now unused, and the `_store_items` import. The folder-safety check uses `name = window["anchor"]`. The docstring gains `on: Collection whose items are the windows, as labels are.`

- [ ] **Step 5: Run the named tests**

Run: `uv run pytest tests/ml tests/model -q`
Expected: all pass.

---

### Task 11: workspace templates and docs on the new flow

**Files:**
- Modify:
  - `src/geosave_engine/templates/workspaces/segmentation/**` (scripts and README using `prepare(...keep=...)`, `cuts.*` or `ChipWriter.read`);
  - `docs/guides/architecture.md`, `docs/guides/workflows.md`;
  - `src/geosave_engine/model/README.md`;
  - `README.md` and `AGENTS.md` where they describe removed APIs.
- Test: `tests/cli/core/test_workspace.py` (generated workspace runs).

- [ ] **Step 1: Find every stale mention**

Run: `grep -rn "create_stack_items\|geosave:stack\|select_pixels\|select_times\|cuts.select\|stac.table\|to_stack()\|\.to_zarr(\"prepared\|ChipWriter.read\|keep=\[" src/geosave_engine/templates docs/guides README.md AGENTS.md src/geosave_engine/model/README.md`

- [ ] **Step 2: Rewrite each mention, file by file**

The flow to describe, in code:

```python
files = sample.gs.to_cog({"s2": "data/s0/s2/s2_{time:%Y%m%d}.tif", "label": "data/s0/label.tif"})
rows = gs.stac.create_items(files, id="s0")
rows.gs.to_geoparquet("data/catalog.parquet")

windows = gs.cuts.chips(gs.cuts.frames(gs.cuts.stacks(gs.open_vector("data/catalog.parquet"), on="label"), 4, tolerance="10D"), 224)
chip = windows.iloc[0].gs.to_stack()

prepared = spec.prepare(gs.open_vector("data/raw.parquet"), "data/prepared", on="label", keep=["label"])
merged = gs.cuts.merge(dataset.frames, dataset.windows, ChipWriter.read("runs/pred"), taper="hann")
```

- [ ] **Step 3: Run the named tests**

Run: `uv run pytest tests/cli -q`
Expected: all pass.

---

### Task 12: removals; full suite green

**Files:**
- Delete:
  - `src/geosave_engine/geodata/stac/extensions/geosave.py`, `zarr.py`, `datacube.py`, and their tests (`tests/geodata/stac/extensions/test_geosave.py`, `test_zarr.py`, `test_datacube.py`);
  - the GeoSave schema file, if one exists under `schemas/geosave/`;
  - `src/geosave_engine/geodata/cuts/read.py` and `tests/geodata/cuts/test_read.py`.
- Modify:
  - `stac/extensions/__init__.py`: `ASSET = (projection, raster, eo, classification)`, exporting no `GeosaveExtension` or `ZarrExtension`.
  - `stac/_item.py`:
    - remove `create_stack_items`, `_store_items`, and `.zarr`/`.nc`/`.nc4`/`.cdf` from `_MEDIA_TYPES`;
    - `create_item` refuses those suffixes with the same message as `create_items` once had (`"... save the scenes with to_cog ..."`).
  - `stac/__init__.py`: drop `create_stack_items` from the exports and from `__all__`, and update Plan 1's namespace test accordingly.
  - `stac/_table.py`: delete `load`.
  - `core/vector.py`: delete `GeoVector.to_stack`.
  - `cuts/__init__.py`: drop the `select*` exports.
- Test: the whole suite.

- [ ] **Step 1: Write the failing test**

Add to `tests/geodata/stac/test_item.py`:

```python
def test_a_named_asset_cannot_be_a_datacube_store(scene, tmp_path) -> None:
    store = scene.gs.to_zarr(tmp_path / "cube.zarr")

    with pytest.raises(ValueError, match="to_cog"):
        stac.create_item({"image": store}, id="cube")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/geodata/stac/test_item.py::test_a_named_asset_cannot_be_a_datacube_store -v`
Expected: FAIL (it does not raise).

- [ ] **Step 3: Delete and edit as listed in Files**

Then list what's left:

```bash
grep -rn "create_stack_items\|_store_items\|GeosaveExtension\|STACK_PROP\|ZarrExtension\|DatacubeExtension\|select_pixels\|select_times\|cuts\.select\|table\.load\|\.gs\.to_stack()" src tests --include=*.py
```

Expected: only `row.gs.to_stack()` / `window.gs.to_stack()` / `chip.gs.to_stack()` calls on Series remain. Fix every other hit, file by file. Tests that asserted the removed behaviour are deleted. Tests that used `create_item` with Zarr stores (`test_the_caller_names_the_assets_of_one_item`, `test_every_extension_module_writes_onto_each_asset`) switch to COGs.

- [ ] **Step 4: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, and ruff passes. Run `uv run pytest tests/test_layering.py -q` as well; it must pass.

---

## Handoff

Report:
1. **What changed.**
2. **Checks run:** the final `uv run pytest -q` counts and ruff.
3. **Breaking changes:**
   - `to_cog` returns `CogFile`s;
   - the `{group}` field becomes `{collection}`;
   - `open_raster`/`open_stack` read records, not path lists or COG folders;
   - `create_items` takes records;
   - `create_stack_items`, `geosave:stack`, the datacube and zarr extensions, `StacMetadata`, `item_properties` and `asset_fields` are gone;
   - `cuts` rows and their columns are new;
   - `Dataset` ids are integers;
   - `ChipWriter` writes COGs and parquet;
   - `prepare` needs `on=`.
4. **Notebooks still using removed names,** which are left untouched by rule.
