# The File Describes Itself — Implementation Plan

> Historical implementation plan, superseded in part. Current `to_cog` returns
> `tuple[RasterFile, ...]`, each carrying a path and pixel-free header, rather
> than the plain paths proposed below. `create_item` accepts those records or
> paths. Remote GDAL reads translate S3 options into GDAL configuration rather
> than using the proposed fsspec opener. See [I/O](../../reference/io.md) and
> [STAC](../../guides/stac.md) for the implemented contracts.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A saved COG states everything an Item needs, so `to_cog` returns plain paths, `stac.create_item` takes paths, and every GDAL read honours `storage_options`.

**Architecture:** `to_cog` writes a stack layer's name into each file as its ACDD `id` (bands and time are already tags) and returns `tuple[UPath, ...]`. STAC Items are built from each file's header, the lazy Dataset `open_raster` returns, so any future format with a reader can describe itself the same way; `CogFile` stays only inside `cogs.open_raster` for reading a catalog's files under their asset-key names, and leaves the public API. The GDAL reader opens a remote path through its UPath's fsspec filesystem (`rasterio.open(..., opener=fs)`) when storage options are given, so lazy chunk reads carry the credentials.

**Tech Stack:** xarray, rioxarray 0.22, rasterio 1.5 (`opener=`), universal-pathlib, pystac, pytest.

**Spec:** this plan (no separate spec). Design agreed in conversation on 2026-10-12; comparison with odc-geo `write_cog` (returns a path), rio-stac `create_stac_item(source, id=basename, collection=…)` and stactools `create.item(href)` (take a path, read the file).

## Global Constraints

- Public names are singular even when they take a sequence: `create_item`, not `create_items`.
- No compatibility aliases: `create_items` and the public `CogFile` export are removed, every caller updated.
- `geodata` imports no torch at import time; `stac` stays lazily imported from the root.
- Rename file by file, reading each site; no repo-wide regex sweep (see memory: working-tree safety).
- Run tests as `PYTHONPYCACHEPREFIX=<scratch>/pycache uv run python -m pytest -q -p no:cacheprovider …` until `uv sync --reinstall` fixes the stale `.venv` shebangs.
- Docstrings pass `uv run python scripts/check_docstrings.py <file>`; ruff passes on changed files only.

## Review Focus

- A raster loaded from STAC already carries an ACDD `id` (e.g. `sentinel-2-l2a`); written alone, `create_item(paths)` puts it in that collection, and `collection=` must still override it. Test in Task 4.
- A stack layer's own ACDD (title, license) must survive `to_cog`; only `id` is set to the layer name. Test in Task 2.
- A local path with `storage_options=None` must keep GDAL's native reader (no fsspec opener), so `/vsis3` speed and `configure_gdal` setups are unchanged. Test in Task 1.
- `open_raster(to_cog(...))` of a raster with several dates and bands must give the same Dataset as before. Test in Task 3.
- An `hf://buckets/` path keeps its S3-gateway route when no storage options are given. Test in Task 1.

---

### Task 1: GDAL reader honours `storage_options`

**Files:**

- Modify: `src/geosave/geodata/io/raster/gdal.py:49-160` (`read`)
- Test: `tests/geodata/io/raster/test_gdal.py`

**Interfaces:**

- Produces: `gdal.open_raster(source: Location, *, chunks=None, mask_and_scale=False, storage_options: StorageOptions | None = None, **open_options) -> Dataset`, now opening a remote source with options through `opener=<its fsspec filesystem>`.

- [ ] **Step 1: Write the failing tests**

```python
def _tiff_bytes(tmp_path) -> bytes:
    import geosave as gs
    from geosave.geodata import GeoAnchor

    anchor = GeoAnchor.from_coordinates(52, 13, 4, 10)
    band = (("y", "x"), np.full((4, 4), 7, "uint16"))
    (path,) = gs.raster({"red": band}, anchor.geobox).gs.to_cog(tmp_path / "a.tif")
    return Path(path).read_bytes()


def test_a_remote_file_reads_through_the_filesystem_its_options_name(tmp_path) -> None:
    import fsspec

    from geosave.geodata.io.raster.gdal import read

    fsspec.filesystem("memory").pipe("/bucket/a.tif", _tiff_bytes(tmp_path))

    raster = read("memory://bucket/a.tif", chunks={}, storage_options={})

    # Lazy: the chunk read happens here, after `read` returned.
    assert int(raster.red.max().compute()) == 7


def test_a_local_file_keeps_gdals_own_reader(tmp_path, monkeypatch) -> None:
    import rasterio

    from geosave.geodata.io.raster.gdal import read

    path = tmp_path / "b.tif"
    path.write_bytes(_tiff_bytes(tmp_path))
    openers = []
    original = rasterio.open

    def spy(fp, *args, **kwargs):
        openers.append(kwargs.get("opener"))
        return original(fp, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", spy)

    read(path)

    assert openers and all(opener is None for opener in openers)
```

- [ ] **Step 2: Run them to verify the first fails**

Run: `… pytest tests/geodata/io/raster/test_gdal.py -k "remote_file or local_file" -v`
Expected: `test_a_remote_file…` FAILS with `RasterioIOError` (GDAL cannot open `memory://`); `test_a_local_file…` passes already.

- [ ] **Step 3: Implement**

In `read`, replace `with rasterio.open(hf_to_s3(source)) as src:` and the `rioxarray.open_rasterio(src, …)` call:

```python
    # A remote path given options reads through its own filesystem, so every
    # lazy chunk read carries them; otherwise GDAL reads it natively.
    target = storage.location(source, storage_options)
    if storage_options is not None and not storage.is_local(target):
        href, opener = target.path, target.fs
    else:
        href, opener = hf_to_s3(source), None
    with rasterio.open(href, opener=opener) as src:
        block_y, block_x = src.block_shapes[0]
        if chunks == {} and block_x < src.width:
            chunks = {"y": block_y, "x": block_x}
        root, bands = _attrs(src)
    cube = rioxarray.open_rasterio(
        href,
        band_as_variable=True,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        opener=opener,
        **open_options,
    )
```

Update the `storage_options` docstring line: "Options for the filesystem a URL names; a remote file given them is read through that filesystem. None reads with GDAL's own drivers and the keys `configure_gdal` sets."
Add `from geosave.geodata.io import storage` if not already imported.

- [ ] **Step 4: Run the gdal and geotiff tests**

Run: `… pytest tests/geodata/io/raster -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/geosave/geodata/io/raster/gdal.py tests/geodata/io/raster/test_gdal.py
git commit -m "feat(io): read a remote GeoTIFF through the filesystem its options name"
```

---

### Task 2: `to_cog` returns paths and writes the layer as ACDD `id`

**Files:**

- Modify: `src/geosave/geodata/io/raster/cogs.py` (`layout`, `write`, `write_files`)
- Modify: `src/geosave/geodata/core/raster.py:662-733`, `core/array.py:264-300`, `core/datatree.py:507-593` (return type and docstrings of `to_cog`)
- Test: `tests/geodata/io/raster/test_cogs.py`

**Interfaces:**

- Consumes: `gdal.open_raster(..., storage_options=...)` from Task 1.
- Produces:
  - `cogs.write(...) -> tuple[UPath, ...]`, `cogs.write_files(...) -> tuple[UPath, ...]`; each UPath carries `storage_options`.
  - `ds.gs.to_cog(...)`, `da.gs.to_cog(...)`, `tree.gs.to_cog(...)` return `tuple[UPath, ...]`.
  - A stack layer's file states the layer as its ACDD `id`: `open_raster(path).gs.acdd.id == "<layer>"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_to_cog_returns_the_paths_it_wrote(scene, tmp_path) -> None:
    from upath import UPath

    paths = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    assert all(isinstance(path, UPath) for path in paths)
    assert [path.name for path in paths] == ["s_20250601.tif", "s_20250611.tif"]


def test_a_stack_layer_states_its_name_as_the_files_acdd_id(scene, label, tmp_path):
    import geosave as gs

    titled = label.gs.write_acdd(gs.attrs.ACDD(id="old", title="Classes"))
    paths = gs.tree({"optical": scene, "label": titled}).gs.to_cog(
        f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif"
    )

    acdd = {path.stem.split("_")[0]: gs.open_raster(path).gs.acdd for path in paths}
    assert acdd["optical"].id == "optical"
    assert (acdd["label"].id, acdd["label"].title) == ("label", "Classes")
```

- [ ] **Step 2: Run to verify they fail**

Run: `… pytest tests/geodata/io/raster/test_cogs.py -k "returns_the_paths or acdd_id" -v`
Expected: FAIL (`CogFile` is not a `UPath`; no `id` attr).

- [ ] **Step 3: Implement**

In `cogs.py`, add `import geosave.geodata.attrs as attrs` and, beside `layout`:

```python
def _stating_layer(raster: xr.Dataset, collection: str | None) -> xr.Dataset:
    """Return the raster stating the stack layer it is, as a loaded STAC raster does."""
    if collection is None:
        return raster
    stated = raster.gs.acdd or attrs.ACDD()
    return raster.gs.write_acdd(stated.model_copy(update={"id": collection}))
```

Where `layout` builds `_Planned(path, rasters[path][names], times_of[path], collection)`, build
`_Planned(path, _stating_layer(rasters[path][names], collection), times_of[path], collection)`.

In `write_files`, return the written locations:

```python
    written = []
    for file in files:
        path = write_cog(
            cast("Dataset", file.raster),
            file.path,
            map_scale=map_scale,
            overwrite=overwrite,
            storage_options=storage_options,
            **options,
        )
        written.append(storage.location(path, storage_options))
    return tuple(written)
```

Change the three `to_cog` return annotations to `tuple[UPath, ...]` and their Returns/Examples to show `PosixUPath('/abs/out/20250601.tif')`.

- [ ] **Step 4: Convert existing record assertions in `test_cogs.py`**

Each assertion on a record field reads the file instead:

```python
# before
assert files[0].bands == ("red", "nir")
assert files[0].collection == "optical"
# after
assert open_raster(files[0]).gs.variables == ("red", "nir")
assert open_raster(files[0]).gs.acdd.id == "optical"
```

`file.href` becomes `str(path)`; `file.datetime` becomes `open_raster(path).gs.times[0]`.

- [ ] **Step 5: Run the io tests**

Run: `… pytest tests/geodata/io/raster -q`
Expected: pass. Callers of `open_raster(records)` and `create_items` may fail until Tasks 3–4; note them, do not fix here.

- [ ] **Step 6: Commit**

```bash
git add src/geosave/geodata/io/raster/cogs.py src/geosave/geodata/core/{raster,array,stack}.py tests/geodata/io/raster/test_cogs.py
git commit -m "feat(io): to_cog returns paths; each COG states its layer as ACDD id"
```

---

### Task 3: Readers take `to_cog`'s paths

**Files:**

- Modify: `src/geosave/geodata/io/raster/cogs.py` (`read` takes paths beside its internal records)
- Modify: `src/geosave/geodata/io/openers.py:38-145` (`open_raster`, `open_tree`)
- Test: `tests/geodata/io/test_readers.py`, `tests/geodata/core/test_stack.py`

**Interfaces:**

- Consumes: Task 2's paths.
- Produces:
  - `cogs.open_raster(files: Sequence[Location | CogFile], *, geobox=None, resampling="nearest", **options) -> Dataset`. A path is read as itself: its own variable names, its own date. A `CogFile` (internal, built by `stac.translate.item.files` for `cuts`) renames bands to asset keys as today.
  - `open_raster(source: Location | Sequence[Location], **options)` and `open_tree(source: Location | Mapping[str, Location] | Sequence[Location], **options)`; `open_tree` groups paths by the ACDD `id` each file states.

- [ ] **Step 1: Write the failing tests**

```python
def test_open_raster_reads_back_what_to_cog_wrote(scene, tmp_path) -> None:
    paths = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}_{{band}}.tif")

    restored = open_raster(paths)

    assert restored.gs.variables == ("red", "nir")
    np.testing.assert_array_equal(restored.red.values, scene.red.values)


def test_open_tree_groups_files_by_the_layer_each_states(scene, label, tmp_path):
    paths = stack({"optical": scene, "label": label}).gs.to_cog(
        f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif"
    )

    assert open_tree(paths).gs.groups == ("optical", "label")


def test_open_tree_refuses_files_stating_no_layer(scene, tmp_path) -> None:
    paths = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    with pytest.raises(ValueError, match="name no layer"):
        open_tree(paths)
```

- [ ] **Step 2: Run to verify they fail**

Run: `… pytest tests/geodata/io/test_readers.py -k "to_cog_wrote or groups_files or no_layer" -v`
Expected: FAIL with `TypeError` (the sequence holds no `CogFile`).

- [ ] **Step 3: Implement**

In `cogs.open_raster`, open each entry once and take its instant from what it is:

```python
    instants: dict[pd.Timestamp | None, list[xr.Dataset]] = {}
    for entry in files:
        href = entry.href if isinstance(entry, CogFile) else entry
        raster = read_tiff(href, **{"chunks": {}, **options})
        if isinstance(entry, CogFile):
            # A catalog names a file's bands by their asset keys.
            raster = raster.rename(_names(entry, raster))
            when = entry.datetime
        else:
            times = raster.gs.times
            when = None if times is None else times[0]
            raster = raster.squeeze("time", drop=True) if times is not None else raster
        instants.setdefault(when, []).append(raster)
```

and place, merge and concatenate the opened rasters as the loop below already does (the `read_tiff` call moves up into this loop). Keep the `ValueError` for an empty sequence: "read needs at least one file".

`open_raster`:

```python
    if not isinstance(source, (str, PathLike, UPath)):
        return cogs.open_raster(list(source), **options)
```

`open_tree`, sequence branch:

```python
    if not isinstance(source, (str, PathLike, UPath, Mapping)):
        storage_options = options.get("storage_options")
        layers: dict[str, list[Location]] = {}
        for path in source:
            header = open_raster(path, storage_options=storage_options)
            acdd = header.gs.acdd
            if acdd is None or acdd.id is None:
                raise ValueError(
                    f"{path} and its siblings name no layer; open_raster reads a lone raster"
                )
            layers.setdefault(acdd.id, []).append(path)
        return stack({name: cogs.open_raster(paths, **options) for name, paths in layers.items()})
```

Update both docstrings: "the paths a `to_cog` returned".

- [ ] **Step 4: Run io, core and cuts tests**

Run: `… pytest tests/geodata/io tests/geodata/core tests/geodata/cuts -q`
Expected: pass, except stac callers handled in Task 4.

- [ ] **Step 5: Commit**

```bash
git add src/geosave/geodata/io/raster/cogs.py src/geosave/geodata/io/openers.py tests/geodata/io/test_readers.py tests/geodata/core/test_stack.py
git commit -m "feat(io): open_raster and open_tree read the paths to_cog returns"
```

---

### Task 4: `stac.create_item` builds Items from headers

**Files:**

- Modify: `src/geosave/geodata/stac/create.py` (rename `create_items` → `create_item`; inputs are paths or `{key: path}`; add `storage_options`)
- Modify: `src/geosave/geodata/stac/translate/item.py` (`AssetKey` and `Href` aliases; `Scene.assets: dict[AssetKey, Href]` with its docstring example; `scenes(headers, *, collection)`)
- Modify: `src/geosave/geodata/stac/__init__.py` (export `create_item`)
- Test: `tests/geodata/stac/test_create.py`, `tests/geodata/stac/translate/test_item.py`, `tests/geodata/stac/test_namespace.py`

**Interfaces:**

- Consumes: `open_raster(path, storage_options=...)` (Task 1), files stating their layer (Task 2).
- Produces:

```python
def create_item(
    files: Iterable[Location] | Mapping[str, Location],
    *,
    collection: pystac.Collection | str | None = None,
    id: str | None = None,
    datetime: ItemTime | None = None,
    storage_options: StorageOptions | None = None,
) -> GeoDataFrame

# stac/translate/item.py
# An Asset's key within its Item, as "label", "B04" or "image".
type AssetKey = str
# Absolute path or URL of a saved file, as `absolute_location` spells it.
type Href = str


@dataclass(frozen=True)
class Scene:
    """The saved files of one collection, at one instant, on one grid: one Item.

    Args:
        collection: Collection id the files belong to, or None.
        datetime: Instant the files hold, or None for timeless files.
        grid: Grid the files share, or None where it is not locatable.
        assets: Each Asset's key mapped to the file it is, as
            {
                "image": "/data/s0/optical_20250601.tif",
                "label": "/data/s0/label.tif",
            }
    """

    collection: str | None
    datetime: pd.Timestamp | None
    grid: GeoBox | None
    assets: dict[AssetKey, Href]

def scenes(headers: Mapping[Href, xr.Dataset], *, collection: str | None = None) -> list[Scene]
```

`collection=` given → every Item takes it; None → each header's ACDD `id`, else no collection. A header's asset key is its one variable's name, or `"image"` for several.

- [ ] **Step 1: Write the failing tests** (in `test_create.py`)

```python
def test_a_file_states_its_own_collection(scene, label, tmp_path):
    paths = stack({"optical": scene, "label": label}).gs.to_cog(
        f"{tmp_path}/s0/{{collection}}_{{time:%Y%m%d}}.tif"
    )

    rows = stac.create_item(paths, id="s0", datetime=datetime(2025, 6, 1, tzinfo=UTC))

    assert sorted(rows["collection"]) == ["label", "optical", "optical"]


def test_collection_overrides_what_files_state(scene, tmp_path):
    from geosave.geodata import attrs

    named = scene.gs.write_acdd(attrs.ACDD(id="sentinel-2-l2a"))
    paths = named.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    rows = stac.create_item(paths, collection="image")

    assert rows["collection"].tolist() == ["image", "image"]


def test_create_item_reads_a_remote_file_with_its_options(scene, tmp_path) -> None:
    import fsspec

    (path, _) = scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")
    fsspec.filesystem("memory").pipe("/bucket/s_20250601.tif", path.read_bytes())

    rows = stac.create_item(["memory://bucket/s_20250601.tif"], storage_options={})

    assert rows["id"].tolist() == ["s_20250601"]
```

Rename every `stac.create_items(` call in `test_create.py` to `stac.create_item(`, reading each site. Delete `test_paths_and_records_build_the_same_items` (records are gone). In `test_namespace.py`, replace `"create_items"` by `"create_item"` in the expected `__all__`. In `translate/test_item.py`, build `scenes` from `{href: open_raster(href)}` instead of records.

- [ ] **Step 2: Run to verify they fail**

Run: `… pytest tests/geodata/stac -q`
Expected: FAIL (`create_item` undefined; `scenes` signature).

- [ ] **Step 3: Implement `scenes`**

```python
def scenes(
    headers: Mapping[Href, xr.Dataset], *, collection: str | None = None
) -> list[Scene]:
    """Group saved files into the scenes they make: one per collection, instant and grid.

    Args:
        headers: Each file's href mapped to its raster as `open_raster` opens it.
        collection: Collection of every file. None takes each file's ACDD `id`.

    Returns:
        One scene per collection, instant and grid, in the order the files
        first name them.

    Raises:
        ValueError: A file holds several dates, or two files of one scene would
            be the same asset.
    """
    grouped: dict[tuple[str | None, pd.Timestamp | None, GridKey | None], Scene] = {}
    for href, header in headers.items():
        times = header.gs.times
        if times is not None and len(times) != 1:
            raise ValueError(f"{href} holds {len(times)} dates; an Item is one date")
        when = None if times is None else times[0]
        acdd = header.gs.acdd
        stated = collection or (None if acdd is None else acdd.id)
        grid = header.gs.geobox
        found = grouped.setdefault(
            (stated, when, None if grid is None else _grid_key(grid)),
            Scene(stated, when, grid, {}),
        )
        variables = header.gs.variables
        key = variables[0] if len(variables) == 1 else "image"
        if key in found.assets:
            raise ValueError(
                f"{href} and {found.assets[key]} would both be the asset {key!r} "
                f"of one Item; split them with {{band}} or {{time}}"
            )
        found.assets[key] = href
    return list(grouped.values())
```

In `Scene.stem` and `write`, use the href values directly (`scene.assets.values()` are hrefs; `asset.write(item, key, href, headers[href])`). Delete `translate.asset.key` (its job moved into `scenes`) and its test.

- [ ] **Step 4: Implement `create_item`**

```python
    if isinstance(files, Mapping):
        named = cast("Mapping[str, Location]", files)
        hrefs = {key: absolute_location(path) for key, path in named.items()}
        headers = {href: _header(href, storage_options) for href in hrefs.values()}
        scene = Scene(_identity(collection), None, None, hrefs)
        item_id = scene.stem if id is None else id
        items = [translate.item.write(scene, headers, id=item_id, when=datetime)]
    else:
        headers = {
            absolute_location(path): _header(path, storage_options) for path in files
        }
        if not headers:
            raise ValueError("create_item needs saved raster files")
        found = translate.item.scenes(headers, collection=_identity(collection))
        # Files stating their layers were written from one stack.
        prefixed = collection is None and any(scene.collection for scene in found)
        undated = datetime or translate.item.span(headers.values())
        items = [
            translate.item.write(
                scene,
                headers,
                id=_scene_id(id, scene, headers, prefixed=prefixed),
                when=None if scene.datetime is not None else undated,
            )
            for scene in found
        ]
```

with `_header(path, storage_options)` calling `open_raster(path, storage_options=storage_options)`. Delete `_records` and `_instant`. `_scene_id` reads `headers[next(iter(scene.assets.values()))]`.

- [ ] **Step 5: Run stac tests**

Run: `… pytest tests/geodata/stac -q`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/geosave/geodata/stac tests/geodata/stac
git commit -m "feat(stac)!: create_item builds Items from saved files' headers"
```

---

### Task 5: Move every caller; drop `CogFile` from the public API

**Files (read each site, edit by hand):**

- `src/geosave/geodata/io/__init__.py`, `src/geosave/geodata/io/raster/__init__.py` — remove `CogFile` from imports and `__all__`
- `src/geosave/geodata/stac/translate/item.py` — `files(item)` still builds the internal `CogFile` for `cuts`; import it from `geosave.geodata.io.raster.cogs`
- `src/geosave/model/spec/model.py`, `src/geosave_flows/tasks/labels.py:50` — `create_items` → `create_item`
- `src/geosave/templates/workspaces/{segmentation,regression}/README.md`, `examples/make_dataset.py`
- `docs/basics/io.md`, `docs/first-raster.md`, `docs/guides/{catalogs-and-cuts,model-spec,stac}.md`, `docs/quickstart.md`, `docs/reference/io.md`
- Tests: `tests/geodata/{attrs/groups/test_legend,core/test_base,core/test_catalog,core/test_raster,cuts/conftest,cuts/test_merge,cuts/test_sample,io/test_backend,io/test_remote,io/vector/test_stac_geoparquet,io/vector/test_stac_json,stac/extensions/test_raster,stac/extensions/test_eo,stac/extensions/test_classification,stac/search/test_table,stac/test_dataflow,stac/translate/test_asset,stac/translate/test_item}.py`, `tests/model/spec/{test_prepare,test_acquisition,test_stac}.py`

**Interfaces:**

- Consumes: Tasks 2–4 (`create_item`, paths from `to_cog`, `open_raster(paths)`).
- Produces: no reference to `create_items` or to a public `CogFile` remains outside `io/raster/cogs.py`, `stac/translate/` and `cuts/`.

- [ ] **Step 1: Write the guard test** (`tests/geodata/io/test_public_surface.py`)

```python
def test_cogfile_is_not_public() -> None:
    from geosave.geodata import io
    from geosave.geodata.io import raster

    assert "CogFile" not in io.__all__
    assert "CogFile" not in raster.__all__
```

- [ ] **Step 2: Run it to verify it fails**

Run: `… pytest tests/geodata/io/test_public_surface.py -v`
Expected: FAIL.

- [ ] **Step 3: Edit each file in the list**

Rules, applied site by site:
- `create_items(` → `create_item(`.
- A test reading `file.href` / `file.bands` / `file.collection` off a `to_cog` result reads the path or `open_raster(path)` instead (example in Task 2 Step 4).
- `{"key": path}` mapping calls stay as they are.
- Docs showing `CogFile(...)` output show `PosixUPath(...)`.

After each file: `git diff <file>` and read it.

- [ ] **Step 4: Run the whole suite and checks**

Run:

```bash
… pytest -q
uv run ruff check <changed files>
uv run python scripts/check_docstrings.py src/geosave/geodata/io/raster/cogs.py src/geosave/geodata/io/raster/gdal.py src/geosave/geodata/io/openers.py src/geosave/geodata/stac/create.py
grep -rn "create_items\|from geosave.geodata.io import CogFile" src tests docs examples
```

Expected: all tests pass; grep prints nothing.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor!: callers use create_item and to_cog paths; CogFile is internal"
```

---

## Not in this plan

- Notebook 03 is rewritten from scratch afterwards ("make a dataset"), using `create_item(label_paths)` and `to_cog` paths.
- `GeoCatalog`/`GeoVector` review and `catalog.gs.explore()` follow as their own plans.
