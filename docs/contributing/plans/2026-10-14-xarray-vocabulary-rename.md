# Xarray Vocabulary Rename Implementation Plan

> Later change (2026-10-15): catalog operations use verb names. The callable
> `stacks()` is now `join_collections()`; its implementation remains in
> `catalog/stacks.py`. The original naming decision below is historical.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Name every public geodata object after the xarray/cuts concept it is, so "stack" stops meaning two things and stops promising alignment.

**Architecture:** Three renames, each landing with green tests: the stack-table accessor (`GeoWindows` → `GeoDataTrees`), the DataTree object (`stack` → `tree`), and the accessor classes (`Geo` + xarray class name). No compatibility aliases (Alpha, CLAUDE.md "Changes"). Edits are made by hand per file, never by regex, because "stack" also legitimately means `np.stack`/`xr.concat` stacking and the viz layer stack.

**Tech Stack:** xarray accessors, pandas accessors, pytest, ruff.

**Spec:** The design agreed in the 2026-10-14 conversation, recorded in "Design" below. No separate spec file.

## Design

```text
Concept                  Before                          After
-----------------------  ------------------------------  ----------------------------------
stack listing            catalog.gs.stacks(on=)          unchanged
sample table accessor     GeoWindows in cuts/windows.py   GeoDataTrees in cuts/accessor.py
DataTree constructor     gs.stack({...})                 gs.tree({...})
DataTree reader          gs.open_tree(...)              gs.open_tree(...)
window row reader        row.gs.to_tree()               row.gs.to_tree()
DataTree accessor        GeoDataTree (core/datatree.py)        GeoDataTree (core/datatree.py)
Dataset accessor         GeoRaster (core/raster.py)      GeoDataset (core/dataset.py)
DataArray accessor       GeoArray (core/array.py)        GeoDataArray (core/dataarray.py)
shared base              GeoRasterAccessor               unchanged
```

`frames`, `chips`, `merge`, `GeoWindows`, `GeoCatalog`, `GeoVector`, `GeoSample`, `gs.raster` (the constructor) are unchanged.
"Stack" is left only where it means joining arrays (`np.stack`, `xr.concat`, `DataArray.stack`) or the viz layer stack.

## Global Constraints

- No alias, shim or deprecated re-export of an old name.
- Never `git checkout`/`restore`/`stash` a dirty file; the working tree holds unrelated uncommitted work. Use `git mv` for renames.
- No regex or `sed` renames. Edit each occurrence with the Edit tool, deciding per occurrence whether "stack" is the DataTree sense.
- Do not commit. Staging would sweep in unrelated uncommitted changes; the user commits when ready.
- Ignore `*.ipynb` (CLAUDE.md). `tests/examples/test_notebooks.py` is source and is updated; notebook breakage is reported, not fixed.
- Dated records stay as written: `docs/contributing/audits/*` and `docs/reference/generated/*` (regenerated, not hand-edited). Live specs and plans naming old identifiers get updated.
- Docstrings follow the CLAUDE.md Google style; examples use `import geosave as gs`.

## Review Focus

- A `from .windows import windows` in `cuts/__init__.py` shadows the submodule attribute; `import geosave.geodata.cuts.windows as m` returns the function. Same pattern already exists for `frames` and `chips`, so tests must import via `geosave.geodata.cuts`, not the module path.
- `xr.register_datatree_accessor("gs")` is keyed by attribute name, so renaming the class must keep `"gs"`; test `tree.gs` still resolves.
- Entry-point plugin `io/backend.py` and `safe.py` expose `open_tree`; the engine name string used in `xr.open_datatree(engine=...)` must not change by accident.
- Model spec JSON/YAML that stores `windows`/`stacks` keys or class paths (`model/spec/cuts.py`, `model/spec/model.py`) must still load a spec written with the new names, and a spec with the old key must fail loudly.
- Docstring examples are executed or rendered by docs; a stale `gs.stack` in a doc snippet fails the docs build only at publish time.

---

### Task 0: Baseline

**Files:** none changed.

- [ ] **Step 1: Record the pre-existing failures**

Run: `uv run pytest -x -q --no-header -p no:cacheprovider 2>&1 | tail -30`
Expected: note every failure or error. Anything failing here is not caused by the rename and is excluded from later comparisons. Paste the list into the final report.

- [ ] **Step 2: Add the vocabulary guard test (fails now)**

Append to `tests/test_init.py`:

```python
import pytest

import geosave as gs

OLD_NAMES = ("stack", "open_tree", "GeoDataTree", "GeoRaster", "GeoArray")
NEW_NAMES = ("tree", "open_tree", "GeoDataTree", "GeoDataset", "GeoDataArray")


@pytest.mark.parametrize("name", NEW_NAMES)
def test_xarray_vocabulary_exported(name):
    assert hasattr(gs, name)


@pytest.mark.parametrize("name", OLD_NAMES)
def test_old_vocabulary_removed(name):
    assert not hasattr(gs, name)
```

Run: `uv run pytest tests/test_init.py -q`
Expected: FAIL on every parametrized case that is not yet true. This file goes green at the end of Task 3.

---

### Task 1: `GeoWindows` → `GeoDataTrees`

Decision (2026-10-14): `stacks()` / `catalog.gs.stacks` / `cuts/stacks.py` keep their names. Once the DataTree is `tree`, a stack is the items of several collections and dates stacked over one anchor footprint; `to_tree()` reads it. Only the accessor class and the "window" wording change.

**Files:**
- Modify: `src/geosave/geodata/cuts/windows.py` → `git mv` to `cuts/accessor.py`, class `GeoWindows` → `GeoDataTrees`; `cuts/__init__.py`; `core/vector.py` docstring; `docs/concepts/catalog-windows.md` → `stack-tables.md` wording; docs/specs that say "window table"
- Test: `tests/geodata/cuts/test_windows.py` → `test_accessor.py`

**Interfaces:**
- Produces: `GeoDataTrees` (pandas accessor class of a sample table), exported from `geosave.geodata.cuts`; module `cuts/accessor.py`.

- [ ] **Step 1: Failing test** — in the accessor test assert `type(table.gs).__name__ == "GeoDataTrees"` and `from geosave.geodata.cuts import GeoDataTrees`. Run `uv run pytest tests/geodata/cuts -q`; Expected: FAIL.
- [ ] **Step 2: Rename** — `git mv` module and test, rename class, exports, docstrings ("window table" → "sample table", "window row" → "stack row" where it means a row of `stacks()`; spatial windows of chips stay "window").
- [ ] **Step 3: Run** — `uv run pytest tests/geodata/cuts tests/model -q`; Expected: PASS.

---

### Task 2: `stack` → `tree`

**Files:**
- Rename: `src/geosave/geodata/core/datatree.py` → `core/datatree.py`; `tests/geodata/core/test_stack.py` → `test_datatree.py`; `tests/geodata/io/raster/test_zarr_stack.py` → `test_zarr_tree.py`; `tests/geodata/transform/test_stack_metadata.py` → `test_tree_metadata.py`
- Modify: `geodata/core/__init__.py`, `geodata/__init__.py`, `geosave/__init__.py`, `io/__init__.py`, `io/openers.py`, `io/backend.py`, `io/raster/safe.py`, `transform/{warp,crop,combine}.py` (`_concat_stacks` → `_concat_trees`), `cuts/sample.py` (`to_tree` → `to_tree`), `cuts/accessor.py`, `core/catalog.py`, `core/accessor.py`, `model/spec/model.py`, `geosave_flows/tasks/sample.py`, `viz/*` only where the DataTree sense appears, `ml/transforms/augmenter.py`, `model/stages/encoder/clay.py`, `features/cloud_mask.py`, `attrs/groups/timespec.py`, `conventions.py` (decide per hit)
- Modify tests: every file in the Task-2 list from the inventory (`tests/geodata/{conftest,core,transform,io,cuts,features,stac,viz}`, `tests/flows/{test_ingest,tasks/test_sample}.py`, `tests/ml/...`, `tests/model/spec/test_model.py`, `tests/examples/test_notebooks.py`)

**Interfaces:**
- Produces: `gs.tree(rasters: Mapping[str, Dataset | DataArray]) -> DataTree`; `gs.open_tree(source, **options) -> DataTree`; `GeoSample.to_tree(self, **options) -> DataTree` (row accessor); module `geosave.geodata.core.datatree` exporting `GeoDataTree` and `tree`.

- [ ] **Step 1: Failing tests**

In test files, by hand: `gs.stack(` → `gs.tree(`, `open_tree` → `open_tree`, `to_tree` → `to_tree`, `GeoDataTree` → `GeoDataTree`, `from geosave.geodata.core.datatree import` → `...core.datatree import`. Add to `tests/geodata/core/test_datatree.py`:

```python
def test_tree_builds_datatree_with_gs_accessor(optical, dem):
    built = gs.tree({"s2": optical, "dem": dem})
    assert isinstance(built, xr.DataTree)
    assert type(built.gs).__name__ == "GeoDataTree"
    assert built.gs.groups == ("s2", "dem")
```

Run: `uv run pytest tests/geodata -q -x`
Expected: FAIL (`AttributeError: module 'geosave' has no attribute 'tree'`).

- [ ] **Step 2: Source rename, module by module**

`git mv` the module, then in `core/datatree.py`: `class GeoDataTree` → `class GeoDataTree` (keep `@xr.register_datatree_accessor("gs")`), `def stack(` → `def tree(`, rewrite the module docstring ("A tree is a flat DataTree holding one raster per group…"; drop "stack" nouns, keep the diagram). Then exports in the three `__init__` files, `geodata/__init__.py:55` (`gs: GeoDataTree`), and each caller. For each `stack` hit, keep it if it means `np.stack`/`xr.concat`/`DataArray.stack`/viz layer stack.

- [ ] **Step 3: Run**

Run: `uv run pytest tests/geodata tests/model tests/flows tests/ml -q`
Expected: PASS except `tests/test_init.py` old-name cases for `GeoRaster`/`GeoArray`/`GeoDataset` (Task 3).

- [ ] **Step 4: Docs**

`docs/reference/stack-accessor.md` → `tree-accessor.md` (git mv; fix nav in the Zensical config and every link: `grep -rn "stack-accessor" docs zensical.toml mkdocs.yml pyproject.toml`), then `docs/basics/{io,rasters,index}.md`, `docs/guides/{ingest-files,workflows,large-data}.md`, `docs/reference/{io,rasters,public-api}.md`, `docs/concepts/raster-grids.md`, `README.md`, live specs/plans naming `open_tree`/`to_tree`.

---

### Task 3: Accessor classes mirror xarray

**Files:**
- Rename: `core/raster.py` → `core/dataset.py`, `core/array.py` → `core/dataarray.py`; `tests/geodata/core/test_raster.py` → `test_dataset.py`, `test_array.py` → `test_dataarray.py`; `docs/reference/raster-accessor.md` → `dataset-accessor.md`, `array-accessor.md` → `dataarray-accessor.md`
- Modify: `core/__init__.py`, `geodata/__init__.py`, `geosave/__init__.py`, `core/dataset.py` (`class GeoRaster` → `GeoDataset`), `core/dataarray.py` (`GeoArray` → `GeoDataArray`), `transform/color.py`, `viz/plot.py`, `viz/style.py`, `tests/geodata/core/test_accessor.py`, `docs/reference/public-api.md`
- Keep: `GeoRasterAccessor` base, `gs.raster(...)` constructor in `core/create.py`

**Interfaces:**
- Produces: `GeoDataset(GeoRasterAccessor[Dataset])`, `GeoDataArray(GeoRasterAccessor[DataArray])`, registered as `"gs"` on `xr.Dataset` / `xr.DataArray`.

- [ ] **Step 1: Failing test**

In `tests/geodata/core/test_accessor.py` replace class-name assertions:

```python
def test_accessor_classes_mirror_xarray(optical):
    assert type(optical.gs).__name__ == "GeoDataset"
    assert type(optical["red"].gs).__name__ == "GeoDataArray"
```

Run: `uv run pytest tests/geodata/core/test_accessor.py -q` → FAIL.

- [ ] **Step 2: Rename classes and modules** (`git mv`, then Edit each reference; the viz/color hits are `isinstance`/type hints/docstrings, check each).

- [ ] **Step 3: Run**

Run: `uv run pytest tests -q`
Expected: PASS, and `tests/test_init.py` fully green.

---

### Task 4: Repo-wide docs, CLAUDE.md, generated reference

**Files:**
- Modify: `CLAUDE.md` and `AGENTS.md` structure lines 35 and 39, `docs/index.md`, `docs/first-raster.md`, `docs/concepts/{index,metadata,training}.md`, `docs/guides/*` remaining hits, `scripts/api_reference.py` if it lists names, `docs/reference/geodata.md`
- Regenerate: `docs/reference/generated/data.md`

- [ ] **Step 1: Instruction files**

`core/` line: `# `gs` accessors (dataset, dataarray, datatree, vector, catalog), GeoAnchor, constructors`
`cuts/` line: `# item table -> windows -> frames -> chips; each window row reads as a DataTree`

- [ ] **Step 2: Regenerate and check links**

Run: `uv run python scripts/api_reference.py`
Expected: `generated/data.md` shows only new names. Then `uv run zensical build` (or the docs command in `pyproject.toml`); expected no broken-link warnings for the renamed reference pages.

- [ ] **Step 3: Memory notes**

Update memory files that cite old names (`project_examples_state_2026_10_13.md` etc.) after the user confirms the rename landed.

---

### Task 5: Verification

- [ ] **Step 1: Leftover sweep**

```bash
grep -rnE "GeoDataTree|GeoRaster\b|GeoArray\b|open_tree|to_tree|\.stacks\b|\bstacks\(|gs\.stack\b|core/stack|cuts/stacks" src tests docs scripts README.md CLAUDE.md AGENTS.md --include='*' -I | grep -v "docs/contributing/audits\|\.ipynb"
```

Expected: no output. Remaining bare "stack" hits are reviewed once by eye and must all be array-joining or viz.

- [ ] **Step 2: Checks**

Run: `uv run ruff check . && uv run pytest -q && uv run pytest -m integration -q`
Expected: PASS apart from Task 0 baseline failures.

- [ ] **Step 3: Smoke test of the renamed flow**

```bash
uv run python - <<'EOF'
import geosave as gs
items = gs.open_vector("tests/data/catalog.parquet")  # use the fixture path the cuts tests use
cut = items.gs.windows(on="label").gs.frames(2, tolerance="10D").gs.chips(64)
tree = cut.iloc[0].gs.to_tree()
print(type(tree.gs).__name__, tree.gs.groups)
EOF
```

Expected: `GeoDataTree (...)`. If the catalog path differs, take it from `tests/geodata/cuts/conftest.py`.

- [ ] **Step 4: Report**

Report only: what changed, checks run, breaking changes (every old name is removed with no alias; notebooks `examples/*.ipynb` still use old names and are untouched; user-authored model specs with old keys, if any, now fail).
