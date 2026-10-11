# Example Notebooks Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder and tighten `examples/` so a reader who knows basic PyTorch reaches each idea one step at a time, and add skeletons for the next two lessons (STAC, model spec).

**Architecture:** One project (Dynamic World segmentation) runs through every notebook. Each notebook adds one idea to the same training loop. Order moves from data to model: basics → rasters → bands → model → STAC → model spec. Lessons 4 and 5 are skeletons: headings, prose intent and the calls they will make, no executed outputs.

**Tech Stack:** Jupyter notebooks, `nbclient` (already installed) for the smoke test, Lightning, torchmetrics, kornia, HoloViews.

**Spec:** Review in the conversation of 2026-10-10 (summarised in "Review findings" below).

## Global Constraints

- Examples use `import geosave_engine as gs`.
- Every notebook stays runnable top-to-bottom on CPU in under ~5 minutes, except skeletons.
- Data comes only from `fatmur/dynamicworld-300` via `snapshot_download(..., local_dir="data/dynamic_world")`.
- Prose names a library call only if that call exists in `src/` today.
- No new dependencies.
- Files under `examples/` are untracked; rename with `mv`, never touch the deleted old notebooks in the index.

## Review findings (the spec)

| Notebook | Density | Main problem |
| --- | --- | --- |
| 00 basics | right | `gs.stack` used before it is introduced; `test` split downloaded but never used. |
| 01 model | **too dense** (chain_step, Published, build_model, registry, list_models, Prithvi, augmentation, run comparison) | Sits before the reader knows bands/time/location, so the Prithvi section is a dead end: it loads a 6-band, time+location encoder with a 2-class `["background", "water"]` head that matches nothing, then says "we can't use it". |
| 02 rasters | right | Retrains a model only to get a prediction; the training cell is noise for this lesson. |
| 03 bands | slightly sparse | No per-band normalisation (first thing a PyTorch user asks after `/ 10_000`); `nan_to_num(0)` unexplained; no comparison of the result. |

Repeated across all four: `Tiles` and `Segmenter` are rewritten each time and have drifted (02's `Segmenter` has no metrics, 03's takes `bands`).

Questions a casual reader will have that no notebook answers yet: *why 255?* (answered), *should I normalise?* (no), *my scenes are bigger than 510 px* (no — `gs.cuts.chips`), *my labels are polygons* (no — `.gs.rasterize` / `open_vector`), *how do I get imagery for my place?* (no — STAC), *how do I score on the test split?* (no — `trainer.test`), *how do I ship the model?* (no — `save_model` / `publish_model` / `load_model`).

Library features no notebook shows: `gs.stac` / `StacClient`, `gs.features.scl_valid_mask` and other cloud masks, `gs.cuts.chips` / `frames` / `merge`, vectors (`open_vector`, `.gs.rasterize`, `.gs.vectorize`), `.gs.reproject` / `.gs.crop`, `.gs.to_zarr`, `ml.segmentation.supervised.Module/DataModule`, `SemanticSegmentationMetrics`, `DensePredictionLogger`, `ModelSpec`, `save_model` / `publish_model` / `load_model`, `GeosaveCLI`.

## New order

| File | Lesson | Question | Status |
| --- | --- | --- | --- |
| `00_basics.ipynb` | 0 · Basics | Can I train on satellite tiles like any image dataset? | edit |
| `01_rasters.ipynb` | 1 · Rasters | What else does each picture know? | was 02 |
| `02_bands.ipynb` | 2 · Bands | What do the numbers measure, and what should the model read? | was 03 |
| `03_model.ipynb` | 3 · Model | How do I grow and swap model parts? | was 01, trimmed |
| `04_stac.ipynb` | 4 · STAC | How do I get imagery for my own place and dates? | skeleton |
| `05_model_spec.ipynb` | 5 · Model spec | How does the model remember what inputs it needs? | skeleton |

Later, not in this plan: 6 · Pretrained encoder (Prithvi, moved out of 01; needs the 6 bands + time + location that 4 and 5 provide), 7 · Chips (big scenes → `cuts.chips` → `cuts.merge`), 8 · Release (`save_model`, `publish_model`, `load_model`), 9 · Workspace (`GeosaveCLI`, library `Module`/`DataModule`).

Why this order: data before model. 03 Bands ends with a `prepare()` function; 05 Model spec is that function written as data so it travels with the model. Prithvi belongs after both, when the reader can actually feed it.

## Open decision: shared lesson code

Option A — keep each notebook self-contained (today):

```python
# every notebook from 01 on
class Tiles(Dataset): ...        # 12 lines, copied
class Segmenter(L.LightningModule): ...   # 30 lines, copied
```

Option B — notebook 0 writes them; later notebooks import them:

```python
# examples/lesson.py  (Tiles + Segmenter exactly as notebook 0 writes them)
# 01_rasters.ipynb
from lesson import Segmenter, Tiles

module = Segmenter.load_from_checkpoint(last_checkpoint())  # notebook 0's run, no retraining
```

B removes ~40 copied lines per notebook and lets 01 reuse notebook 0's trained model instead of retraining. Cost: the reader opens a second file. **Recommend B**; tasks below assume B. If A is chosen, drop Task 2 and keep the copies identical instead.

## Review Focus

- Notebook run out of order (01 before 00): `last_checkpoint()` must fail with a message saying "run 00_basics first", not a bare `StopIteration`.
- Re-running a notebook: `lightning_logs/` gains a version; checkpoint lookup must pick the newest.
- Running from repo root instead of `examples/`: paths are relative to the notebook; smoke test sets `cwd=examples/`.
- Skeleton notebooks must not be executed by the smoke test (they call network/unfinished APIs).
- Prose mentions a call that does not exist (`gs.features.ndvi` style drift): smoke test catches executed cells; skeleton calls are checked by Task 7 step 2.

---

### Task 1: Smoke test that runs every lesson notebook

**Files:**
- Create: `tests/examples/test_notebooks.py`

**Interfaces:**
- Produces: `LESSONS` list, the executed order every later task relies on.

- [ ] **Step 1: Write the failing test**

```python
"""Execute the runnable example notebooks in lesson order."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES = Path(__file__).parents[2] / "examples"
LESSONS = ["00_basics", "01_rasters", "02_bands", "03_model"]
SKELETONS = ["04_stac", "05_model_spec"]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("name", LESSONS)
def test_lesson_runs(name):
    notebook = nbformat.read(EXAMPLES / f"{name}.ipynb", as_version=4)
    NotebookClient(notebook, timeout=900, resources={"metadata": {"path": EXAMPLES}}).execute()


@pytest.mark.parametrize("name", SKELETONS)
def test_skeleton_has_no_outputs(name):
    notebook = nbformat.read(EXAMPLES / f"{name}.ipynb", as_version=4)
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
```

`%tensorboard` cells start a server; tag those cells `skip-execution` in each notebook and pass `skip_cells_with_tag="skip-execution"` to `NotebookClient`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/examples/test_notebooks.py -m "slow and integration" -x`
Expected: FAIL, `01_rasters.ipynb` not found.

- [ ] **Step 3: Commit test only after Task 3 makes it pass.**

### Task 2: `examples/lesson.py` (Option B)

**Files:**
- Create: `examples/lesson.py`
- Modify: `examples/00_basics.ipynb` (cells 6, 10: keep the code inline, add one line "this code is saved as `lesson.py` so later notebooks can import it")

**Interfaces:**
- Produces: `root: Path`, `Tiles(split, prepare=None)`, `Segmenter(model, bands=4)`, `last_checkpoint() -> Path`.

- [ ] **Step 1: Write `lesson.py`** — `Tiles` and `Segmenter` copied verbatim from 00 cells 6 and 10, plus:

```python
def last_checkpoint() -> Path:
    """Return notebook 0's newest checkpoint."""
    paths = sorted(Path("lightning_logs").glob("version_*/checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError("No checkpoint yet: run 00_basics.ipynb first.")
    return paths[-1]
```

`Tiles` gains `prepare` (default: `/ 10_000`) so 02 Bands passes its own instead of redefining the class. `Segmenter` calls `self.save_hyperparameters(ignore="model")` is **not** used — `load_from_checkpoint(path, model=...)` passes the model explicitly; keep it that simple.

- [ ] **Step 2: Smoke** `cd examples && uv run python -c "from lesson import Tiles; print(Tiles('val')[0][0].shape)"` → `torch.Size([4, 510, 510])`.

### Task 3: Rename, reorder, fix cross-references

**Files:**
- Rename: `02_rasters → 01_rasters`, `03_bands → 02_bands`, `01_model → 03_model`
- Modify: title cell and closing "Next" table of all four notebooks.

- [ ] **Step 1:** `mv` the three files.
- [ ] **Step 2:** Update titles (`# 1 · Rasters` …) and every "Notebook N" mention (`grep -n "otebook [0-3]" examples/*.ipynb`).
- [ ] **Step 3:** Run Task 1 test; expect PASS for all four (content unchanged yet).
- [ ] **Step 4:** Commit `examples/ tests/examples/` — "docs(examples): reorder lessons data-first".

### Task 4: 00 Basics — small fixes

- [ ] Cell 3: introduce `gs.stack` in one sentence where first used: "`gs.stack` lays several rasters that share a grid side by side; lesson 1 explains it."
- [ ] Cell 12: add `ModelCheckpoint` default is enough; add one sentence that the checkpoint in `lightning_logs/version_N/checkpoints/` is what lesson 1 reloads.
- [ ] Cell 16 (prediction): replace `labels["label"].copy(data=...)` with the same `gs.raster(...).gs.write_legend(...)` pattern lesson 1 teaches? **No** — keep the copy trick here (fewer new names) and let lesson 1 show the general route. Only reword: "Copy those numbers into a copy of the label band, so they keep its grid, class names and colours."
- [ ] Add cell after validation: `trainer.test(module, DataLoader(Tiles("test"), batch_size=8))` with one line on why the test split is touched once, at the end. Requires `test_step` in `Segmenter` (same as `validation_step`, `test_` prefix).
- [ ] Run Task 1 for `00_basics`; PASS.

### Task 5: 01 Rasters — reuse the trained model

- [ ] Delete cell 17 (the retraining `Tiles` + `Segmenter` + `fit`).
- [ ] Replace with:

```python
from lesson import Segmenter, last_checkpoint

module = Segmenter.load_from_checkpoint(last_checkpoint(), model=...)  # model as notebook 0 built it
```

and one sentence: "This is notebook 0's model, loaded from the checkpoint Lightning saved."
- [ ] Fix cell 21: `to_cog` returns `CogFile`, which `open_raster` rejects (`TypeError: 'CogFile' object is not iterable`); use `gs.open_raster(path.href)`.
- [ ] Reword (cell 10): "The image keeps the satellite's exact time; the label, drawn on that picture, the same day." → "The image records the exact time the satellite passed. The label was drawn on that same picture, so it carries the same day."
- [ ] Reword (cell 12): "Without being told, four bands are just four numbers" → "Bare arrays don't say which band is red, green or blue, so `.gs.plot()` won't guess. `write_rgb` tells it."
- [ ] Add one short "Coordinates" sentence answering *what is a CRS*: "`image.gs.crs` names the map projection; the `x`/`y` values are metres in it, not pixel numbers."
- [ ] Run Task 1 for `00_basics, 01_rasters` in order; PASS.

### Task 6: 02 Bands — normalisation and fair comparison

- [ ] After "Choosing the bands", add a **Normalise** section: per-band mean and std from a handful of training tiles via `.gs.statistics()`, then `(x - mean) / std` inside `prepare`. Code:

```python
stats = pd.concat([prepare(gs.open_raster(path).isel(time=0)).gs.statistics()
                   for path in sorted((root / "train/images").iterdir())[:20]])
mean = stats.groupby(level=0)["mean"].mean()
std = stats.groupby(level=0)["stddev"].mean()
```

One sentence: "PyTorch image models expect inputs near zero with similar spread per channel, the role `transforms.Normalize` plays for photos."
- [ ] Explain `nan_to_num(0)`: "after normalising, 0 is the band's average, the least surprising value for a missing pixel."
- [ ] Use `Tiles(split, prepare=prepare)` and `Segmenter(..., bands=6)` from `lesson.py`; delete the redefinitions.
- [ ] Match notebook 0's training budget (`max_epochs`) or say plainly that the runs differ in length, so the TensorBoard comparison is honest.
- [ ] Verify the "lake" / "fields" sentence against the actual first val tile; reword to what the plot shows.
- [ ] Closing: point to 03 Model, not STAC.
- [ ] Run Task 1 for `00..02`; PASS.

### Task 7: 03 Model — trim to one idea per section

Keep: why `nn.Sequential` hurts → `chain_step` + `ModelChain` → swap decoder → `Published` + `build_model` → registry by name → augmentation → train + compare.
Remove: `list_models()` and the Prithvi section (cells 17–21). They move to lesson 6.

- [ ] Delete cells 17–21; closing cell gets one line: "The catalogue also holds pretrained encoders (`gs.model.registry.list_models()`); lesson 6 uses one once lessons 4–5 give it the bands, date and place it needs."
- [ ] Cell 6: lead with the concrete problem in code before the concept — show the two-input decoder failing in `nn.Sequential` (`TypeError: forward() missing 1 required positional argument: 'image'`) and then fixing it with `ModelChain`.
- [ ] Use `Tiles`/`Segmenter` from `lesson.py`; `Segmenter.forward` must call `self.model(image=image)` for a `ModelChain` — add `chain: bool` handling **only** if needed; otherwise subclass in the notebook with the one changed method.
- [ ] Closing table points to 04 STAC.
- [ ] Run Task 1 for all four; PASS. Commit.

### Task 8: Skeletons 04 STAC and 05 Model spec

Skeleton = markdown headings + intent + code cells containing the intended calls, never executed, no outputs.

`04_stac.ipynb`:

| Section | Intent | Calls |
| --- | --- | --- |
| Why STAC | The dataset tiles came from a catalog; get your own. | — |
| Pick a place and dates | Anchor = grid + time window. | `gs.GeoAnchor.from_bbox(bbox, resolution=10, crs=..., timespan=...)` |
| Search | See which scenes exist before loading. | `StacClient.planetary_computer().source("sentinel-2-l2a")`, `.search(anchor)` |
| Load lazily | Same `xarray.Dataset` as `open_raster`. | `.set_config(bands=[...,"SCL"]).load(anchor)` |
| Mask clouds | The step lesson 2 skipped. | `gs.features.scl_valid_mask(scene, scl="SCL")`, `.where(mask)` |
| Predict on it | Lesson 2's `prepare` + checkpoint on new imagery. | `prepare`, `Segmenter.load_from_checkpoint` |
| Save | | `.gs.to_cog(...)` |

`05_model_spec.ipynb`:

| Section | Intent | Calls |
| --- | --- | --- |
| The problem | Lesson 2's `prepare()` lives in a notebook; whoever loads the checkpoint has to know it. | — |
| Rasters the model reads | Bands, packing, nodata as requirements. | `ModelSpec(schema_version=2, rasters={...})` |
| Preprocessing as data | `prepare()` written as named calls. | `preprocessing=`, `inputs=` |
| Normalise | Lesson 2's mean/std as transforms. | `transforms={"image": [{"name": "Normalize", ...}]}` |
| Same inputs everywhere | Training and inference read inputs one way. | `ml.segmentation.supervised.data.model_inputs(spec, rasters)` |
| Save with the model | Preview of lesson 8. | `save_model(model, path, spec=spec)`, `load_spec` |

- [ ] **Step 1:** Write both notebooks with those sections; every code cell starts with `# Skeleton: not runnable yet.`
- [ ] **Step 2:** For each call in the tables, confirm it exists: `uv run python -c "import geosave_engine as gs; gs.GeoAnchor.from_bbox; gs.features.scl_valid_mask; from geosave_engine.model.spec import ModelSpec; from geosave_engine.model.release import save_model"`.
- [ ] **Step 3:** Run `uv run pytest tests/examples/test_notebooks.py -k skeleton`; PASS.
- [ ] **Step 4:** Commit.
