# Examples cycle 1: `.gs.to_tensor()` and notebook 0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put tensor conversion on the `gs` accessors, delete `ml/inputs.py`, and ship notebook 0, "Train on image files".

**Architecture:** One helper in `geodata/core/base.py` turns an accessor's `to_numpy` into a tensor and imports torch inside the call. `GeoArray`, `GeoRaster` and `GeoStack` each gain a `to_tensor` beside their `to_numpy`. `model_inputs` moves to its one caller. Notebook 0 reads plain RGB and label GeoTIFFs from a folder, trains a tiny convolutional encoder / decoder / head with a plain `LightningModule`, and plots the prediction beside the label.

**Tech Stack:** xarray accessors, PyTorch, Lightning, nbformat / nbclient, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-examples-series-design.md`

## Global Constraints

- `import geosave_engine` and `import geosave_engine.geodata` must not import torch. Torch is imported inside the method that needs it.
- Notebooks use `import geosave_engine as gs` as the only alias for data.
- Notebook 0 uses no geospatial vocabulary: no catalog, chips, STAC, spec, CRS, bands, `chain_step`.
- Around five code cells per notebook; one sentence of markdown states the problem before each cell; every cell ends in something visible.
- Notebook 0 runs offline.
- No compatibility alias for `geosave_engine.ml.inputs`; the module is deleted.
- Public methods carry concise Google-style docstrings.
- Do not commit. The working tree holds unrelated uncommitted work; the user commits.
- Do not touch `.ipynb` files other than the ones this plan names.

## Review Focus

1. A lazy (dask-backed) raster: `to_tensor` computes it and returns a writable tensor. Test in Task 1.
2. A raster whose variables differ in dtype with `dtype=None`: refuses with the same `ValueError` as `to_numpy`, not a torch error. Test in Task 1.
3. A stack whose groups differ in dtype: each group keeps its own dtype. Test in Task 1.
4. Importing the package in a fresh process: torch is not loaded until `to_tensor` is called. Test in Task 1.
5. Notebook 0 with no network: plots caption the place through a Nominatim request. The notebook must still run when that request fails. Check in Task 4, Step 6.

---

### Task 1: `.gs.to_tensor()` on array, raster and stack

**Files:**
- Modify: `src/geosave_engine/geodata/core/base.py` (add module function `as_tensor`)
- Modify: `src/geosave_engine/geodata/core/array.py` (method after `to_numpy`, line ~173)
- Modify: `src/geosave_engine/geodata/core/raster.py` (method after `to_numpy`, line ~532)
- Modify: `src/geosave_engine/geodata/core/stack.py` (method after `to_numpy`, line ~457)
- Test: `tests/geodata/core/test_model_io.py`

**Interfaces:**
- Produces: `DataArray.gs.to_tensor(*, dtype: str | torch.dtype | None = None) -> torch.Tensor`, the same on `Dataset.gs`, and `DataTree.gs.to_tensor(*, dtype=None) -> dict[str, torch.Tensor]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/geodata/core/test_model_io.py` (it already defines `optical()`: two uint16 bands `B04`=4, `B08`=8 over `time: 2`, `y: 8`, `x: 8`):

```python
import subprocess
import sys

torch = pytest.importorskip("torch")


def test_to_tensor_keeps_the_pixel_dtype_and_the_to_numpy_layout() -> None:
    tensor = optical().gs.to_tensor()

    assert tensor.dtype is torch.uint16
    assert tuple(tensor.shape) == (2, 2, 8, 8)
    assert tensor.is_contiguous()
    torch.testing.assert_close(tensor, torch.tensor(optical().gs.to_numpy()))


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [("float32", torch.float32), (torch.int16, torch.int16), (torch.bfloat16, torch.bfloat16)],
)
def test_to_tensor_casts_to_a_named_or_torch_dtype(dtype, expected) -> None:
    assert optical().gs.to_tensor(dtype=dtype).dtype is expected
    assert optical().B04.gs.to_tensor(dtype=dtype).dtype is expected


def test_to_tensor_refuses_an_unknown_dtype_name() -> None:
    with pytest.raises(ValueError, match="not-a-dtype"):
        optical().gs.to_tensor(dtype="not-a-dtype")


def test_a_band_converts_without_gaining_a_band_axis() -> None:
    tensor = optical().B04.gs.to_tensor()

    assert tensor.dtype is torch.uint16
    assert tuple(tensor.shape) == (2, 8, 8)


def test_a_stack_converts_each_group_on_its_own_dtype() -> None:
    from geosave_engine.geodata.core.stack import stack

    scene = stack(
        {"optical": optical(), "height": optical()[["B04"]].astype("float32")}
    )

    tensors = scene.gs.to_tensor()
    assert tuple(tensors) == ("optical", "height")
    assert tensors["optical"].dtype is torch.uint16
    assert tensors["height"].dtype is torch.float32
    assert scene.gs.to_tensor(dtype="float32")["optical"].dtype is torch.float32


def test_to_tensor_copies_read_only_pixels() -> None:
    band = optical().B04
    band.values.flags.writeable = False
    before = band.values.copy()

    tensor = band.gs.to_tensor()
    tensor.fill_(0)

    np.testing.assert_array_equal(band.values, before)


def test_to_tensor_computes_a_lazy_raster() -> None:
    tensor = optical().chunk({"y": 4}).gs.to_tensor()

    assert tuple(tensor.shape) == (2, 2, 8, 8)
    tensor.fill_(0)


def test_to_tensor_refuses_variables_on_different_dtypes() -> None:
    mixed = optical().assign(B08=optical().B08.astype("float32"))

    with pytest.raises(ValueError):
        mixed.gs.to_tensor()


def test_importing_geodata_does_not_import_torch() -> None:
    code = (
        "import sys, geosave_engine, geosave_engine.geodata; "
        "assert 'torch' not in sys.modules, 'torch was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/geodata/core/test_model_io.py -k "to_tensor or converts or importing" -v`
Expected: every new test but `test_importing_geodata_does_not_import_torch` FAILS with `AttributeError: 'GeoRaster' object has no attribute 'to_tensor'` (or `GeoArray` / `GeoStack`). The import test PASSES; it guards the constraint from here on.

- [ ] **Step 3: Add the shared helper**

In `src/geosave_engine/geodata/core/base.py`, add `import numpy as np` to the imports if absent, add under the existing `if TYPE_CHECKING:` block:

```python
    from collections.abc import Callable

    import torch
    from numpy.typing import DTypeLike
```

(skip any name already imported there), and add this module-level function above `class GeoRasterAccessor`:

```python
def as_tensor(
    to_numpy: Callable[..., np.ndarray], dtype: str | torch.dtype | None
) -> torch.Tensor:
    """Read an accessor's model-input array as a writable, contiguous tensor.

    Args:
        to_numpy: The accessor's own `to_numpy`, which fixes the axis order.
        dtype: Torch dtype or its string name. None keeps the pixel dtype. A
            dtype NumPy cannot hold, such as bfloat16, reads as float32 first.

    Returns:
        Tensor laid out as `to_numpy` lays out its array.

    Raises:
        ValueError: A dtype name is unknown.
        TypeError: The pixel dtype has no tensor equivalent.
    """
    # Imported here so that importing geodata never pays for torch.
    import torch

    if isinstance(dtype, str):
        target = getattr(torch, dtype, None)
        if not isinstance(target, torch.dtype):
            raise ValueError(f"Unknown torch dtype {dtype!r}")
    else:
        target = dtype

    # Read-only pixel buffers are copied, so tensor operations can safely write.
    if target is None:
        values = np.require(to_numpy(), requirements=["C", "W"])
        try:
            return torch.as_tensor(values)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Cannot convert numpy dtype {values.dtype} to a torch tensor"
            ) from error

    try:
        reading = torch.empty(0, dtype=target).numpy().dtype
    except TypeError:
        reading = np.dtype("float32")
    return torch.as_tensor(
        np.require(to_numpy(dtype=reading), requirements=["C", "W"]), dtype=target
    )
```

- [ ] **Step 4: Add the three methods**

`src/geosave_engine/geodata/core/array.py`, directly after `GeoArray.to_numpy`:

```python
    def to_tensor(self, *, dtype: str | torch.dtype | None = None) -> torch.Tensor:
        """Read this band as one model-input tensor, laid out as `to_numpy`.

        Args:
            dtype: Torch dtype or its string name. None keeps the pixel dtype.

        Returns:
            Tensor shaped `(*axes, band, y, x)`, or `(*axes, y, x)` where this
            band spans no `band` axis.

        Raises:
            ValueError: A dtype name is unknown.

        Examples:
            >>> ds["ndvi"].gs.to_tensor(dtype="float32").shape
            torch.Size([2, 256, 256])
        """
        return as_tensor(self.to_numpy, dtype)
```

`src/geosave_engine/geodata/core/raster.py`, directly after `GeoRaster.to_numpy`:

```python
    def to_tensor(self, *, dtype: str | torch.dtype | None = None) -> torch.Tensor:
        """Stack every variable into one model-input tensor, laid out as `to_numpy`.

        Args:
            dtype: Torch dtype or its string name. None keeps the source dtype,
                which every variable must then share.

        Returns:
            Tensor shaped `(*axes, band, y, x)`.

        Raises:
            ValueError: A dtype name is unknown, or `to_numpy` refuses the
                raster.

        Examples:
            >>> ds[["B04", "B08"]].gs.to_tensor(dtype="float32").shape
            torch.Size([2, 256, 256])
        """
        return as_tensor(self.to_numpy, dtype)
```

`src/geosave_engine/geodata/core/stack.py`, directly after `GeoStack.to_numpy`:

```python
    def to_tensor(
        self, *, dtype: str | torch.dtype | None = None
    ) -> dict[str, torch.Tensor]:
        """Stack each group's variables into one model-input tensor.

        Args:
            dtype: Cast applied to every group. None keeps each group's own
                dtype.

        Returns:
            {
                group name: tensor shaped `(*axes, band, y, x)`,
            }

        Raises:
            ValueError: A dtype name is unknown, or a group's `to_numpy`
                refuses it.

        Examples:
            >>> {name: t.shape for name, t in scene.gs.to_tensor().items()}
            {'image': torch.Size([4, 256, 256]), 'label': torch.Size([1, 256, 256])}
        """
        return {
            name: raster.gs.to_tensor(dtype=dtype)
            for name, raster in self.rasters.items()
        }
```

In `array.py` and `raster.py`, change the existing `from .base import GeoRasterAccessor` to `from .base import GeoRasterAccessor, as_tensor`. In all three files add `import torch` inside the existing `if TYPE_CHECKING:` block. Each file already has `from __future__ import annotations`, so the annotation costs nothing at run time.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/geodata/core/test_model_io.py -v`
Expected: all PASS, including the pre-existing `to_numpy` tests.

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/geosave_engine/geodata/core tests/geodata/core/test_model_io.py`
Expected: no findings.

---

### Task 2: Delete `ml/inputs.py`

**Files:**
- Delete: `src/geosave_engine/ml/inputs.py`
- Modify: `src/geosave_engine/ml/segmentation/supervised/data.py:18,29` and add `model_inputs`
- Modify: `tests/ml/test_inputs.py`, `tests/model/encoder/test_context.py:3,264`, `tests/ml/segmentation/supervised/test_module.py:3,618,670`, `tests/ml/segmentation/supervised/test_data.py:3,255,550`
- Modify: `AGENTS.md:43`, `docs/guides/architecture.md:108-118,505`, `src/geosave_engine/model/README.md:79,203`, `src/geosave_engine/templates/workspaces/segmentation/README.md:112`

**Interfaces:**
- Consumes: `.gs.to_tensor()` from Task 1.
- Produces: `geosave_engine.ml.segmentation.supervised.data.model_inputs(spec, rasters, row, *, context=None) -> dict[str, torch.Tensor]`, signature unchanged from the deleted module.

- [ ] **Step 1: Point the tests at the new homes**

In `tests/ml/test_inputs.py`:

- Replace the import line with
  `from geosave_engine.ml.segmentation.supervised.data import model_inputs`.
- Delete the tests Task 1 re-homed: `test_native_rasters_convert_with_explicit_tensor_dtype`, `test_read_only_pixels_convert_to_a_safely_mutable_tensor`, `test_tensor_conversion_rejects_unknown_dtype`, `test_to_tensor_preserves_the_prepared_dtype_by_default`, `test_to_tensor_honours_a_requested_dtype`, `test_to_tensor_accepts_a_yaml_dtype_name`, `test_to_tensor_rejects_an_unknown_dtype_name`, `test_a_stack_preserves_and_explicitly_casts_group_dtypes`, `test_tensor_conversion_places_spatial_axes_last`, `test_tensor_conversion_of_one_band_preserves_or_casts_dtype`, and the now-unused `_optical` helper.
- In the remaining code replace `to_tensor(parent)` with `parent.gs.to_tensor()` (two places) and `{"image": to_tensor(tile)}` with `{"image": tile.gs.to_tensor()}`.

In the other three test files delete the line `from geosave_engine.ml.inputs import to_tensor` and rewrite each call:

| File | Before | After |
|---|---|---|
| `tests/model/encoder/test_context.py:264` | `to_tensor(tiles.read(index))` | `tiles.read(index).gs.to_tensor()` |
| `tests/ml/segmentation/supervised/test_module.py:618,670` | `to_tensor(scene)` | `scene.gs.to_tensor()` |
| `tests/ml/segmentation/supervised/test_data.py:255` | `to_tensor(dataset.parents[key][dataset.target].dataset)` | `dataset.parents[key][dataset.target].dataset.gs.to_tensor()` |
| `tests/ml/segmentation/supervised/test_data.py:550` | `to_tensor(tile.gs.rasters["label"])` | `tile.gs.rasters["label"].gs.to_tensor()` |

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/ml/test_inputs.py -x -q`
Expected: collection ERROR, `ImportError: cannot import name 'model_inputs' from 'geosave_engine.ml.segmentation.supervised.data'`.

- [ ] **Step 3: Move `model_inputs` and delete the module**

In `src/geosave_engine/ml/segmentation/supervised/data.py`:

- Delete line 18, `from geosave_engine.ml.inputs import model_inputs, to_tensor`.
- Add `import pandas as pd` and `import xarray as xr` to the run-time imports (remove `xarray` from the `TYPE_CHECKING` block if that leaves it duplicated).
- In `_read_target`, change `to_tensor(data).squeeze(0)` to `data.gs.to_tensor().squeeze(0)`.
- Add after `_read_target`:

```python
def model_inputs(
    spec: ModelSpec,
    rasters: Mapping[str, xr.Dataset | xr.DataArray],
    row: pd.Series,
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare identical model inputs for a training or inference sample.

    Args:
        spec: Raster bindings and optional row-based context recipe.
        rasters: Prepared input rasters, with the sample's window already applied.
        row: Metadata describing the actual selected frames and spatial window.
        context: Explicit cached encoding for this row and recipe.

    Returns:
        Named tensors retaining raster time/band axes.
    """
    values = spec.model_inputs(rasters, row, context=context)
    return {
        name: value.gs.to_tensor()
        if isinstance(value, (xr.Dataset, xr.DataArray))
        else torch.as_tensor(value)
        for name, value in values.items()
    }
```

Then delete the file: `rm src/geosave_engine/ml/inputs.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ml tests/model/encoder/test_context.py tests/geodata/core -q`
Expected: all PASS.

Run: `grep -rn "ml.inputs\|ml/inputs" src tests docs AGENTS.md`
Expected: only the four doc lines Step 5 edits.

- [ ] **Step 5: Update the documents**

`AGENTS.md:43`: change "`geodata` imports no torch" to "`geodata` imports no torch at import time (`.gs.to_tensor()` imports it when called)".

`docs/guides/architecture.md:108-118`: replace the block from "Tensor conversion belongs to ML:" through "same conversion." with:

````markdown
Tensor conversion sits beside NumPy conversion on the accessors:

```python
image = scene[["red", "nir"]].gs.to_tensor(dtype="float32")
layers = sample.gs.to_tensor()  # DataTree group names mapped to tensors
```

Geodata imports no Torch at import time; `to_tensor` imports it when called.
It preserves pixel dtypes by default and reads NumPy-unsupported Torch dtypes
through float32. Model input assembly and target loading use the same
conversion.
````

`docs/guides/architecture.md:505`, `src/geosave_engine/model/README.md:79,203` and `src/geosave_engine/templates/workspaces/segmentation/README.md:112`: replace `geosave_engine.ml.inputs` / `ml.inputs` with `geosave_engine.ml.segmentation.supervised.data` / `ml.segmentation.supervised.data`.

- [ ] **Step 6: Full check**

Run: `uv run pytest -q`
Expected: all PASS.

Run: `uv run ruff check .`
Expected: no new findings in the files this task touched.

---

### Task 3: Example tiles for notebook 0

**Files:**
- Create: `examples/data/tiles/<name>/image.tif` and `label.tif` for the three samples in `examples/data/dw_raw`

**Interfaces:**
- Produces: for each `<name>`, `image.tif` holding three uint8 variables `red`, `green`, `blue` marked as a colour composite, and `label.tif` holding one uint8 variable `label` with its class names and colours, both 510 × 510 on the same grid.

- [ ] **Step 1: Generate the tiles**

Save as a script in the scratchpad (it is run once and not kept in the repository) and run it from the repository root with `uv run python <script>`:

```python
"""Derive plain 8-bit RGB tiles from the committed four-band samples."""

from pathlib import Path

import xarray as xr

import geosave_engine as gs

SOURCE, OUT = Path("examples/data/dw_raw"), Path("examples/data/tiles")
COLOURS = {"red": "B04", "green": "B03", "blue": "B02"}

for folder in sorted(path for path in SOURCE.iterdir() if path.is_dir()):
    sample = gs.read_stack(folder).gs.rasters
    scene = sample["sentinel_2_l2a"].squeeze("time", drop=True)
    # 3000 stored units is a common display stretch for Sentinel-2 true colour.
    rgb = xr.Dataset(
        {name: (scene[band] / 3000).clip(0, 1) * 255 for name, band in COLOURS.items()}
    ).astype("uint8")
    rgb = rgb.gs.write_rgb(red="red", green="green", blue="blue")

    (OUT / folder.name).mkdir(parents=True, exist_ok=True)
    gs.io.geotiff.write_cog(rgb, OUT / folder.name / "image.tif")
    gs.io.geotiff.write_cog(sample["label"], OUT / folder.name / "label.tif")
```

- [ ] **Step 2: Verify what was written**

Run:

```bash
uv run python -c "
from pathlib import Path
import geosave_engine as gs
for folder in sorted(Path('examples/data/tiles').iterdir()):
    image = gs.read_raster(folder / 'image.tif'); label = gs.read_raster(folder / 'label.tif')
    print(folder.name, image.gs.variables, image.red.dtype, dict(image.sizes), label.gs.variables, 'flag_meanings' in label.label.attrs)
"
du -sh examples/data/tiles
```

Expected: three lines, each `('red', 'green', 'blue') uint8 {'x': 510, 'y': 510} ('label',) True`; total about 2 MB.

---

### Task 4: Notebook 0, "Train on image files"

**Files:**
- Create: `examples/00_train_on_image_files.ipynb`
- Delete: `examples/00_quickstart.ipynb` (untracked: deletion is permanent, confirm with the user first)
- Modify: `tests/examples/test_notebooks.py`

**Interfaces:**
- Consumes: `.gs.to_tensor()` (Task 1), `examples/data/tiles/` (Task 3).

- [ ] **Step 1: Write the failing test**

In `tests/examples/test_notebooks.py`, make `_run` return the executed notebook by changing its last line `client.execute()` to `return client.execute()` and its return annotation to `nbformat.NotebookNode`. Replace `test_quickstart_runs` with:

```python
@pytest.mark.slow
def test_train_on_image_files_runs(tmp_path: Path) -> None:
    notebook = _run("00_train_on_image_files.ipynb", tmp_path)

    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) <= 6
    assert all(cell.outputs for cell in code)
    source = "\n".join(cell.source for cell in code)
    # "stac" alone would match `gs.stack`, which this notebook does use.
    for word in ("read_vector", "cuts.", "StacClient", "chain_step", "ModelSpec", "crs"):
        assert word not in source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/examples/test_notebooks.py::test_train_on_image_files_runs -m slow -v`
Expected: FAIL with `FileNotFoundError` for `00_train_on_image_files.ipynb`.

- [ ] **Step 3: Write the notebook**

Build it with `nbformat` from a scratchpad script (`nbformat.v4.new_notebook`, `new_markdown_cell`, `new_code_cell`, then `nbformat.write(nb, "examples/00_train_on_image_files.ipynb")`), with kernel metadata `{"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}`. Cells in order:

Markdown:

```markdown
# 0 · Train on image files

**The project.** We have three satellite pictures and, for each, a second picture that says what is on the ground at every pixel: trees, water, buildings and so on. We want a model that draws the second picture from the first.

This is ordinary image segmentation, and this notebook treats it that way. The pictures happen to be satellite tiles; nothing here needs you to know more than that.
```

Markdown: `Each folder in `data/tiles` holds one picture and its labels. Read a pair and look at it.`

Code:

```python
from pathlib import Path

import geosave_engine as gs

tiles = sorted(Path("data/tiles").iterdir())
image = gs.read_raster(tiles[0] / "image.tif")
label = gs.read_raster(tiles[0] / "label.tif")

gs.stack({"image": image, "label": label}).gs.plot(cols=2)
```

Markdown: `PyTorch wants tensors. `.gs.to_tensor()` gives one shaped `(channels, height, width)`, so a dataset is a few lines.`

Code:

```python
from torch.utils.data import DataLoader, Dataset


class Tiles(Dataset):
    def __init__(self, folders):
        self.folders = folders

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]
        image = gs.read_raster(folder / "image.tif").gs.to_tensor(dtype="float32") / 255
        label = gs.read_raster(folder / "label.tif").gs.to_tensor(dtype="int64")[0]
        return image, label


pixels, classes = Tiles(tiles)[0]
pixels.shape, classes.shape
```

Markdown: `The model has the three parts most segmentation models have: an encoder that turns pixels into features, a decoder that brings the features back to full size, and a head that scores each class. All three are convolutional, so a tile of any size goes straight in.`

Code:

```python
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, channels, 3, stride=2, padding=1), nn.ReLU())

    def forward(self, image):
        return self.net(image)


class Decoder(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU()
        )

    def forward(self, features):
        return self.net(features)


class Head(nn.Module):
    def __init__(self, channels=16, num_classes=11):
        super().__init__()
        self.net = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feature_map):
        return self.net(feature_map)


Head()(Decoder()(Encoder()(pixels[None]))).shape
```

Markdown: `Training is plain Lightning: a module that says how one batch is scored, and a `Trainer` that runs the loop.`

Code:

```python
import lightning as L


class Segmenter(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder, self.head = Encoder(), Decoder(), Head()

    def forward(self, image):
        return self.head(self.decoder(self.encoder(image)))

    def training_step(self, batch, _):
        image, label = batch
        loss = nn.functional.cross_entropy(self(image), label)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-2)


L.seed_everything(0)
model = Segmenter()
trainer = L.Trainer(
    max_epochs=30, accelerator="cpu", logger=False, enable_checkpointing=False,
    enable_progress_bar=False, log_every_n_steps=1,
)
trainer.fit(model, DataLoader(Tiles(tiles), batch_size=3, shuffle=True))
float(trainer.callback_metrics["loss"])
```

Markdown: `Predict the first tile and put the result beside its label. Copying the label and swapping in the predicted classes keeps the class names and colours.`

Code:

```python
model.eval()
with torch.no_grad():
    predicted_classes = model(pixels[None])[0].argmax(0)

predicted = label.copy(data={"label": predicted_classes.numpy().astype("uint8")})
print(f"{float((predicted_classes == classes).float().mean()):.0%} of pixels match the label")
gs.stack({"label": label, "predicted": predicted}).gs.plot(cols=2)
```

Markdown:

```markdown
## What happened

The score looks good and the map is useless: the model learned to call almost everything *trees*, because most pixels are trees. Three pictures, three colours and thirty epochs do not go far.

The rest of the series improves this same project one step at a time:

| Next | Question |
| --- | --- |
| 1 · Improve the training | How do I grow this model and see what training is doing? |
| 2 · A raster is more than pixels | Where did these tiles come from, and what else do they know? |
| 3 · Working with rasters | The satellite records more than three colours. Can the model use them? |
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/examples/test_notebooks.py::test_train_on_image_files_runs -m slow -v`
Expected: PASS, in under two minutes.

- [ ] **Step 5: Read the result as a reader would**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace examples/00_train_on_image_files.ipynb` from `examples/`, then open the notebook. Confirm that the first plot shows a colour picture beside a label with a class legend, the shapes print as `(torch.Size([3, 510, 510]), torch.Size([510, 510]))`, and the last plot shows label beside prediction. If the "What happened" text no longer matches what the prediction shows, rewrite that text to match; do not tune the model to match the text.

- [ ] **Step 6: Check it runs with no network**

Run from `examples/`:

```bash
HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 \
  uv run jupyter nbconvert --to notebook --execute --stdout 00_train_on_image_files.ipynb > /dev/null
```

Expected: exit code 0; the plots draw without a place caption. If a cell raises instead, stop and report it: the plot's place lookup is then a library defect to fix before this notebook ships, not something to work around in the notebook.

- [ ] **Step 7: Remove the old quickstart**

After the user confirms: `rm examples/00_quickstart.ipynb`.

Run: `uv run pytest tests/examples -m slow -q`
Expected: every remaining slow notebook test PASSES.

---

## Known leftovers

- `tests/ml/test_inputs.py` keeps its name although `ml/inputs.py` is gone. Its remaining tests exercise `cuts` reads and `model_inputs` together, and `tests/model/encoder/test_context.py` imports its `_samples` helper. Re-homing it is a separate change.
- `raster.gs.to_cog("x.tif")` refuses a file name; writing one file needs `gs.io.geotiff.write_cog`. Added to the spec's weakness list when this cycle closes.
- Every plot makes a Nominatim request to caption the place. Task 4 Step 6 checks that failure is tolerated; whether a plot should reach the network at all is an open question for the spec's weakness list.
