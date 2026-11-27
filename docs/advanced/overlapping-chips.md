# Read overlapping training chips

Read overlapping image/label pairs without preparing a separate file for
every chip. Both layers follow the same windows and keep their own values.

Overlap gives neighboring samples some shared context. Split scenes or
regions into train and validation sets before creating these windows, so
shared pixels do not appear on both sides of the evaluation.

**You need:** `geosave[ml]` and the aligned files from
[images to training chips](../basics/training-chips.md).

## Plan overlapping samples

```python
from itertools import islice

import xarray as xr

import geosave as gs

image = gs.open_raster("training-image.tif")
labels = gs.open_raster("training-labels.tif")
tree = xr.DataTree.from_dict({
    "image": image.gs.mask_and_scale(),
    "labels": labels,
})
chips = gs.RasterChipper([tree], size=128, overlap=0.25, mode="shift")
```

Adjacent windows start 96 pixels apart, sharing at least 32 pixels. At the
scene boundary, `shift` moves the last windows inward to keep them inside
the image. Planning windows reads no pixels.

`overlap=0.25` shares a quarter of each 128-pixel side, equivalent to
`overlap=32`. The last move can be shorter than 96 pixels because the
256-pixel scene does not divide evenly into those steps.

## Inspect the middle pair

```python
location, chip = next(islice(chips, 4, None))
chip_plot = chip["image"].to_dataset().gs.plot(
    rgb=("red", "green", "blue"), xlabel="",
) + chip["labels"].to_dataset().gs.plot("label", xlabel="")
chip_plot
```

![The same overlapping window in imagery and land-cover labels](../assets/overlapping-chip.png)

Image and label boundaries stay aligned. Iteration returns a lazy window;
plotting or converting it reads its pixels.

```python
pixels = chip.gs.to_tensor()
{name: tuple(value.shape) for name, value in pixels.items()}
```

```text
{'image': (4, 128, 128), 'labels': (1, 128, 128)}
```

To feed every chip to a model, `chips.iter_batches(4)` yields the chips four
at a time, and `batch.to_tensor()` stacks the same tensors. It reads one block
of chunks at a time, so pixels that overlapping chips share are read once:

```python
for batch in chips.iter_batches(4):
    images = batch.to_tensor()["image"]
```

The sampling package owns chip placement and overlap weighting.
[Sampling reference](../reference/chips.md) ·
[Open prediction decisions](../contributing/design/release-inference.md#open-prediction-decisions)
