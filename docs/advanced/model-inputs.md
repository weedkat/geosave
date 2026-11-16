# Prepare model inputs

A preprocessor defines band order, decoding, missing values and normalization.
Use the same preprocessor for training and the released model.

Each input channel must mean the same thing in both places. Keeping preparation
together prevents a project from training on reflectance in one band order and
deploying with a different interpretation of the imagery.

**You need:** `geosave[ml]` and the [sample scene](../basics/first-raster.md#open-the-sample).

## Select and decode the bands

The built-in dense preprocessor selects named bands, decodes their stored
values and returns float32 pixel inputs. Here, missing pixels are filled with zero.
This example prepares a two-channel input: red followed by near-infrared.

```python
import xarray as xr

import geosave as gs
from geosave.ml.processing.raster.preprocessing.dense import DensePreprocessor

preprocessor = DensePreprocessor(bands=["red", "nir"], fill=0.0)
image = gs.open_raster("sample-scene.tif")
rasters = xr.DataTree.from_dict({"image": image})
prepared = preprocessor(rasters)
sample = prepared.gs.to_tensor()
sample["image"].shape
```

```text
torch.Size([2, 337, 267])
```

Channel 0 is red; channel 1 is near-infrared. The `image` key is the model's
input name.

Preparation returns a lazy raster tree. Tensor conversion reads its selected
pixels; use [chips](overlapping-chips.md) instead when the whole image is too
large to materialize as one model input.

```python
sample["image"].dtype
```

```text
torch.float32
```

## Match the trained model

Set `mean` and `std` from the training data when the model needs normalization.
The preprocessor standardizes the bands when `mean` and `std` are given. Follow the selected encoder's
[band, unit and context requirements](pretrained.md).

The preprocessor applies band standardization before sampling, using the
training statistics supplied in its settings. This remains lazy until pixels
are read. Reusing those settings keeps training and inference consistent.

Keep categorical labels separate from decoded imagery. Their class codes and
ignore value belong to the training task's [target contract](../concepts/training.md).

## Reuse the preparation

`RasterPipeline` runs this preprocessor around the model. Its release records
the registered name and constructor settings. See [raster prediction](release.md#run-with-a-raster-pipeline).

Projects can develop their own preprocessor implementations. A release stores
one by its class path, so it loads wherever the project's code is installed.
