# Prepare model inputs

Use the same preparation function for training samples and inference chips.
Choose band order, physical units, missing-value handling and tensor shape
explicitly, then pass named tensors to the model.

**You need:** `geosave[ml]` and the [sample scene](../first-raster.md#open-the-sample).
This example prepares two reflectance channels without downloading a model.

```python
import torch

import geosave as gs


def prepare_inputs(scene):
    selected = scene[["red", "nir"]]
    selected.gs.validate()
    reflectance = selected.gs.mask_and_scale()
    image = reflectance.gs.to_tensor(dtype="float32")
    if not torch.isfinite(image).all():
        raise ValueError("Choose a missing-input policy before using this image")
    return {"image": image}


with gs.open_raster("sample-scene.tif") as scene:
    chip = scene.isel(y=slice(0, 128), x=slice(0, 128))
    inputs = prepare_inputs(chip)
inputs["image"].shape
```

```text
torch.Size([2, 128, 128])
```

Band 0 is red and band 1 is near-infrared. Decoding turns declared stored values
into physical reflectance. Tensor conversion reads the pixels. This example
rejects missing inputs; a real project can instead mask or fill them according
to its training policy. Validation checks georeferencing, not whether these
bands and units suit your model.

## Batch samples

A torch Dataset can call `prepare_inputs(chip)` in `__getitem__`. A DataLoader
adds the batch axis, producing `(batch, band, y, x)`. For one inference chip,
add it explicitly:

```python
batch = {"image": inputs["image"].unsqueeze(0)}
batch["image"].shape
```

```text
torch.Size([1, 2, 128, 128])
```

A matching model is called as `model(**batch)`. The sample function does not
normalize for a pretrained checkpoint. Follow the selected encoder's band,
unit and normalization contract; the [pretrained guide](pretrained.md#prepare-bands-values-and-context-together)
shows a six-band Prithvi preparation function and its statistics.

## Dates, locations, and targets

For temporal models, retain the time dimension and prepare matching acquisition
context. Derive dates and location from each actual chip after selection, as
the pretrained guide does. Geometry-changing augmentation also needs an
explicit policy for time and location context; tensor transforms do not update
geographic metadata.

Keep categorical targets separate from decoded imagery. Map class codes to the
loss's expected range and ignore convention. See [training chips](../quickstart.md)
for image/target conversion and [alignment](align-rasters.md) for differing grids.

## Reuse a trained model

To ship preparation with a model, put it in a registered preprocessor class:
`preprocess(raster)` selects and decodes bands, and `normalize(image)`
standardizes a batch. Use the same class in your training Dataset, then attach
it to the release through a processor; `load_processor` rebuilds it with the
saved arguments. See [run the model with a processor](release.md#run-the-model-with-a-processor).

A release stores the class's registered name and constructor arguments, not
its code. Keep the module that registers it importable and versioned with
your experiment, along with normalization statistics and missing-value policy.
