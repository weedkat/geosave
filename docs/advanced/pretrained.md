# Build from pretrained weights

Install the [ML extra](../basics/install.md#install-to-follow-these-docs) before running this guide.

Reuse a published encoder, add a decoder and a head for your labels, then train
the resulting model. This guide uses
[Prithvi EO 2.0 Tiny TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL),
a small encoder with acquisition-time and location inputs.

## Choose an encoder

```python
from geosave.ml import list_models

list_models().loc["encoder", ["forward", "requires"]]
```

Output:

```text
                                                             forward                 requires
name
clay      forward_pyramid(image, centroid=None, week=None, hour=None)  in_channels, waves, gsd
dinov3                                        forward_pyramid(image)
prithvi                                       forward_pyramid(image)
prithvi_tl          forward_pyramid(image, centroid, year, day_of_year)
```

These names select GeoSave adapters. `model_name` selects an architecture
variant within an adapter. Choose using the data you can supply:

| Encoder | Suitable inputs | Named model inputs | Settings to match to your data | Default `pretrained` |
| --- | --- | --- | --- | --- |
| `dinov3` | One image, with RGB or additional channels. | `image` | `in_channels`, `input_size`; RGB checkpoint kernels are adapted for additional channels. Match RGB channel semantics when retaining those slices. | `True` |
| `prithvi` | Multispectral imagery, with one or several temporal frames. | `image` | HLS `bands`, `num_frames`, and `input_size`. | `False` |
| `prithvi_tl` | Multispectral imagery with acquisition time and location. | `image`, `centroid`, `year`, `day_of_year` | HLS `bands`, `num_frames`, `input_size`, and context from the same raster. | `False` |
| `clay` | One multispectral image with known wavelengths and ground resolution. | `image`, `centroid`, `week`, `hour` | `in_channels`, ordered `waves` in micrometres, `gsd` in metres, and a square chip compatible with the patch size. | `False` |

Prithvi provides v1 100 and v2 300/600 variants; Prithvi TL provides v2
tiny/100/300/600 variants. DINOv3 offers the timm variants listed by its
constructor. Clay offers tiny/small/base/large architectures, with a published
checkpoint for large. See the [stage reference](../reference/model-stages.md)
for exact variant names, patch-size constraints, and local checkpoint options.

For a small geospatial starting point, use the Tiny TL recipe below. Its
encoder, U-Net decoder, and two-class head together have about 8.5 million
parameters. Test with `pretrained=False` on CPU before downloading weights;
this checks the interface, while a trained experiment determines model quality.
Model comparisons, experiment protocols, and measured results are planned for
a separate benchmark repository.

The staged recipe remains the construction API. Named encoder selection does
not select or resample raster bands; prepare the channels and context for that
encoder before calling the chain. The [model notebook](../examples.md) teaches
composition with a small custom model before introducing larger encoders.

## Assemble the model

**This constructor downloads encoder weights from Hugging Face on the first
run.** To try the same architecture offline, change `pretrained` to `False`
before calling `build_model`; every stage will then start with random weights.

```python
import pandas as pd
import torch
import xarray as xr

from geosave.ml import build_model

stages = {
    "encoder": {
        "name": "prithvi_tl",
        "init_args": {
            "model_name": "prithvi_eo_v2_tiny_tl",
            "pretrained": True,
            "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "input_size": 224,
            "num_frames": 1,
        },
    },
    "decoder": {"name": "unet"},
    "head": {
        "name": "segmentation",
        "init_args": {"num_classes": 2},
    },
}
chain = build_model(stages).eval()
```

The encoder receives the published weights. The U-Net decoder and segmentation
head are newly initialized, so this chain still needs labeled training data.
Class `0` means background and class `1` means crop, following `classes` order.

The encoder publishes its pyramid channels and strides to the decoder; the
decoder publishes its output channel count to the head. `build_model` supplies
those constructor arguments automatically. Put stages in encoder → decoder →
head order, and keep the head last for the segmentation training task.

Prithvi defaults to `pretrained=False`; set it explicitly to load weights.
To load the same upstream encoder from a local file, set `pretrained=True`
and `ckpt_path="weights/Prithvi_EO_V2_tiny_TL.pt"` in its `init_args`. That file
must contain the matching Prithvi backbone weights, not a GeoSave release or
a training task checkpoint.

## Call it with the required inputs

```python
inputs = {
    "image": torch.zeros(1, 6, 224, 224),
    "centroid": torch.tensor([[-6.1, 106.8]]),
    "year": torch.tensor([[2025]]),
    "day_of_year": torch.tensor([[152]]),
}
with torch.inference_mode():
    logits = chain(**inputs)
logits.shape
```

```text
torch.Size([1, 2, 224, 224])
```

The zero image demonstrates the tensor contract. It does not demonstrate a
trained crop classifier. Supply prepared, normalized imagery for real data.

| Input | Shape | Values |
| --- | --- | --- |
| `image` | `(batch, 6, 224, 224)` | One frame, six bands in the configured order |
| `centroid` | `(batch, 2)` float32 | Latitude and longitude of each grid's centre |
| `year` | `(batch, 1)` int64 | Year of each frame |
| `day_of_year` | `(batch, 1)` int64 | Day of year of each frame, from 1 |

These are context entries `.gs.to_tensor(context=["centroid", "year",
"day_of_year"])` writes beside the pixels; `geosave.geodata.TENSOR_CONTEXT`
holds their readers. For multiple frames, set `num_frames` accordingly and
supply an image shaped `(batch, time, band, y, x)` with `year` and
`day_of_year` shaped `(batch, time)`.

The tiny encoder uses 16 × 16 spatial patches. Use chip dimensions divisible
by 16 to retain every border pixel. `input_size` is also supplied to this head,
which resizes logits to that configured size. Keep chip size, encoder
`input_size`, and target size aligned when training.

## Prepare bands and values from a raster

Use a project function to prepare pixels; the context is read from the same raster:

```python
def prepare_inputs(scene):
    selected = scene[["B02", "B03", "B04", "B8A", "B11", "B12"]]
    selected.gs.validate()
    reflectance = selected.gs.mask_and_scale()
    spatial = reflectance.gs.spatial_dims
    array = reflectance.to_array(dim="band").transpose("time", "band", *spatial)
    image = torch.tensor(array.to_numpy(), dtype=torch.float32)
    mean = image.new_tensor([0.1087, 0.1342, 0.1433, 0.2734, 0.1958, 0.1363])
    std = image.new_tensor([0.2248, 0.2179, 0.2178, 0.1850, 0.1242, 0.1049])
    image = (image - mean.reshape(1, 6, 1, 1)) / std.reshape(1, 6, 1, 1)
    names = ["centroid", "year", "day_of_year"]
    sample = reflectance.gs.to_tensor("image", context=names)
    sample["image"] = image
    return torch.utils.data.default_collate([sample])

inputs = prepare_inputs(scene)
with torch.inference_mode():
    logits = chain(**inputs)
```

This example assumes one already selected chip with a time dimension and
fractional reflectance decoded from valid scale/offset metadata. The statistics
come from [TerraTorch's Prithvi EO 2 implementation](https://github.com/torchgeo/terratorch/blob/main/terratorch/models/backbones/prithvi_vit.py),
divided by 10,000. Check units and missing-value policy for your dataset.

`bands` configures pretrained encoder channels; the explicit Dataset selection
sets their order. B8A is `NIR_NARROW`, whereas B08 is `NIR_BROAD`. All selected
variables must already share a grid. Reprojection and resampling are explicit.
Derive context from each actual chip, after spatial and temporal selection.

ModelSpec is disabled pending redesign. Call the preparation function explicitly
and pass its returned named tensors to the chain.
