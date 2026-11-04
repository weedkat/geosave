# Build from pretrained weights

```bash
pip install 'geosave[ml]'
```

Reuse a published encoder, add a decoder and a head for your labels, then train
the resulting model. This guide uses
[Prithvi EO 2.0 Tiny TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL),
a small encoder with acquisition-time and location inputs.

## Choose an encoder

```python
from geosave.model import list_models

list_models().loc["encoder", ["forward", "requires"]]
```

Output:

```text
                                                             forward                 requires
name
clay                  forward_pyramid(image, time=None, latlon=None)  in_channels, waves, gsd
dinov3                                        forward_pyramid(image)
prithvi                                       forward_pyramid(image)
prithvi_tl  forward_pyramid(image, temporal_coords, location_coords)
```

These names select GeoSave adapters. `model_name` selects an architecture
variant within an adapter. Choose using the data you can supply:

| Encoder | Suitable inputs | Named model inputs | Settings to match to your data | Default `pretrained` |
| --- | --- | --- | --- | --- |
| `dinov3` | One image, with RGB or additional channels. | `image` | `in_channels`, `input_size`; RGB checkpoint kernels are adapted for additional channels. Match RGB channel semantics when retaining those slices. | `True` |
| `prithvi` | Multispectral imagery, with one or several temporal frames. | `image` | HLS `bands`, `num_frames`, and `input_size`. | `False` |
| `prithvi_tl` | Multispectral imagery with acquisition time and location. | `image`, `temporal_coords`, `location_coords` | HLS `bands`, `num_frames`, `input_size`, and context from the same raster. | `False` |
| `clay` | One multispectral image with known wavelengths and ground resolution. | `image`, `time`, `latlon` | `in_channels`, ordered `waves` in micrometres, `gsd` in metres, and a square chip compatible with the patch size. | `False` |

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
import torch

from geosave.model.registry import build_model

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
    "temporal_coords": torch.tensor([[[2025.0, 151.0]]]),
    "location_coords": torch.tensor([[-6.2, 106.8]]),
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
| `temporal_coords` | `(batch, 1, 2)` | Acquisition year and zero-based day of year; June 1, 2025 is day `151` |
| `location_coords` | `(batch, 2)` | Raster centre as latitude, longitude in degrees |

Context tensors are float32 calendar/geographic values, without normalization.
For multiple frames, set `num_frames` accordingly and supply an image shaped
`(batch, time, band, y, x)` with matching `(batch, time, 2)` temporal coordinates.

The tiny encoder uses 16 × 16 spatial patches. Use chip dimensions divisible
by 16 to retain every border pixel. `input_size` is also supplied to this head,
which resizes logits to that configured size. Keep chip size, encoder
`input_size`, and target size aligned when training.

## Prepare bands, values, and context together

Save this as `model_spec.yaml` for Sentinel-2 variables corresponding to the
six pretrained bands:

```yaml
schema_version: 3
rasters:
  optical:
    variables: [B02, B03, B04, B8A, B11, B12]
    coordinates: [time, x, y]
    require_crs: true
preprocessing:
  image:
    call: !ref optical.gs.mask_and_scale
encoding:
  tensor:
    call: geosave.model.processing.to_tensor
    kwargs: {raster: !ref image, dims: [time, band, y, x]}
  context:
    call: geosave.model.stages.encoder.prithvi.model_context
    kwargs: {raster: !ref image}
inputs:
  image: !ref tensor
  temporal_coords: !ref 'context["temporal_coords"]'
  location_coords: !ref 'context["location_coords"]'
pixels: [image]
transforms:
  image:
    call: geosave.model.processing.normalize
    kwargs:
      value: !ref image
      mean: [0.1087, 0.1342, 0.1433, 0.2734, 0.1958, 0.1363]
      std: [0.2248, 0.2179, 0.2178, 0.1850, 0.1242, 0.1049]
      dim: -3
```

The normalization uses
[TerraTorch's Prithvi EO 2 statistics](https://github.com/torchgeo/terratorch/blob/main/terratorch/models/backbones/prithvi_vit.py),
divided by 10,000 for fractional reflectance. It assumes `mask_and_scale`
decodes the stored pixels to those units using their scale/offset metadata.
Check the source's units and metadata before using these values.

`bands` configures the encoder's input channels and pretrained weight slices;
it does not select or reorder raster variables. `variables` performs that
selection. Their orders must describe the same physical bands. `NIR_NARROW`
maps to Sentinel-2 B8A; B08 is `NIR_BROAD`. A different or misspelled band name
gets a random patch-embedding slice, so check names before building the model.

All selected variables must already share a spatial grid. Loading or warping
them onto that grid is explicit; this spec does not resample bands to align them.

Given one already cut 224 × 224 Dataset `scene`, with these variables, a length-one
acquisition `time` dimension, and a geospatial grid, a direct inference call is:

```python
from geosave.model.spec import ModelSpec

spec = ModelSpec.load("model_spec.yaml")
values = spec.encode(spec.preprocess({"optical": scene}))
bound = spec.model_inputs(values)
inputs = {}
for name, value in bound.items():
    inputs[name] = value.unsqueeze(0)
inputs = spec.transform(inputs, batched=True)
inputs["image"] = torch.nan_to_num(inputs["image"])
with torch.inference_mode():
    logits = chain(**inputs)
```

Encoding derives pixels and context from the same prepared chip. Missing time
or grid metadata raises. Tensor conversion reads pixel values; `transform`
applies the same normalization as training. Track invalid pixels separately when
interpreting predictions; replacing NaN with zero only makes a tensor usable by
the network. Include any such replacement in your actual shared recipe when it
is part of the model's processing contract.
