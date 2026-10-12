# Build and use a model

## Start with PyTorch

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(2, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 3, kernel_size=1),
)
image = torch.zeros(1, 2, 16, 16)
logits = model(image)
logits.shape
```

```text
torch.Size([1, 3, 16, 16])
```

The input has one sample, two bands, and a 16 × 16 grid. The model produces
three class scores at each pixel. For mutually exclusive classes,
`logits.argmax(dim=1)` gives a class code per pixel. These weights are randomly
initialized; train the model before using its predictions.

GeoSave models are ordinary PyTorch modules. You can call them directly, move
them to a device, and put them inside a LightningModule.

## Compose named stages

When stages need more than a single unnamed tensor, `ModelChain` connects them
through named values. An encoder extracts features; a head turns features into
outputs.

```python
from geosave_engine.model.chain import ModelChain, chain_step


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 2, width: int = 8):
        super().__init__()
        self.body = nn.Conv2d(in_channels, width, 3, padding=1)

    @chain_step(outputs=("feature_map",))
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.body(image)


class Head(nn.Module):
    def __init__(self, width: int = 8, classes: int = 3):
        super().__init__()
        self.num_classes = classes
        self.projection = nn.Conv2d(width, classes, 1)

    @chain_step(head=True)
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.projection(feature_map)


chain = ModelChain(encoder=Encoder(), head=Head())
logits = chain(image=image)
logits.shape
```

```text
torch.Size([1, 3, 16, 16])
```

The encoder produces the runtime value `feature_map`; the head asks for that
same name and declared type. The chain finds the dependencies and runs the
stages in that order. `chain_step` leaves the method's normal calling
convention intact, so `chain.encoder(image)` still returns a tensor.

`outputs=("feature_map",)` names an intermediate result. `head=True` marks a
terminal result returned to the caller. With one terminal the chain returns
that result unchanged. With several it returns a dictionary keyed by stage
name. With no terminals it returns all produced intermediate values by name.

The segmentation training Module reads `num_classes` from the final stage.

Parameters and results need type annotations. A parameter with a default is
optional. Use explicit output names for expression returns, as above;
`@chain_step()` can also infer names from a single `return feature_map`.
See the [chain reference](../reference/model-chain.md) for the complete contract.

A chain containing just one module can also run its annotated `forward`
without a decorator. Its return is terminal. For multiple modules, each stage
declares a `chain_step` method so the chain can route their values.

## Use a registered stage

`build_model` constructs stages from dictionaries suitable for a Python or YAML
configuration. Start with a built-in head to see this without downloading an
encoder checkpoint.

```python
from geosave_engine.model.registry import build_model

head = build_model({
    "head": {
        "name": "segmentation",
        "init_args": {
            "feature_channels": 8,
            "classes": ["background", "crop", "forest"],
        },
    },
})
head(feature_map=torch.zeros(1, 8, 16, 16)).shape
```

```text
torch.Size([1, 3, 16, 16])
```

`name` selects a factory within its stage; names are case insensitive.
`init_args` are constructor arguments. The segmentation head maps decoded
features to class logits, and the order of `classes` defines the class codes.

| Stage | Built-in names | Role |
| --- | --- | --- |
| `encoder` | `CLAY`, `DINOV3`, `PRITHVI`, `PRITHVI_TL` | Extract image features |
| `decoder` | `DPT`, `UNET` | Fuse a feature pyramid into a dense feature map |
| `head` | `CLASSIFICATION`, `REGRESSION`, `SEGMENTATION` | Produce task outputs |
| `model` | `IBM_GRANITE_BIOMASS` | Run an end-to-end model |

Use `list_models()` from the same registry module to see the installed names.
Stage contracts matter: classification consumes an encoder pyramid; dense heads
consume `feature_map`. See [built-in stages](../reference/model-stages.md) for
their inputs and constructor arguments. Encoder defaults may download pretrained
weights and may require access to the checkpoint repository.

## Share constructor information

`Published` lets an earlier stage supply configuration, such as its channel
width, to a later stage's constructor.

```python
from geosave_engine.model.chain import Published


class SizedEncoder(Encoder):
    feature_channels: Published[int]

    def __init__(self, in_channels: int = 2, width: int = 8):
        super().__init__(in_channels=in_channels, width=width)
        self.feature_channels = width


chain = build_model({
    "encoder": SizedEncoder,
    "head": {
        "name": "segmentation",
        "init_args": {"classes": ["background", "crop", "forest"]},
    },
})
chain(image=image).shape
```

```text
torch.Size([1, 3, 16, 16])
```

The head receives `feature_channels=8`. Matching uses the attribute name and
its declared type. Explicit `init_args` take priority. Construction follows the
order of the stage mapping, so put providers before consumers.

This is separate from runtime routing: `Published` supplies constructor values,
while `chain_step` routes values produced during a model call.

## Register your own stage

After the stage works as a normal module, give it a configured name:

```python
from geosave_engine.model.registry import register_model


@register_model("encoder", "tiny")
class TinyEncoder(SizedEncoder):
    pass


chain = build_model({
    "encoder": {"name": "tiny", "init_args": {"width": 8}},
    "head": {
        "name": "segmentation",
        "init_args": {"classes": ["background", "crop", "forest"]},
    },
})
```

Import the Python module containing the registration before rebuilding this
chain in another process. Registration records a factory name; it does not
package your Python code.

For experiments, `build_model` also accepts module classes, importable
`class_path` specifications, and already constructed modules. A
[release](release.md) requires registered-name specifications for every stage:
classes and instances are useful for exploration but do not satisfy that
export contract.

Continue with [model inputs](model-spec.md) to turn raster bands into the
`image` tensor, then [training](training.md) to optimize the weights.
