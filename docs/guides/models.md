# Build and use a model

```bash
pip install 'geosave[ml]'
```

Start with an ordinary PyTorch module, then add named stages when you need to
reuse an encoder or swap a decoder or head. This guide uses random weights and
runs without downloads. For published encoder weights, follow
[Build from pretrained weights](pretrained.md).

For a real-image learning project, follow the
[model composition notebook](../examples.md), which builds on the same Dataset
and training loop as the basics notebook. Use the
[encoder comparison](pretrained.md#choose-an-encoder) when choosing a built-in
model for your data.

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

Inputs use `(batch, band, y, x)` here. The model does not read raster metadata,
mask nodata, reorder bands, or normalize values. Those belong to the
[input preparation](model-spec.md) you run before the tensor call.

GeoSave models are ordinary PyTorch modules. You can call them directly, move
them to a device, and put them inside a LightningModule.

## Compose named stages

When stages need more than a single unnamed tensor, `ModelChain` connects them
through named values. An encoder extracts features; a head turns features into
outputs.

```python
from geosave.model.chain import ModelChain, chain_step


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

Intermediate tensors retain their gradients. You can train the whole chain
with a normal loss and `loss.backward()`; routing does not detach features.

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

## Use a built-in stage

GeoSave ships built-in stages, selected by a short `name`. Your own stages are
selected by `class_path` or by a name you register, as the next sections show.
A release requires stages selected by a registered `name`. Since an artifact
does not bundle Python code, the supported path is contributing the
implementation to GeoSave before publishing it.

`build_model` constructs stages from dictionaries suitable for a Python or YAML
configuration. Start with a built-in head to see this without downloading an
encoder checkpoint.

```python
from geosave.model.registry import build_model

head = build_model({
    "head": {
        "name": "segmentation",
        "init_args": {
            "feature_channels": 8,
            "num_classes": 3,
        },
    },
})
head(feature_map=torch.zeros(1, 8, 16, 16)).shape
```

```text
torch.Size([1, 3, 16, 16])
```

`name` selects a built-in within its stage; names are case insensitive.
`init_args` are constructor arguments. The segmentation head maps decoded
features to one logit per class; class names stay with the label's legend.

| Stage | Built-in names | Role |
| --- | --- | --- |
| `encoder` | `clay`, `dinov3`, `prithvi`, `prithvi_tl` | Extract image features |
| `decoder` | `dpt`, `unet` | Fuse a feature pyramid into a dense feature map |
| `head` | `classification`, `regression`, `segmentation` | Produce task outputs |
| `model` | `ibm_granite_biomass` | Run an end-to-end model |

`list_models()` lists them as a table: each model's `forward` call, what it
`publishes` for later stages, and which arguments it `requires` from you.
`list_models("unet")` lists what can run before and after one model. Stage
contracts matter: classification reads an encoder `pyramid`; dense heads read
`feature_map`. See [built-in stages](../reference/model-stages.md) for
their inputs and constructor arguments. Encoder defaults may download pretrained
weights and may require access to the checkpoint repository.

## Share constructor information

`Published` lets an earlier stage supply configuration, such as its channel
width, to a later stage's constructor.

```python
from geosave.model.chain import Published


class SizedEncoder(Encoder):
    feature_channels: Published[int]

    def __init__(self, in_channels: int = 2, width: int = 8):
        super().__init__(in_channels=in_channels, width=width)
        self.feature_channels = width


chain = build_model({
    "encoder": SizedEncoder,
    "head": {
        "name": "segmentation",
        "init_args": {"num_classes": 3},
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

## Select your own stage by class path

Once the stage works as a normal module, a configuration selects it by
`class_path`: the module that defines it and the class name.

```python
chain = build_model({
    "encoder": {"class_path": "my_project.models.SizedEncoder", "init_args": {"width": 8}},
    "head": {
        "name": "segmentation",
        "init_args": {"num_classes": 3},
    },
})
```

Rebuilding the chain imports `my_project.models`, so it must be importable in
every process that builds or loads the model. A class defined in a notebook is
`__main__.SizedEncoder`. `build_model` also accepts module classes and already
constructed modules for exploration.

To select your stage by a short `name`, as the built-ins are, register it:

```python
from geosave.model.registry import register_model


@register_model("encoder", "sized")
class SizedEncoder(Encoder):
    feature_channels: Published[int]

    def __init__(self, in_channels: int = 2, width: int = 8):
        super().__init__(in_channels=in_channels, width=width)
        self.feature_channels = width


chain = build_model({
    "encoder": {"name": "sized", "init_args": {"width": 8}},
    "head": {
        "name": "segmentation",
        "init_args": {"num_classes": 3},
    },
})
```

Import the module that registers it before building in another process;
registration names a class, it does not package your code. A
[release](release.md) requires stages selected by a registered `name`.

Continue with [model inputs](model-spec.md) to turn raster bands into the
`image` tensor, then [training](training.md) to optimize the weights. To reuse
a published encoder with a task-specific head, see [pretrained weights](pretrained.md).
