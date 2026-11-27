# Model chains and registry

A ModelChain is a PyTorch module that runs modules in the order written,
each reading named values that the chain's inputs or earlier modules provide.
See [Build a model](../advanced/models.md) for examples before writing custom modules.

## Composition

Pass existing module instances in calling order, for example
`ModelChain(encoder=encoder, decoder=decoder, head=head)`. The chain runs
them like straight-line code:

```text
pyramid, prefix_tokens = encoder(image)
feature_map = decoder(pyramid)
logits = head(feature_map)
```

Names connect the modules. The chain keeps one record of named values,
starting with its inputs, the first module's forward parameters. Each module
reads its parameters from the record by name, and writes what it returns
under the names in its `returns`, adding new names and replacing existing
ones. A later module can so read any input or earlier value, such as the
encoder's `centroid` in a head, and may leave values unread. The chain checks
every read against the record's type hint before it builds anything. Two
modules of one class writing the same names, such as two backbones for
optical and radar imagery, belong inside one module.

::: geosave.ml.torch.chain

## Construction

Pass the module mapping parsed from YAML or JSON to `ModelChain.from_config`.
Each key names a module, in calling order, and each value selects its
constructor with `name` or `class_path`. Supply explicit constructor
arguments under `init_args`. An omitted argument comes from the nearest
earlier module that publishes it. The connections are checked from the
classes' forward hints before any module is built.

```python
import torch

from geosave.ml.torch import ModelChain

config = {
    "encoder": {"name": "small_cnn", "init_args": {"input_size": 64}},
    "decoder": {"name": "unet"},
    "head": {"name": "segmentation", "init_args": {"num_classes": 6}},
}

model = ModelChain.from_config(config).eval()
image = torch.zeros(1, 3, 64, 64)
logits = model(image=image)
assert logits.shape == (1, 6, 64, 64)
```

## Module contract

A module is one class with one `forward`, as `ChainModule` describes:

```python title="A decoder's side of the contract"
from geosave.ml.torch.typing import FeatureMap


@MODULES.register("unet")
class Unet(nn.Module):
    returns = ("feature_map",)
    feature_channels: Published[int]

    def __init__(self, pyramid_channels: list[int], pyramid_strides: list[int]): ...

    def forward(self, pyramid: list[FeatureMap]) -> FeatureMap: ...
```

| Part | Declares | Read by |
| --- | --- | --- |
| `MODULES.register(name)` | The name in configurations and releases | `from_config` |
| `Published[...]` attributes | Construction facts, such as channel widths | Later modules' constructors |
| `forward` parameters, all required | The names it reads, with their type hints | The chain, when calling it |
| `returns` | The names of the values `forward` returns, one per member of a `tuple[...]` hint | Later modules |

A module with another input set is another class. Variants sharing a
construction are siblings of one base class, such as `Clay` and `ClayTL` of
`BaseClay`, each with its own `forward`. A `class_path` selects any
importable module without registration.

## Value types

`geosave.ml.torch.typing` holds the types of the values modules pass. Reuse
them, and these names, before defining new ones:

| Name | Type | Returned by | Read by |
| --- | --- | --- | --- |
| `pyramid` | `list[FeatureMap]` | Encoders | Decoders, `classification` |
| `prefix_tokens` | `list[PrefixTokens]` | Transformer encoders | `dpt` |
| `embedding` | `Embedding` | `dinov3_embedding` | The caller, as image features |
| `feature_map` | `FeatureMap` | Decoders | Dense heads |
| `logits` | `Tensor` | Heads | The caller |

| Chain input | Type | Read by |
| --- | --- | --- |
| `image` | `Image`, or `Image \| ImageSeries` for `prithvi` | Encoders, monoliths |
| `centroid`, `week`, `hour`, `year`, `day_of_year` | `Centroid`, `Week`, `Hour`, `Year`, `DayOfYear` | Encoders conditioned on location and time |

::: geosave.ml.torch.typing

| Published attribute | Type | Published by | Taken by |
| --- | --- | --- | --- |
| `pyramid_channels` | `list[int]` | Encoders | Decoders, `classification` |
| `pyramid_strides` | `list[int]` | Encoders | Decoders |
| `input_size` | `int \| tuple[int, int]` | Encoders | Dense heads |
| `feature_channels` | `int` | Decoders | Dense heads |
| `embedding_channels` | `int` | `dinov3_embedding` | Modules reading `embedding` |

`list_models()` lists every registered module, grouped by family (`encoder`,
`decoder`, `head`, `monolith`), with its forward, publications and required
arguments; `list_models(name)` lists the modules that can run
right before and after it.

::: geosave.ml.torch.chain.catalog.list_models

::: geosave.ml.torch.chain.registry
