# Model composition

`model` owns the native model graph, input specification, and release artifacts.
Lightning training lives in `ml`. See the
[architecture guide](../../../docs/contributing/architecture.md) for package ownership.

```python
class Encoder(nn.Module):
    feature_channels: Published[int]

    @chain_step(outputs=("features",))
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

features = encoder(image)                 # Ordinary PyTorch call.
model = ModelChain(encoder=encoder, head=head)
logits = model(image=image)                # Named routing through selected steps.
```

`chain_step` attaches a `Step` declaration and returns the original method.
Direct calls keep their signatures and return values. The chain selects one
method per stage, supplies its inputs, and checks its result. A step declared
on `forward` runs through the module's normal PyTorch call hooks. Other named
methods run directly.

Without `outputs`, names are inferred from a single `return variable` or
`return first, second` statement. Supply `outputs` for expressions, branches,
or methods without readable source. Types always come from annotations;
container checks validate the outer type. Defaulted parameters are optional
inputs. A `head=True` method returns a terminal native result, validated against
its annotated outer type. One terminal result is returned unchanged; multiple
results are returned under their stage names. Intermediate named outputs remain
internal when terminals exist; encoder-only chains return named outputs.

```python
class Detector(nn.Module):
    @chain_step(head=True)
    def forward(self, image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return self.detector(list(image))

# Heads may return different native containers in the same model call.
model = ModelChain(features=encoder, cover=segmentation, objects=detector)
outputs = model(image=image)
logits = outputs["cover"]
objects = outputs["objects"]
```

The detector in this example already decodes its outputs into tile-local boxes.
Terminal tuples and dictionaries stay whole; they are not unpacked into chain
features.

`Published[T]` marks constructor attributes available to later stages.
Explicit `init_args` win. Missing arguments may use one matching published
attribute; ambiguous sources and incompatible annotations raise errors.

| File | Responsibility |
| --- | --- |
| `chain/step.py` | Method declarations and runtime value checks |
| `chain/routing.py` | Method selection and dependency ordering |
| `chain/published.py` | Constructor attribute declarations and argument wiring |
| `chain/chain.py` | Registered modules and execution |
| `registry.py` | Factory registration, construction, and saved arguments |
| `release/` | Native release saving, loading, and Hub publication |

## Split and end-to-end models

A split model is stages that declare steps: an encoder, a decoder, and a head.
An end-to-end model is one module registered under the `model` stage with no
`chain_step`; a chain holding it alone runs its `forward`, whose annotated
parameters name the inputs.

```yaml
model_chain:                      # split
  encoder: {name: dinov3}
  decoder: {name: dpt}
  head: {name: regression, init_args: {variables: [biomass]}}

model_chain:                      # end-to-end
  model: {name: ibm_granite_biomass}
```

Either way the training method receives one `ModelChain`, calls it with the
spec's named inputs, and gets one tensor back. The module that produces the
result names what it predicts: `classes` for segmentation and classification
heads, `variables` for regression heads and end-to-end regressors. A release
wraps the same chain with `GeoSaveModel.from_chain`.

## Geospatial context

```yaml
# model_spec.yaml
inputs:
  image: !ref image
context:
  call: geosave.model.stages.encoder.prithvi.model_context
  kwargs: {raster: !ref image}
```

```python
rasters = windows.iloc[position].gs.to_xarray().gs.rasters
inputs = spec.model_inputs(rasters)
for name in spec.pixel_inputs:
    inputs[name] = inputs[name].gs.to_tensor()
```

`inputs` binds prepared raster pixels; `context` references rasters the same
way and encodes their time labels and grid. Training and inference use the
same declared function. Encoder-specific `model_context(raster)` functions
return dictionaries of tensors, read from the raster's `time` coordinate and
georeferenced grid; neither reads pixels.

| Encoder | Per-sample context |
| --- | --- |
| `PrithviTL` | `temporal_coords[T, 2]`: year and zero-based day of year; `location_coords[2]`: latitude, longitude |
| `Clay` | `time[2]`: ISO week and hour; `latlon[2]`: latitude, longitude |

Missing metadata raises an error. For Clay, cut frames of one instant. Wavelengths and GSD retain explicit encoder constructor defaults; a
CRS resolution is not assumed to be a distance in metres. Pixel normalization
remains separate from context encoding.

Explicit caching uses the same recipe and native Python arrays or numbers:

```python
cached = spec.model_context(rasters)
inputs = spec.model_inputs(rasters, context=cached)
```

Keep cached context associated with the same sample metadata and encoding recipe.
If a crop changes the actual input window, update its metadata before encoding
context. There is no automatic cache lookup or hidden encoder selection.

## Saving and loading native releases

```python
import geosave.model.release  # registers GeoSaveModel with AutoModel
from transformers import AutoModel
from geosave.model.registry import build_model
from geosave.model.release import GeoSaveModel
from geosave.model.spec import ModelSpec

stages = {
    "encoder": {
        "name": "prithvi_tl",
        "init_args": {"pretrained": False, "num_frames": 2},
    },
}
model = build_model(stages)
spec = ModelSpec.load("configs/model_spec.yaml")
GeoSaveModel.from_chain(model, spec=spec).save_pretrained("artifacts/model")
restored = AutoModel.from_pretrained("artifacts/model").chain
spec = ModelSpec.load("artifacts/model/model_spec.yaml")
```

`GeoSaveModel` is a Transformers `PreTrainedModel` around the chain, so
`save_pretrained`, `push_to_hub`, and `AutoModel.from_pretrained` work as
for any Transformers model. It writes `config.json`, `model.safetensors`, and,
when given, `model_spec.yaml`. The repository contains no executable Python.
Generated configuration records stage order, registered selectors, constructor
defaults, resolved `Published` arguments, and the exact GeoSave version needed
to rebuild the graph. Saved `pretrained` flags are false because exported
weights supply the model state. Loading rejects checkpoints whose keys or
shapes differ from the configured chain.

Published stages must use registered `name` selectors implemented by the
installed GeoSave version. Workspace `class_path` stages remain valid for
experimentation, but must be promoted into the library before release. An
encoder's context functions live in its installed module, and the released
spec's `context` declares the recipe. Coordinates for a particular image are extracted
after loading, not stored in the artifact.

## Releasing a trained model

```python
# In a generated segmentation/supervised workspace:
from modules.segmentation import supervised

# Restore the selected checkpoint, not whichever epoch remains in memory.
task = supervised.Module.load_from_checkpoint(checkpoint)
release = GeoSaveModel.from_chain(task.model, spec=spec)
commit = release.push_to_hub("your-account/your-model")
```

`push_to_hub` uploads the weights, configuration, spec, and a model card in
one commit. Install the exact GeoSave version the configuration names, and pin
the commit when loading a production deployment:

```python
model = AutoModel.from_pretrained("your-account/your-model", revision=commit.oid).eval()
spec = ModelSpec.load(f"hf://your-account/your-model@{commit.oid}/model_spec.yaml")
output = model(**model_inputs)
```

The spec loads through `hf://` without downloading weights. Lightning
checkpoints stay training artifacts: a generated workspace's
`scripts/segmentation/supervised/publish.py` uploads them with the workspace code under `workspace/`.

### Reusing a backbone

The loaded chain is plain PyTorch, so a downstream model takes a stage as a
built module:

```python
encoder = AutoModel.from_pretrained("your-account/gfm", revision=commit.oid).chain.encoder
model = build_model(
    {
        "encoder": encoder,
        "decoder": {"name": "unet"},
        "head": {"name": "segmentation", "init_args": {"num_classes": len(classes)}},
    }
)
```

A chain built from a module carries no construction recipe for that stage, so
it cannot be published until the stage is given as a specification.
