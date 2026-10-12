# Save and load a release

Install the Hub integration to use the Transformers release adapter:

```bash
pip install 'geosave[hub]'
```

## Save the model for reuse

```python
import torch

from geosave.model.registry import build_model
from geosave.model.release import GeoSaveModel
from geosave.model.spec import ModelSpec, Ref

chain = build_model({
    "head": {
        "name": "segmentation",
        "init_args": {
            "feature_channels": 2,
            "classes": ["background", "crop", "forest"],
        },
    },
})
spec = ModelSpec(
    schema_version=2,
    rasters={"features": {"variables": ["red", "nir"]}},
    inputs={"feature_map": Ref("features")},
)
release = GeoSaveModel.from_chain(chain, spec=spec).eval()
release.save_pretrained("release")
```

This small example saves a randomly initialized built-in head without downloading
weights. For a trained model, pass its chain, such as `task.model`, instead.
Wrapping shares the existing modules and weights; it does not rebuild them.

Every stage must have been constructed by `build_model` with a registered `name`
specification. A directly composed `ModelChain`, a prebuilt module instance, or
a `class_path` stage cannot be exported this way. Use registered construction
from the start when you intend to release the trained model.

The local folder contains:

```text
release/
├── config.json
├── model.safetensors
└── model_spec.yaml
```

`config.json` records stage order, registered names, resolved constructor
arguments, and the exact GeoSave version. `model.safetensors` stores the weights.
The optional `model_spec.yaml` stores the processing contract. Larger models
can produce multiple weight shards through Transformers' saving options.

A Lightning checkpoint is for resuming training. A release holds the model
chain, without the optimizer, scheduler, DataModule, metrics, or callbacks.

## Load and call the model

```python
import geosave.model.release
from transformers import AutoModel

model = AutoModel.from_pretrained("release", local_files_only=True).eval()
spec = ModelSpec.load("release/model_spec.yaml")

with torch.inference_mode():
    logits = model(feature_map=torch.zeros(1, 2, 16, 16))
logits.shape
```

```text
torch.Size([1, 3, 16, 16])
```

Importing `geosave.model.release` registers the GeoSave model type with
Transformers. Loading reconstructs the chain from its configuration and restores
the weights. `model.chain` is the native `ModelChain`.

Load the spec separately: `AutoModel.from_pretrained` does not automatically
load `model_spec.yaml` or execute input preparation. Prepare rasters, convert
their pixels to tensors, and supply the chain's named inputs as shown in
[model inputs](model-spec.md).

## Publish to Hugging Face

This example needs a Hugging Face token with write access to the named
repository. Replace the repository name with your own.

<!-- CREDENTIALS: Hugging Face token with repository write access -->
```python
commit = release.push_to_hub("your-account/your-model")
model = AutoModel.from_pretrained(
    "your-account/your-model", revision=commit.oid
).eval()
spec = ModelSpec.load(
    f"hf://your-account/your-model@{commit.oid}/model_spec.yaml"
)
```

The push saves the model files, optional spec, and model card in one commit.
Pin the returned commit hash for reproducible deployment. Loading the spec
through `hf://` reads the YAML without downloading model weights.

## What a loader needs

- Install the exact GeoSave version recorded in the release configuration.
  Loading rejects a different version or an unsupported artifact format.
- Make every registered stage factory available. Built-in factories ship with
  GeoSave; custom factories need their Python package installed and their
  registration module imported before loading. Registration does not upload
  executable Python with the release.
- Preserve the saved input preparation and class order. The wrapper delegates
  to the chain; it does not normalize inputs, turn logits into labels, or attach
  geospatial metadata to predictions.

Loading rejects missing or unexpected weight keys; incompatible tensor shapes
also fail under the default loading behavior. Task-owned calibration buffers,
such as thresholds on a LightningModule, are outside a chain-only release.

See the [release reference](../reference/model-release.md) for configuration and
adapter methods.
