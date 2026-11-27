# Release and run a model

Install `geosave[ml]` to save and reload a trained chain.
Keep [data, experiments and releases in separate repositories](../mlops/index.md).

## Save and reload a chain

This small example uses random weights and needs no model download.
For a trained model, use its registered chain, such as `module.model`.

```python
from pathlib import Path

from geosave.ml.torch import ModelChain

config = {
    "encoder": {"name": "small_cnn", "init_args": {"in_channels": 2, "input_size": 64}},
    "decoder": {"name": "unet"},
    "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
}
chain = ModelChain.from_config(config).eval()
chain.save_pretrained("release")
model = ModelChain.from_pretrained("release").eval()

sorted(path.name for path in Path("release").iterdir())
```

```text
['README.md', 'config.json', 'model.safetensors']
```

The folder holds `config.json`, `model.safetensors` and a model card
(`README.md`, with `library_name: geosave`). The configuration is the chain's
construction recipe, every constructor argument written out. A Lightning
checkpoint also holds training state; this release does not include the
optimizer, DataModule or callbacks.

Build the chain with `ModelChain.from_config(config)` to release it: a chain
assembled from instances has no construction recipe. A `class_path` module
loads wherever its module is importable. Reloading restores the release's
weights without fetching upstream weights again.

## Prepare inputs for a released model

Use a Python processor to prepare source rasters for the model. The trained
weights can be shared now. Packaging model-specific preprocessing and bounded
prediction is an open design decision. [Processor data flow](../contributing/design/release-inference.md).

## Publish to Hugging Face

Replace the repository name and provide a Hub token with write access.

<!-- CREDENTIALS: Hugging Face token with repository write access -->
```python title="Requires a Hub token — replace the repository name"
commit = chain.push_to_hub("your-account/your-model")
reloaded = ModelChain.from_pretrained("your-account/your-model", revision=commit.oid)
```

Pin that commit, or a tag, for deployment. The
[workspace publication script](training.md#save-and-share-your-work) tags each
release (`--version v1.0`) and writes the base model, training data and
training commit into the model card; add evaluation and input/output contracts
yourself. Keep one model per repo and release new versions as tags.

Missing or unexpected weight keys and incompatible shapes fail on load.
