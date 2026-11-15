# Release and run a model

Install the [Hub extra](../install.md#choose-what-you-need) on the source or
wheel installation you chose. It adds the Transformers release adapter.

The supported deployment workflow uses model implementations integrated into
GeoSave. Prototype a custom architecture in ordinary PyTorch or a workspace,
then contribute its implementation and registration to the library before
deploying it through this workflow. Releases store registered construction
recipes and weights; they do not upload executable model code.

## Save the model for reuse

```python
import torch

from geosave.ml import build_model
from geosave.ml.release import ChainModel

chain = build_model({
    "head": {
        "name": "segmentation",
        "init_args": {
            "feature_channels": 2,
            "num_classes": 3,
        },
    },
})
release = ChainModel.from_chain(chain).eval()
release.save_pretrained("release")
```

This small example saves a randomly initialized built-in head without downloading
weights. For a trained model, pass its chain, such as `task.model`, instead.
Wrapping shares the existing modules and weights; it does not rebuild them.

For a project Module whose `model` is a registered chain, save its trained chain:

```python
module.eval()
trained = ChainModel.from_chain(module.model)
trained.save_pretrained("trained-release")
```

The plain Lightning starter in [Training](training.md) uses a custom network,
so it cannot use this registered-chain export. Use its Lightning checkpoint
or build a registered architecture first.

Every stage must have been constructed by `build_model` with a registered `name`
specification. A directly composed `ModelChain`, a prebuilt module instance, or
a `class_path` stage cannot be exported this way. Use registered construction
from the start when you intend to release the trained model.

The local folder contains:

```text
release/
├── config.json
└── model.safetensors
```

`config.json` records stage order, registered names, resolved constructor
arguments, the processor when one is attached, and the exact GeoSave version. `model.safetensors` stores the weights.
Larger models can produce multiple weight shards through Transformers' saving options.

If the chain started from pretrained encoder weights, the saved construction
recipe records `pretrained=False`. Reloading constructs the architecture and
restores this release's weights without fetching the upstream encoder again.
See [pretrained weights](pretrained.md) for the difference between initializing
a backbone and loading a trained task.

A Lightning checkpoint is for resuming training. A release holds the model
chain, without the optimizer, scheduler, DataModule, metrics, or callbacks.

## Load and call the model

```python
import geosave.ml.release
from transformers import AutoModel

model = AutoModel.from_pretrained("release", local_files_only=True).eval()

with torch.inference_mode():
    logits = model(feature_map=torch.zeros(1, 2, 16, 16))
logits.shape
```

```text
torch.Size([1, 3, 16, 16])
```

Importing `geosave.ml.release` registers the GeoSave model type with
Transformers. Loading reconstructs the chain from its configuration and restores
the weights. `model.chain` is the native `ModelChain`.

Called directly, the model takes prepared tensors and returns tensors without
geographic metadata. To prepare a raster and map the output, attach a processor.

## Run the model with a processor

A processor holds the model and four registered sub-processors:

| Role | Method | Does |
| --- | --- | --- |
| ingestor | `ingest(region) -> list[Dataset]` | Loads the rasters to predict for a region. |
| preprocessor | `preprocess(raster) -> Dataset`, `normalize(image) -> Tensor` | Selects and decodes bands; standardizes each batch. The same class serves training. |
| predictor | `predict(forward, image) -> DataArray` | Runs the model over a prepared raster of any size. |
| postprocessor | `postprocess(logits) -> Dataset` | Turns merged `(band, y, x)` scores into the product. |

The processor runs `preprocess → predict → postprocess`. The built-in `chips`
predictor runs `RasterChipper → normalize → model → RasterMerger`, so a raster of any
size is predicted on its own grid. The library registers one of each role;
select them by name:

```python
from geosave.ml.processing.processor import ProcessorSpec, build_processor
from geosave.ml.release import ChainModel, load_processor

chain = build_model({
    "encoder": {"name": "small_cnn", "init_args": {"in_channels": 2, "input_size": 64}},
    "decoder": {"name": "unet"},
    "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
}).eval()
spec = ProcessorSpec(
    ingestor={"name": "sentinel2_l2a", "init_args": {"bands": ["B04", "B08"]}},
    preprocessor={"name": "sentinel2_reflectance", "init_args": {"bands": ["red", "nir"]}},
    predictor={"name": "chips", "init_args": {"size": 64, "stride": 48}},
    postprocessor={
        "name": "class_map",
        "init_args": {
            "classes": ["water", "vegetation", "bare"],
            "colors": ["#419bdf", "#397d49", "#a59b8f"],
        },
    },
)
processor = build_processor(spec, chain)
product = processor.predict_path("sample-scene.tif")
product["class_id"].shape
```

```text
(337, 267)
```

`product` is a Dataset on the [sample scene](../first-raster.md#open-the-sample)'s
grid; the random weights make its classes meaningless until trained. Chips of
64 pixels overlap by 16 and blend with the default `"hann"` window.
`predict_raster(raster)` takes an open raster and `predict_region(region)`
predicts each raster the ingestor loads.

A project may write its own sub-processor: a class with its role's method,
registered as `@PREPROCESSORS.register("my_name")` or selected by
`class_path`. It predicts locally, but a release refuses it. A release rebuilds
from `pip install geosave` alone, so a model is deployed by adding its
components and sub-processors to the library first.

Save the processor with the chain, then rebuild both in one call:

```python
ChainModel.from_processor(processor).save_pretrained("release-with-processor")
reloaded = load_processor("release-with-processor")
reloaded.predict_path("sample-scene.tif").equals(product)
```

```text
True
```

`config.json` stores the processor's name and every argument, defaults
included, as `processor.spec` shows. The release stores names, not code:
import the module that registers your ingestor, preprocessor and postprocessor
before `load_processor`. A release saved without a processor makes
`load_processor` raise `ValueError`.

## Publish to Hugging Face

This example needs a Hugging Face token with write access to the named
repository. Replace the repository name with your own.

<!-- CREDENTIALS: Hugging Face token with repository write access -->
```python
commit = release.push_to_hub("your-account/your-model")
model = AutoModel.from_pretrained(
    "your-account/your-model", revision=commit.oid
).eval()
```

The push saves model files and the model card in one commit.
Pin the returned commit hash for reproducible deployment.

For a trained release made above, publish `trained` instead of the randomly
initialized demonstration `release`.

## Three repositories

A trained project lives in three Hugging Face repositories, one audience each:

```text
your-account/<name>-data       dataset repo          raw and processed rasters, labels
your-account/<name>-training   model repo, private   workspace code, run configs,
                                                     checkpoints and TensorBoard metrics
your-account/<name>            model repo, public    config.json + model.safetensors
```

A workspace's `publish.py` writes the release folder and, given
`--release-repo` and `--training-repo`, uploads the release and archives the
run, one commit each. LitData stores are preloads tailored to one model; keep
them local or in a bucket, not in the data repository.

## What a loader needs

- `pip install geosave`. A release names only components and sub-processors
  the library registers, so it rebuilds without the project's code. To deploy
  a model, add its components and sub-processors to geosave first; releasing
  a project's own class is refused.
- The same artifact format version. A different GeoSave version loads with a
  warning; the weights must still match the rebuilt chain exactly.
- Without a processor, the model delegates to the chain; it does not
  normalize inputs, turn logits into labels, or attach geospatial metadata.

Loading rejects missing or unexpected weight keys; incompatible tensor shapes
also fail under the default loading behavior. Class thresholds belong to the
postprocessor's configuration, not to the model.

See the [release reference](../reference/model-release.md) for configuration and
adapter methods.
