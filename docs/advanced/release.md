# Release and run a model

Install `geosave[hub]`. A release saves a trained registered chain and,
optionally, the pipeline that prepares rasters and decodes predictions.
Keep [data, experiments and releases in separate repositories](../mlops/index.md).

## Save and reload a chain

This small example uses random weights and needs no model download.
For a trained model, use its registered chain, such as `module.model`.

```python
from pathlib import Path

from transformers import AutoModel

from geosave.ml import build_model
from geosave.ml.release import ChainModel, load_pipeline

chain = build_model({
    "encoder": {"name": "small_cnn", "init_args": {"in_channels": 2, "input_size": 64}},
    "decoder": {"name": "unet"},
    "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
}).eval()
release = ChainModel.from_chain(chain)
release.save_pretrained("release")
model = AutoModel.from_pretrained("release", local_files_only=True).eval()

sorted(path.name for path in Path("release").iterdir())
```

```text
['config.json', 'model.safetensors']
```

The folder holds `config.json` and `model.safetensors` (or weight shards).
Configuration records stage order, resolved constructor arguments and the
GeoSave version. A Lightning checkpoint also holds training state; this release
does not include the optimizer, DataModule or callbacks.

Stages must use registered `name` specs through `build_model`. Prebuilt modules
and `class_path` stages cannot be exported. Integrate custom components into
GeoSave before native release. Reloading saved pretrained stages restores the
release's weights without fetching upstream weights again.

## Run with a raster pipeline

A `RasterPipeline` runs a model over rasters. Each part is called directly and
hands its result to the next:

| Part | Call | Returns |
| --- | --- | --- |
| Preprocessor | `preprocessor(rasters)` | `sample`: named pixel-input DataTree nodes, standardized as training did. |
| Predictor | `predictor(model, sample)` | `output`: the model's merged scores on the sample's grid, undecoded. |
| Postprocessor | `postprocessor(output, sample)` | `product`: the decoded, masked product Dataset. |
| Optional ingestor | `ingestor(region)` | One source tree per acquisition, when the caller requests them. |

Use the [sample scene](../basics/first-raster.md#open-the-sample):

```python
import xarray as xr

import geosave as gs
from geosave.ml import RasterPipeline
from geosave.ml.processing.raster.postprocessing.dense import SegmentationPostprocessor
from geosave.ml.processing.raster.prediction.dense import DensePredictor
from geosave.ml.processing.raster.preprocessing.dense import DensePreprocessor

pipeline = RasterPipeline(
    chain,
    preprocessor=DensePreprocessor(bands=["red", "nir"]),
    predictor=DensePredictor(size=64, stride=48),
    postprocessor=SegmentationPostprocessor(
        classes=["water", "vegetation", "bare"],
        colors=["#419bdf", "#397d49", "#a59b8f"],
    ),
)
scene = gs.open_raster("sample-scene.tif")
rasters = xr.DataTree.from_dict({"image": scene})
product = pipeline.predict(rasters)
product["class_id"].shape
```

```text
(337, 267)
```

`image` is the node read by the preprocessor. The output keeps the source grid;
its class IDs are meaningless until the model is trained. Scores blend before
argmax.

Save the pipeline with the chain and check the reloaded prediction:

```python
ChainModel.from_pipeline(pipeline).save_pretrained("release-with-pipeline")
reloaded = load_pipeline("release-with-pipeline")
scene = gs.open_raster("sample-scene.tif")
rasters = xr.DataTree.from_dict({"image": scene})
reloaded_product = reloaded.predict(rasters)
reloaded_product.equals(product)
```

```text
True
```

The release reads each part's class and constructor arguments from the part
itself, so a part must keep every constructor argument in an attribute of the
same name. A library part is stored by its registered name and loads
automatically. A project's own part is stored by its class path and loads
wherever that project's code is installed.

`AutoModel.from_pretrained` loads the model of such a release alone;
`load_pipeline` loads the pipeline, whose `model` is the native chain.

The pipeline accepts source trees directly. If it carries an ingestor, the
caller invokes `pipeline.ingestor(region)` and predicts each returned
tree. File opening and acquisition remain explicit.

## Write large predictions to Zarr

`predict` holds whole-raster scores in memory. `predict_to_zarr` predicts and
writes one chunk at a time:

```python
scene = gs.open_raster("sample-scene.tif")
rasters = xr.DataTree.from_dict({"image": scene})
store = pipeline.predict_to_zarr(
    rasters, "product.zarr", chunks=128,
)
saved = gs.open_raster(store)
saved_grid = saved.gs.geobox
saved_grid == product.gs.geobox
```

```text
True
```

`chunks` sets the chunk predicted at once and the store's chunk size, rounded
down to a multiple of the predictor's stride; choose the largest whose model
output fits memory. Each chunk is read with a halo, the pixels of every chip
touching it, so the store holds what `predict` returns. The destination
must not exist. See [pipeline contracts](../reference/model-release.md#pipelines).

## Publish predictions

A product is a raster like any other: publish it as COGs with an index, as
[Store imagery as a catalog](../basics/catalog.md) does for imagery.

```python
from geosave.geodata import stac

item = saved.gs.publish_item("predictions", id="20250627")
predictions = stac.create_catalog([item])
list(item.assets), predictions["id"].tolist()
```

```text
(['class_id'], ['20250627'])
```

The product keeps the scene's acquisition time, so the Item is dated like the
imagery it was predicted from.

## Publish to Hugging Face

Replace the repository name and provide a Hub token with write access.

<!-- CREDENTIALS: Hugging Face token with repository write access -->
```python title="Requires a Hub token — replace the repository name"
commit = ChainModel.from_pipeline(pipeline).push_to_hub("your-account/your-model")
reloaded = load_pipeline("your-account/your-model", revision=commit.oid)
```

Pin that commit for deployment. Add the data/experiment revisions, evaluation
and input/output contracts to the model card; they are not recorded automatically.
The [workspace publication guide](training.md#save-and-share-your-work) describes
separate training archives.

Loading requires the same artifact format. A different GeoSave version warns;
missing/unexpected weight keys or incompatible shapes fail. A chain-only release
takes prepared tensors; `load_pipeline` requires a release saved with a pipeline.
