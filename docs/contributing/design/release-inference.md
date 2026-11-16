# Release and inference

Updated against library source on 2026-11-16. Pipelines are implemented and
stored in releases; this page records that contract and what remains open.
See [samples and chips](samples-chips.md) for chip placement and merging, and
the user guide [Release and run a model](../../advanced/release.md) for a runnable
example.

## Current contract

`ml/processing/raster/` owns raster inference. Each role package has a
`base.py` with its abstract role and Registry; implementations register by name.
`RasterPipeline`, the orchestrator of inference, is built directly from its
sub-processors, without a registry of its own:

```text
ingestion/       RasterIngestor       INGESTORS       sentinel2_l2a
preprocessing/   RasterPreprocessor   PREPROCESSORS   dense
prediction/      RasterPredictor      PREDICTORS      dense
postprocessing/  RasterPostprocessor  POSTPROCESSORS segmentation, regression
pipeline.py      RasterPipeline
```

Every sub-processor is entered by `__call__`:

| Role | Call | Owns |
| --- | --- | --- |
| ingestor | `ingestor(region) -> list[DataTree]` | acquiring one source tree per observation, for inference only |
| preprocessor | `preprocessor(rasters) -> sample` | the model-ready sample: named inputs on one grid, lazy, standardized; other nodes, such as a mask, are for the postprocessor |
| predictor | `predictor(model, sample) -> output` | chipping prepared inputs, deriving requested context, running batches and merging model output; the output is the model's values, undecoded and unmasked |
| postprocessor | `postprocessor(output, sample) -> product` | decoding, masking and building the product: a raster Dataset for per-pixel tasks (with the sample's time and optional mask), a GeoDataFrame for a task whose results are shapes |
| pipeline | `predict(rasters)`, `predict_to_zarr(rasters, destination, chunks)` | running the model over raw rasters; the caller runs `pipeline.ingestor` |

The contract between sub-processors is declared, not validated by a separate
class. Each `__call__` carries `@chain_step()`: its annotated parameters name
the values it takes, its returned variable names the value it hands on, and
its docstring states that value in detail.

```python title="A sub-processor declares what it takes and returns"
class DensePredictor(RasterPredictor):
    @chain_step()
    def __call__(self, model: nn.Module, sample: xr.DataTree) -> xr.DataArray:
        ...
        return output
```

```text
pipeline supplies    rasters, model
preprocessor         rasters        -> sample
predictor            model, sample  -> output
postprocessor        output, sample -> product
```

`RasterPipeline` refuses, when constructed, a sub-processor without a declared
step, one that takes a value no earlier step returns, and one that takes a
value as another type than it is returned as. Each value's outer type is
checked again as it is handed over. Raw rasters are not validated: the
release's model card states the rasters a model takes.

The `dense` predictor serves segmentation and regression:

```text
rasters DataTree → preprocess (standardized) → sample DataTree
    → RasterChipper → to_tensor(context=...) → model(**inputs)
    → RasterMerger → merged DataArray → postprocess → product Dataset
```

A release reads each sub-processor's spec from the sub-processor itself:
`Registry.read_spec(built)` returns its registered name or class path with
every constructor argument, read from attributes of the same names. There is
no separate spec object, so the pipeline that was run is the one released.
`chunks` is an argument of `predict_to_zarr`, not stored.

```python title="What a release stores and rebuilds"
pipeline = RasterPipeline(
    chain,
    preprocessor=DensePreprocessor(bands=["B04", "B08"]),
    predictor=DensePredictor(size=256),
    postprocessor=SegmentationPostprocessor(classes, colors),
)
ChainModel.from_pipeline(pipeline).save_pretrained("release")
# config.json: stages=[...],
#     "preprocessor": {"name": "dense", "init_args": {...}},
#     "predictor": {"name": "dense", "init_args": {...}},
#     "postprocessor": {"name": "segmentation", "init_args": {...}},
#     "ingestor": null
model = AutoModel.from_pretrained("release")   # the model alone, a ChainModel
pipeline = load_pipeline("release")            # pipeline.model is the native ModelChain, in eval mode
```

`ARTIFACT_FORMAT_VERSION` is 4; loading rejects other format versions and warns
for a different GeoSave version. Missing/unexpected weight keys are rejected.
A release without a pipeline still loads as a model; `load_pipeline` raises
`ValueError` for it. A chain's components must be library-registered. A
library sub-processor is stored by name; a project's own is stored by class
path and rebuilds wherever the project's code is installed.

The ModelSpec declaration, its graph executor and the plain-function registry
sketches are superseded by these classes.

## Decoding by task

| Kind | Chip output | Merged, then | Published as |
| --- | --- | --- | --- |
| segmentation | logits | argmax | uint8, `CFFlags` from classes/colors, nodata 255 |
| regression | values | — | float32, units, NaN nodata |
| detection (future) | boxes, scores, classes | NMS across chip overlaps | GeoDataFrame |
| classification (future) | logits | — | GeoDataFrame per chip footprint |

The built-in postprocessors cover the two dense rows. Merging logits before
argmax makes overlap blending meaningful. `calibrate_thresholds` remains a
stub; thresholds and a matching decoder are future work.

## Open

| Question | Notes |
| --- | --- |
| Different statistics per input | `DensePredictor` supports named pixel inputs and requested context; each preprocessor standardizes its own nodes, so inputs can differ; the old shared `normalize` served every pixel input |
| Context after augmentation | `TENSOR_CONTEXT` reads chip context at tensor conversion; training geometry transforms do not recompute it |
| Normalization statistics | projects measure them on training data and pass them as preprocessor arguments; recording their provenance remains project-owned |
| Missing steps or sources | zero-fill with a validity input, or drop the sample; must match between training and inference |
| How a time step is filled | first, least cloudy, composite; see [sample rows](samples-chips.md#sample-rows) |
| Offline stores | project-owned optimization; no processor preload/store API |
| Whole-scene evaluation | supervised-segmentation template has tensor chipping/merging, but its flat/nested sample migration is incomplete |
| Vector outputs | detection, instances and chip classification need their own processors and merge rules |
| Memory | processor chunks limit score accumulation; decoded chunks still assemble a complete product in memory, so streaming writes remain open |

Keep data snapshots, experiments and inference artifacts in
[three repositories](../../mlops/index.md); data and training
commit links in the model card are currently written by the project.
