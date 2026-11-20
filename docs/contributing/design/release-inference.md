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

The contract between sub-processors is the roles' signatures. Within one
pipeline every handoff has one type; what varies by task, the predictor's
`Output` and the postprocessor's `Product`, is a type parameter, so a type
checker refuses a predictor and postprocessor that disagree.

```python title="A task states its types in the class header"
class DensePredictor(RasterPredictor[xr.DataArray]):
    def __call__(self, model: nn.Module, sample: xr.DataTree) -> xr.DataArray: ...

class SegmentationPostprocessor(RasterPostprocessor[xr.DataArray, xr.Dataset]):
    def __call__(self, output: xr.DataArray, sample: xr.DataTree) -> xr.Dataset: ...
```

```python title="RasterPipeline.predict"
sample = self.preprocessor(rasters)
output = self.predictor(self.model, sample)
product = self.postprocessor(output, sample)
```

A new input shape gets its own pipeline, never an optional argument. Each
declares only the roles whose input differs and reuses the rest:

```text
RasterPipeline.predict(rasters)            RasterPreprocessor  RasterPredictor  RasterPostprocessor
ChangePipeline.predict(before, after)      ChangePreprocessor  RasterPredictor  RasterPostprocessor
PromptPipeline.predict(rasters, prompts)   RasterPreprocessor  PromptPredictor  PromptPostprocessor
```

A product is a raster Dataset or a GeoDataFrame. `predict_to_zarr` is
declared for raster products only; a predictor that cuts no chips leaves
`measure_chips` out and predicts in memory only. Raw rasters are not
validated: the release's model card states the rasters a model takes.

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
#     "ingestor": null, "pipeline": "raster"
model = AutoModel.from_pretrained("release")   # the model alone, a ChainModel
pipeline = load_pipeline("release")            # pipeline.model is the native ModelChain, in eval mode
```

`ARTIFACT_FORMAT_VERSION` is 5; only a raster pipeline is released so far. Loading rejects other format versions and warns
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
