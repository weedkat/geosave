# Releases and pipelines

Save and load the native model chain with Transformers, optionally with the
pipeline that runs it over rasters. Lightning checkpoints remain training
artifacts. See [Save and publish a model](../advanced/release.md).

## Releases

::: geosave.ml.release

## Pipelines

A pipeline orchestrates inference over what a caller has; there is one
pipeline per input shape. It holds a model and its sub-processors:
preprocessor, predictor, postprocessor and an optional ingestor. Each
sub-processor is called directly, and the roles' signatures are the contract
between steps: within one pipeline every handoff has one type. The
predictor's output and the postprocessor's product are type parameters a task
fills in, so a type checker refuses a predictor and postprocessor that
disagree. Each step is a package holding its abstract role and the library's
implementations; `dense` covers per-pixel segmentation and regression. A
release stores a library sub-processor by name and a project's own by class
path; only a raster pipeline can be released so far.

```text
RasterPipeline.predict(rasters)            one acquisition, any sensors
ChangePipeline.predict(before, after)      two acquisitions of one place
PromptPipeline.predict(rasters, prompts)   rasters and the shapes to look at
```

::: geosave.ml.processing.raster.pipeline

::: geosave.ml.processing.raster.ingestion.base

::: geosave.ml.processing.raster.ingestion.sentinel2.Sentinel2Ingestor

::: geosave.ml.processing.raster.preprocessing.base

::: geosave.ml.processing.raster.preprocessing.dense.DensePreprocessor

::: geosave.ml.processing.raster.prediction.base

::: geosave.ml.processing.raster.prediction.dense.DensePredictor

::: geosave.ml.processing.raster.postprocessing.base

::: geosave.ml.processing.raster.postprocessing.dense.SegmentationPostprocessor

::: geosave.ml.processing.raster.postprocessing.dense.RegressionPostprocessor

### Change and prompt pipelines

The change and prompt pipelines are skeletons: their roles are declared, and
the library has no implementations of their own steps yet. A change pipeline
reuses the raster predictor and postprocessor roles.

::: geosave.ml.processing.change.pipeline

::: geosave.ml.processing.change.preprocessing.base

::: geosave.ml.processing.prompt.pipeline

::: geosave.ml.processing.prompt.prediction.base

::: geosave.ml.processing.prompt.postprocessing.base
