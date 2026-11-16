# Releases and pipelines

Save and load the native model chain with Transformers, optionally with the
pipeline that runs it over rasters. Lightning checkpoints remain training
artifacts. See [Save and publish a model](../advanced/release.md).

## Releases

::: geosave.ml.release

## Pipelines

A raster pipeline orchestrates inference: raster in, product out. It holds a
model and its sub-processors: preprocessor, predictor, postprocessor and an
optional ingestor. Each sub-processor is called directly, and its `__call__`
declares with `@chain_step()` the values it takes and returns by name; the
pipeline refuses sub-processors whose values do not connect. Each step is a
package holding its abstract role and the library's implementations; `dense`
covers per-pixel segmentation and regression. A release stores a library
sub-processor by name and a project's own by class path.

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
