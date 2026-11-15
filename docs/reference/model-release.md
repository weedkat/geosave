# Releases and processors

Save and load the native model chain with Transformers, optionally with the
processor that runs it over rasters. Lightning checkpoints remain training
artifacts. See [Save and publish a model](../guides/release.md).

## Releases

::: geosave.ml.release

## Processors

A processor holds a model and four registered sub-processors: ingestor,
preprocessor, predictor and postprocessor. The library registers one of each;
a release may name only these, so deploying a model means adding its
sub-processors here.

::: geosave.ml.processing.processor

::: geosave.ml.processing.roles

::: geosave.ml.processing.ingestors.Sentinel2Ingestor

::: geosave.ml.processing.preprocessors.Reflectance

::: geosave.ml.processing.predictors.ChipPredictor

::: geosave.ml.processing.postprocessors.ClassMap
