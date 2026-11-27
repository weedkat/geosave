# Model releases and processors

## Releases

A `ModelChain` built with `from_config` saves and loads through the Hugging
Face Hub: `save_pretrained`, `push_to_hub` and `from_pretrained`.
[Save and publish a model](../advanced/release.md).

## Pipeline

`Pipeline` runs ingestors, a preprocessor, a predictor and a postprocessor
over a `GeoRegion`. A release holds the pipeline and its model in one folder
or Hub repository:

```python
pipeline.push_to_hub("org/landcover-s2")
pipeline = Pipeline.from_pretrained("org/landcover-s2")
model = ModelChain.from_pretrained("org/landcover-s2")   # the model alone
```

```text
org/landcover-s2/
├── pipeline.yaml        # the model's class and every stage's arguments
├── config.json          # the ModelChain recipe
├── model.safetensors
└── README.md
```

`pipeline.yaml` is `pipeline.to_config()` plus the model's class, so a
pipeline built from instances releases too, when each stage's class is
importable, keeps each constructor argument under an attribute of the same
name, and takes only values a config can hold.

::: geosave.ml.pipeline.ingestion.base.Ingestor
    options:
      members:
        - __call__

::: geosave.ml.pipeline.pipeline.Pipeline

::: geosave.ml.pipeline.ingestion.sentinel2_l2a.Sentinel2L2AIngestor

::: geosave.ml.pipeline.ingestion.sentinel1_rtc.Sentinel1RTCIngestor

::: geosave.ml.pipeline.preprocessing.preprocessor.Preprocessor

::: geosave.ml.pipeline.preprocessing.steps

::: geosave.ml.pipeline.prediction.base.Predictor

::: geosave.ml.torch.prediction.predictor.TorchPredictor

::: geosave.ml.pipeline.postprocessing.base.Postprocessor

::: geosave.ml.pipeline.postprocessing.segmentation.SegmentationPostprocessor

::: geosave.ml.pipeline.postprocessing.regression.RegressionPostprocessor
