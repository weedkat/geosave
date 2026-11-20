# Training configuration and utilities

Task modules and data modules use these builders and transforms. Add logging
through Trainer. See [Train a model](../advanced/training.md).

## Optimizer, scheduler, and loss registries

Configs select each one by registered name or `class_path`, with `init_args`.
The Module builds it with the registry, passing live objects first:

```python
optimizer = OPTIMIZERS.build(BuildSpec.model_validate(spec), self.parameters())
```

::: geosave.ml.builders.CRITERIA

::: geosave.ml.builders.OPTIMIZERS

::: geosave.ml.builders.SCHEDULERS

## Losses

::: geosave.ml.criterion

## Augmentation and mixing

::: geosave.ml.transforms

## Prediction logging

::: geosave.ml.callbacks.segmentation
    options:
      members:
        - DensePredictionLogger

## Threshold calibration

Thresholds are postprocessor configuration, not model state. Calibration is in
progress: `calibrate_thresholds` raises `NotImplementedError`.

::: geosave.ml.processing.raster.postprocessing.thresholds.calibrate_thresholds

## Workspace entry point

::: geosave.ml.cli.GeosaveCLI
