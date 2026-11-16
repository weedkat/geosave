# Training configuration and utilities

Task modules and data modules use these builders and transforms. Add logging
through Trainer. See [Train a model](../advanced/training.md).

## Optimizer, scheduler, and loss builders

::: geosave.ml.builders.CriterionSpec
    options:
      inherited_members:
        - name
        - class_path
        - init_args

::: geosave.ml.builders.OptimizerSpec
    options:
      inherited_members:
        - name
        - class_path
        - init_args

::: geosave.ml.builders.SchedulerSpec
    options:
      inherited_members:
        - name
        - class_path
        - init_args

::: geosave.ml.builders.build_criterion

::: geosave.ml.builders.build_optimizer

::: geosave.ml.builders.build_scheduler

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
