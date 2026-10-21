# Training configuration and utilities

Task modules and data modules use these builders and transforms. Add logging
and calibration callbacks through Trainer. See [Train a model](../guides/training.md).

## Optimizer, scheduler, and loss builders

::: geosave.ml.builders.CriterionSpec
    options:
      inherited_members:
        - name
        - class_path
        - init_args
        - resolve

::: geosave.ml.builders.OptimizerSpec
    options:
      inherited_members:
        - name
        - class_path
        - init_args
        - resolve

::: geosave.ml.builders.SchedulerSpec
    options:
      inherited_members:
        - name
        - class_path
        - init_args
        - resolve

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

::: geosave.ml.callbacks.calibrate
    options:
      members:
        - ThresholdCalibrator
        - softmax_argmax

## Workspace entry point

::: geosave.ml.cli.GeosaveCLI
