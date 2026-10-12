# Training configuration and utilities

Task modules and data modules use these builders and transforms. Add logging
and calibration callbacks through Trainer. See [Train a model](../guides/training.md).

## Optimizer, scheduler, and loss builders

::: geosave_engine.ml.builders.CriterionSpec
    options:
      inherited_members:
        - name
        - class_path
        - init_args
        - resolve

::: geosave_engine.ml.builders.OptimizerSpec
    options:
      inherited_members:
        - name
        - class_path
        - init_args
        - resolve

::: geosave_engine.ml.builders.SchedulerSpec
    options:
      inherited_members:
        - name
        - class_path
        - init_args
        - resolve

::: geosave_engine.ml.builders.build_criterion

::: geosave_engine.ml.builders.build_optimizer

::: geosave_engine.ml.builders.build_scheduler

## Losses

::: geosave_engine.ml.criterion

## Augmentation and mixing

::: geosave_engine.ml.transforms

## Prediction logging

::: geosave_engine.ml.tasks.segmentation.callbacks
    options:
      members:
        - DensePredictionLogger

## Threshold calibration

::: geosave_engine.ml.tasks.segmentation.calibrate
    options:
      members:
        - ThresholdCalibrator
        - softmax_argmax

## Workspace entry point

::: geosave_engine.ml.cli.GeosaveCLI
