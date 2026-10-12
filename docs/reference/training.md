# Training tasks and data modules

Each method pairs a Lightning Module and DataModule with its sample contracts.
You supply torch Datasets for each split. Start with [Train a model](../guides/training.md)
and read [Training behavior](../concepts/training.md) before adapting a method.

## Supervised segmentation

::: geosave.ml.tasks.segmentation.supervised

## FixMatch segmentation

::: geosave.ml.tasks.segmentation.fixmatch.Module
    options:
      inherited_members:
        - configure_model
        - configure_optimizers
        - forward
        - validation_step
        - test_step

::: geosave.ml.tasks.segmentation.fixmatch.DataModule
    options:
      inherited_members:
        - val_dataloader
        - test_dataloader

::: geosave.ml.tasks.segmentation.fixmatch.Unlabeled

::: geosave.ml.tasks.segmentation.fixmatch.pseudo_label_loss

## Supervised regression

::: geosave.ml.tasks.regression.supervised

## Mean Teacher regression

::: geosave.ml.tasks.regression.mean_teacher.Module
    options:
      inherited_members:
        - configure_model
        - configure_optimizers
        - forward
        - validation_step
        - test_step

::: geosave.ml.tasks.regression.mean_teacher.DataModule
    options:
      inherited_members:
        - val_dataloader
        - test_dataloader

::: geosave.ml.tasks.regression.mean_teacher.Unlabeled

## Supervised classification

::: geosave.ml.tasks.classification.supervised

## Masked image modeling

::: geosave.ml.tasks.pretraining.mim
