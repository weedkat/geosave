# Training configuration and utilities

Task modules and data modules use these tools and transforms. Add logging
through Trainer. See [Train a model](../advanced/training.md).

## Optimizer, scheduler, and loss

A task module takes its loss, optimizer and scheduler as typed parameters.
LightningCLI builds each from `class_path` and `init_args`; the optimizer and
scheduler arrive as callables that the module calls with its parameter groups
and its optimizer:

```yaml
criterion: {class_path: torch.nn.CrossEntropyLoss, init_args: {ignore_index: 255}}
optimizer: {class_path: torch.optim.AdamW, init_args: {lr: 0.001}}
param_groups: {encoder: {lr: 0.0001}}
lr_scheduler: {class_path: torch.optim.lr_scheduler.CosineAnnealingLR, init_args: {T_max: 100}}
lr_scheduler_config: {interval: epoch}
```

::: geosave.ml.torch.training.optim.configure_optimizers

## Losses

::: geosave.ml.torch.training.criterion

## Augmentation and mixing

::: geosave.ml.torch.training.transforms

## Prediction logging

::: geosave.ml.torch.training.callbacks.segmentation
    options:
      members:
        - DensePredictionLogger

## Workspace entry point

::: geosave.ml.torch.training.cli.GeosaveCLI
