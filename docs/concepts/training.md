# Training contracts

Geospatial workspace recipes are paused while ModelSpec is redesigned.
The working [Lightning boilerplate](../guides/workspaces.md) and caller-owned
Datasets do not require it. The method contracts below describe template code;
they are not a claim that each generated geospatial workflow currently runs.

Start with [a complete offline segmentation run](../guides/training.md). This
page explains what changes when you connect real rasters, add transforms, or
choose another training method.

## Feed the project's DataModule

The Dataset reads one sample from your source. Its DataModule batches samples
and prepares tensors on the device; the Module owns loss and metrics. Lightning
runs the training loop. The generated DataModules do not discover files, split
geographic regions, run raster preprocessing, or construct chips. Use
[Chips](../guides/chips.md) for raster windows and
[explicit preparation functions](../guides/model-inputs.md) for model inputs.

Each generated method owns its Module, DataModule, sample dictionary type and
Dataset protocol under `modules.<task>.<method>`. Generate it with
`geosave create demo --task TASK --method METHOD`, then work inside `demo`. Pass torch Datasets for `train` and
`val`; supervised pairs also accept `test`. A test loader requires its Dataset.

## Choose the target for the task

| Task and method | One sample's `target` | Model output | Missing targets |
| --- | --- | --- | --- |
| `segmentation.supervised` | `int64 (y, x)` class codes | `(B, classes, y, x)` logits | `ignore_index`, default `255` |
| `classification.supervised` | Scalar `int64` class code | `(B, classes)` logits | Exclude unlabeled chips in the Dataset |
| `regression.supervised` | `float32 (variable, y, x)` physical values | `(B, variable, y, x)` values | NaN |
| `pretraining.mim` (experimental) | No target key | Reconstruction of selected pixel input | Scores the selected hidden patches |

`B` is the number of samples in a batch. Supervised samples also contain
`inputs`, a dictionary of named model-ready tensors. There is no input-validity
field: the source and selected model own handling unavailable input data.
Pixel inputs end in `(band, y, x)`; a temporal input may include preceding
axes such as `(time, band, y, x)`. The Dataset must produce the layout the
selected model expects and include its contextual inputs when required.

For segmentation and classification, the last model stage must expose an
integer `num_classes`. For regression it must expose a list named
`variables`, in output channel order. The project’s training Module discovers this
information when Lightning calls `configure_model`. Supplying a stage without
the required declaration raises an error before training.

### Dataset outputs and target availability

Each method's `SampleDataset` protocol requires `__len__()` and
`__getitem__(index: int) -> Sample`. Source datasets can remain ordinary torch
Datasets; no GeoSave Dataset wrapper is required. The protocol declares the
Python output type, while tensor shape, dtype and acceptable input values remain
the selected model's contract.

Segmentation scores supplied class labels in every split and excludes
`ignore_index`. Unknown labels must already carry that value. The Dataset,
DataModule, Module and criterion use the same ignore value; a conflicting
criterion value raises `ValueError` at construction. Geometric augmentation
assigns that value to target pixels it introduces.

Regression scores finite target values in every split. Values for all scored
variables are flattened together for the configured regression metrics, giving
aggregate scores rather than separate scores per variable. Unknown targets and
augmentation-created target pixels are NaN. The default loss is mean squared
error; metrics include `mae`, `rmse` and `r2`.

Classification scores one class target per chip. Exclude unlabeled chips in the
source Dataset; its metrics require class codes in the head's class range.

An entirely ignored segmentation target or an empty regression target selection
produces differentiable zero loss and no metric update. An entirely unscored
split has no target metrics to report. No task infers input availability,
repairs NaN/Inf inputs or filters evaluation using an input mask.

## Apply augmentation and normalization

Recipe-based DataModules are paused while ModelSpec is rebuilt. Use caller-owned
DataLoaders and explicit preparation with the same band order and normalization
at training and inference. `ImageAugmenter` remains available for synchronized
tensor geometry.

## Change loss, optimizer, and metrics

```python title="Adaptation example — inside a segmentation.supervised workspace"
from modules.segmentation.supervised.module import Module

module = Module(
    model_chain={
        "encoder": {"name": "small_cnn", "init_args": {"in_channels": 4, "input_size": 256}},
        "decoder": {"name": "unet"},
        "head": {"name": "segmentation", "init_args": {"num_classes": 2}},
    },
    ignore_index=255,
    criterion={"name": "cross_entropy", "init_args": {"ignore_index": 255}},
    optimizer={"name": "adamw", "init_args": {"lr": 0.001}},
    metrics=["iou.macro", "f1.macro"],
)
```

`model_chain` is a [`build_model`](../guides/models.md#use-a-built-in-stage)
specification. Builder settings accept a registered `name` or an importable
`class_path` with constructor `init_args`. Schedulers additionally carry
Lightning settings such as `interval` and, for ReduceLROnPlateau, `monitor`.
Optimizer groups can select named model stages, for example a lower learning
rate for an encoder.

Multiclass metrics accept names such as `iou.macro`, `f1.micro`, and
`f1.per_class`. Unknown metrics and unsupported aggregation combinations raise
`ValueError`; see [metrics](../reference/metrics.md). Modules update separate
training, validation, and test metrics. Dense validation scores the sampled
chips, including pixels in overlaps when your Dataset returns overlapping
chips; it does not merge predictions to remove repeated pixels.

## Use unlabeled data

| Method | Additional Dataset | Training behavior |
| --- | --- | --- |
| `segmentation.fixmatch` | `unlabeled`, returning only `inputs` | Confident predictions on a weakly augmented view supervise a strongly augmented view. |
| `regression.mean_teacher` | `unlabeled`, returning only `inputs` | A teacher updated by an exponential moving average supervises a strongly augmented view. |

These methods extend their task's supervised pair. Their training DataModules
return labeled batches alongside weak and strong views of unlabeled batches;
validation and test retain the supervised task's contracts. Strong
augmentations must preserve geometry so a prediction at a weak-view pixel still
matches that pixel in the strong view. Photometric changes, such as noise,
are suitable; geometric strong augmentations are rejected at construction.
Mean Teacher updates its EMA once per optimizer step, including when gradients
are accumulated. FixMatch averages confidence-gated loss over all unlabeled
pixels, so its contribution increases as more pseudo-labels become confident.

### Experimental masked image modeling

`pretraining.mim` is scaffolding for developing self-supervised methods. Its
masking, reconstruction loss, and Lightning wiring have small tests; a validated
pretraining-to-downstream transfer workflow is still being developed.

Its Dataset returns only `inputs`. The Module zero-masks patches of
the input named by `reconstruct` and scores dense reconstruction with mean
absolute error on hidden pixels. Validation masks are deterministic for
the same batch position. The final regression head predicts one channel per
input band and instant. This is the current reconstruction contract; a complete
MAE architecture and fine-tuning recipe are not supplied by selecting a head.
See the [task API](../reference/training.md) when experimenting with this method.

## Time and place of each sample

A sample carries its grid and dates as `DataTree.gs.to_tensor` writes them:
`transform`, `crs` and, when dated, `time`. The template Module passes them to
the model with the pixels; a validation chip's `transform` is cut to the chip
by `TensorChipper`. An encoder pretrained on location or dates, such as
Prithvi or Clay, derives its own inputs from them; other encoders ignore them.

Geometric augmentation moves pixels but not `transform`, so a training
sample's place is the one it was read at.

## Calibrate segmentation confidence

Per-class confidence thresholds are part of the release's postprocessor
configuration, not the trained model. After training, run
`calibrate_thresholds` on the best checkpoint's validation predictions and put
its result in the postprocessor's `init_args`. Calibration is in progress.

`DensePredictionLogger` remains available for label and argmax panels to
TensorBoard or MLflow. See [training tools](../reference/training-tools.md).
