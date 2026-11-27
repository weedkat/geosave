# Training contracts

The [Lightning boilerplate](../advanced/training.md) runs offline. For geospatial kits, check
[method status](../reference/training.md) and the generated README for the
expected data and supported training stages.

Start with [the Lightning workspace](../advanced/training.md). This
page explains what changes when you connect real rasters, add transforms, or
choose another training method.

## Feed the project's DataModule

The Dataset reads and prepares one sample. The DataModule batches samples
and applies training transforms; the Module owns loss and metrics.
Lightning runs training. You choose geographic splits and supply each Dataset. Use
[Chips](../advanced/overlapping-chips.md) for raster windows and
[shared input preparation](../advanced/model-inputs.md) for model inputs.

Each generated method owns its Module, DataModule, sample dictionary type and
Dataset contract under `training/` in the generated workspace. Generate it with
`geosave create demo --task TASK --method METHOD`, then work inside `demo`. Pass torch Datasets for `train` and
`val`; supervised pairs also accept `test`. A test loader requires its Dataset.

## Choose the target for the task

| Task and method | One sample's `target` | Model output | Missing targets |
| --- | --- | --- | --- |
| `segmentation.supervised` | `int64 (y, x)` class IDs | `(B, classes, y, x)` logits | `ignore_index`, default `255` |
| `classification.supervised` | Scalar `int64` class ID | `(B, classes)` logits | Exclude unlabeled chips in the Dataset |
| `regression.supervised` | `float32 (variable, y, x)` physical values | `(B, variable, y, x)` values | NaN |
| `pretraining.mim` (experimental) | No target key | Reconstruction of selected pixel input | Scores the selected hidden patches |

`B` is the number of samples in a batch. The geospatial kits group model-ready
tensors in an `inputs` dictionary. Supervised Datasets return `inputs` and
`target`. The source and selected model own handling unavailable input data.
Segmentation validation and test hooks await an evaluation input contract.
Pixel inputs end in `(band, y, x)`; a temporal input may include preceding
axes such as `(time, band, y, x)`. The Dataset must produce the layout the
selected model expects and include its contextual inputs when required.

For segmentation and classification, the last module of the chain must expose an
integer `num_classes`. For regression it must expose a list named
`variables`, in output channel order. The project’s training Module discovers this
information when Lightning calls `configure_model`. Supplying a module without
the required declaration raises an error before training.

### Dataset outputs and target availability

Each method's `SampleDataset` protocol requires `__len__()` and
`__getitem__(index: int) -> Sample`. Source datasets can remain ordinary torch
Datasets; no GeoSave Dataset wrapper is required. The protocol declares the
Python output type, while tensor shape, dtype and acceptable input values remain
the selected model's contract.

Supervised segmentation scores supplied training labels and excludes
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
source Dataset; its metrics require class IDs in the head's class range.

An entirely ignored segmentation target or an empty regression target selection
produces differentiable zero loss and no metric update. An entirely unscored
split has no target metrics to report. No task infers input availability,
repairs NaN/Inf inputs or filters evaluation using an input mask.

## Apply augmentation and normalization

Shared preprocessors prepare and standardize raster values before sampling.
DataModules apply synchronized pixel/target augmentation during training.
Keep band order, units and missing-value policy consistent with inference.
`ImageAugmenter` supplies synchronized tensor geometry; it does not update
geographic context after an augmentation.

Geospatial starters put their input preparation in `processing/preprocessing.py`.
The project Dataset and inference Pipeline can use the same preparation.

## Change loss, optimizer, and metrics

```python title="Adaptation example — inside a segmentation.supervised workspace"
from training.module import Module

module = Module(
    model_chain={
        "encoder": {"name": "small_cnn", "init_args": {"in_channels": 4, "input_size": 256}},
        "decoder": {"name": "unet"},
        "head": {"name": "segmentation", "init_args": {"num_classes": 2}},
    },
    ignore_index=255,
    criterion={"name": "cross_entropy", "init_args": {"ignore_index": 255}},
    optimizer={"name": "adamw", "init_args": {"lr": 0.001}},
    lr_scheduler=None,
    metrics=["iou_macro", "f1_macro"],
)
```

The training module passes `model_chain` to `ModelChain.from_config`. Each
child specification selects a registered `name` or an importable `class_path`
and can supply constructor `init_args`. [Compose the model](../advanced/models.md#compose-the-model).
Schedulers additionally carry
Lightning settings such as `interval` and, for ReduceLROnPlateau, `monitor`.
Optimizer groups can select named chain modules, for example a lower learning
rate for an encoder.

Each task has its own metrics, selected by name: `SemanticSegmentationMetrics`
offers names such as `iou_macro`, `iou_per_class` and `accuracy_micro`; a
selected name is also its logged key. Unknown names raise `ValueError`; see
[metrics](../reference/metrics.md). The supervised segmentation
template's validation and test hooks raise `NotImplementedError` pending the
evaluation input contract. Other dense recipes evaluate supplied samples, so
overlapping evaluation chips can count
pixels repeatedly. Choose independent scenes or non-overlapping evaluation samples.

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

The segmentation Dataset converts the prepared image to pixels and requests
the context its model consumes. For example, a location-aware encoder can use
the source centroid alongside the image.

```python
pixels, context = image.gs.to_tensor(dtype="float32", context={"centroid": "centroid"})
sample = {"inputs": {"image": pixels, **context}, "target": target}
```

A key maps a model input to a reader: a name in `geosave.geodata.CONTEXT_READERS` or a
function of the raster. Requesting `"transform"` and `"crs"` adds the source
affine and EPSG code to `inputs`.
To place model output on the input's grid when the model preserves its
resolution, wrap it with odc-geo's `wrap_xr(array, image.gs.geobox)`.

Geometric augmentation moves pixels but not `transform`, so a training
sample's place is the one it was read at.

## Inspect segmentation predictions

The project postprocessor determines how scores become class IDs. If a project
calibrates thresholds, reserve validation data for that step.

`DensePredictionLogger` remains available for label and argmax panels to
TensorBoard or MLflow. See [training tools](../reference/training-tools.md).
