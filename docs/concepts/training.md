# Training contracts

Start with [a complete offline segmentation run](../guides/training.md). This
page explains what changes when you connect real rasters, add transforms, or
choose another training method.

## Feed the task's DataModule

The Dataset reads one sample from your source. Its DataModule batches samples
and prepares tensors on the device; the Module owns loss and metrics. Lightning
runs the training loop. The library DataModules do not discover files, split
geographic regions, run raster preprocessing, or construct chips. Use
[catalogs and cuts](../guides/catalogs-and-cuts.md) for raster windows and
[ModelSpec](../guides/model-spec.md) for raster requirements and preprocessing.

Each method exports its Module, DataModule, sample dictionary type and Dataset protocol under
`geosave.ml.tasks.<task>.<method>`. Pass torch Datasets for `train` and
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
`variables`, in output channel order. The training Module discovers this
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

```python
# Add these options when constructing a segmentation DataModule.
data = DataModule(
    spec="model_spec.yaml",
    train=train_dataset,
    val=val_dataset,
    augmentations=[
        {"name": "RandomHorizontalFlip", "init_args": {"p": 0.5}},
    ],
    cutmix=0.2,
    ignore_index=255,
    batch_size=8,
)
```

This fragment assumes the segmentation `DataModule` import and your two
Datasets. Training augmentation moves the pixel inputs and segmentation labels
together. `cutmix` optionally pastes a rectangular region from another sample
into both the image and target. Regression also moves pixel inputs and targets
together; classification augments the image and keeps the chip label.

Declare normalization in the model spec so it runs during evaluation too:

```yaml
# Add to the two-band model_spec.yaml from the training guide.
transforms:
  image:
    - name: Normalize
      init_args:
        mean: [0.5, 0.5]
        std: [0.25, 0.25]
```

In `on_after_batch_transfer`, the DataModule applies training augmentation,
optional CutMix, then the spec's input transforms in every split. `ImageAugmenter`
owns synchronized tensor geometry; `InputTransform` applies the named spec
transforms; `CutMix` pastes the same region into images and targets. Context
inputs pass through. Input values are not repaired automatically.

For temporal inputs, geometry is shared across all instants of each chip and
normalization applies the same per-band statistics to every instant. Per-input
spec transforms must preserve geometry; place spatial transforms in the joint
training augmentation. Choose statistics in the input's band order.

For direct use outside a DataModule, `ImageAugmenter` constructs a Kornia
pipeline. Supply `data_keys` when moving several tensors together:

```python
import torch
from geosave.ml.transforms import ImageAugmenter

augment = ImageAugmenter(
    [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}], size=16
)
image = torch.rand(1, 2, 16, 16)
mask = torch.zeros(1, 1, 16, 16)
image, mask = augment(image, mask, data_keys=["input", "mask"])
tuple(image.shape), tuple(mask.shape)
```

```text
((1, 2, 16, 16), (1, 1, 16, 16))
```

## Change loss, optimizer, and metrics

```python
module = Module(
    model_chain={"model": {"name": "tiny_segmentation"}},
    ignore_index=255,
    criterion={"name": "cross_entropy", "init_args": {"ignore_index": 255}},
    optimizer={"name": "adamw", "init_args": {"lr": 0.001}},
    metrics=["iou.macro", "f1.macro"],
)
```

This uses the registered model and segmentation `Module` from the training
guide. Builder settings accept a registered `name` or an importable
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

## Calibrate segmentation confidence

```python
from lightning import Trainer
from geosave.ml.tasks.segmentation.calibrate import ThresholdCalibrator

trainer = Trainer(
    max_epochs=20,
    callbacks=[ThresholdCalibrator(num_classes=2, ignore_index=255)],
)
```

The callback collects the segmentation Module's validation logits and labels
on the last scheduled epoch. It sweeps per-class confidence thresholds to
maximize F1 or IoU and updates `module.class_thresholds`, a buffer saved in
checkpoints. Match its class count and ignore value to the Module. It requires
a finite `max_epochs` and validation on that final epoch; sanity validation
is skipped.

The Module's forward call still returns raw logits. Its validation metrics
and a plain `argmax` prediction do not apply these thresholds automatically.
If your application needs low-confidence pixels to be unclassified, apply the
thresholds explicitly after softmax and document that output convention.
Calibration also retains final-epoch validation predictions on CPU, so its
memory use grows with the evaluated pixels.

`DensePredictionLogger` can log label and argmax panels to TensorBoard or
MLflow. It uses the segmentation validation/test output dictionaries and
requires a compatible logger. Both callbacks are optional; see
[training tools](../reference/training-tools.md).
