# ml: methods, optimisation, and task layout

Goal: keep `ml/` the training driver for the current `geodata` and `model`
packages, and make room for semi-supervised and self-supervised methods and
for further AI tasks, without a shared cross-task abstraction.

Layout stays task first, then method (`ml/<task>/<method>/`). A method that
only changes the training step subclasses the task's supervised `Module`.

```text
ml/
├── data.py          # model_inputs, open_prepared, Chips, UnlabeledDataset, ChipDataModule
├── metrics.py       # MulticlassMetrics, RegressionMetrics
├── builders/        # criterion / optimizer / scheduler / build_optimization
├── criterion/       # losses the repo owns
├── transforms/      # ImageAugmenter, cutmix_boxes
├── segmentation/
│   ├── calibrate.py  callbacks.py  transforms.py
│   ├── supervised/  module.py data.py merge.py
│   ├── fixmatch/    module.py data.py
│   └── unimatch_v2/ module.py data.py
├── classification/supervised/   regression/supervised/
├── detection/supervised/        # stub
└── pretraining/mim/
```

Status 2026-10-11: Tasks 1–7 implemented. Shared pieces were lifted once a
second task repeated them: `ml/data.py` (chip datasets, `ChipDataModule`),
`ml/metrics.py`, `build_optimization`. Change detection gets no package:
segmentation with `frames: {length: 2}` covers it. Detection stays a stub
until geodata clips vector labels to chips.

## Task 1 — lift raster merging out of the supervised Module

- Create `ml/segmentation/supervised/merge.py`: `RasterMerger(num_classes,
  ignore_index)` with `add(logits, ids, valid, dataset, dataloader_idx)`
  returning completed `(logits, target)` rasters and `raster_count`.
- `Module` keeps the Lightning hooks and owns one `RasterMerger` per epoch.
- Tests: move the `_merge` tests to `tests/ml/segmentation/supervised/test_merge.py`
  and call `RasterMerger.add`; module tests keep the hook behaviour.

## Task 2 — optimizer groups by module path, frozen paths, warmup

- `OptimizerSpec.groups` keys are dotted module paths (`encoder.blocks.11`);
  the deepest path owning a parameter wins; tied parameters across unrelated
  paths still raise.
- `OptimizerSpec.frozen`: module paths whose parameters stop requiring grad
  and are left out of the optimizer.
- `SchedulerSpec.warmup`: linear warmup over that many intervals, chained
  before the configured scheduler with `SequentialLR`; refused for
  `ReduceLROnPlateau`.
- Tests in `tests/ml/builders/test_builders.py`.

## Task 3 — losses

- Register `DICE`, `FOCAL`, `LOVASZ`, `TVERSKY`, `JACCARD` from
  `segmentation_models_pytorch.losses`; declare the dependency.
- Tests: each name builds and scores multiclass logits.

## Task 4 — CutMix

- `ml/transforms/mixing.py`: `cutmix_boxes(batch, shape, ...)` returns
  boolean `(B, H, W)` box masks; the caller pastes with `torch.where`.
- Supervised `DataModule(cutmix=p)` mixes images and labels in the batch.
- Tests: box area within the scale bounds, the same box moves image and label.

## Task 5 — FixMatch and UniMatch V2 for segmentation

- `fixmatch.DataModule(supervised.DataModule)`: adds `unlabeled` catalog and
  `strong_augmentations`; training batch is
  `{"labeled": (inputs, target, ids, valid), "unlabeled": (weak, strong)}`.
- `fixmatch.Module(supervised.Module)`: confidence-thresholded pseudo-labels
  from the weak view supervise the strong view.
- `unimatch_v2.Module(supervised.Module)`: EMA teacher, two strong views with
  CutMix, complementary channel dropout on the encoder features.
- Tests: one-step fits on synthetic batches; pseudo-label masking; EMA update.

## Task 6 — masked image modelling pretraining

- `pretraining/mim/`: masks input patches and reconstructs them with the
  chain's last stage; loss on masked pixels only.

## Task 7 — task stubs, then implementations

- `classification`, `regression`, `detection`, `change` supervised pairs.
