# ML task structure review — 2026-10-13

> Audit snapshot. Findings, paths, and check counts describe the working tree
> reviewed in this record and may be superseded by later changes. Consult
> the [current guides](../../activities.md) and [API reference](../../reference/index.md)
> for supported behavior.

Status: initial review followed by the approved implementation in
[the execution plan](../plans/2026-10-13-ml-task-data.md). Findings below describe
the pre-refactor code. The user explicitly rejected the shared DataLoader
proposal; loader construction remains local to each DataModule.

## Intended contract

The user confirmed that a Dataset returns inputs suitable for the selected
model. Input validity belongs to the source/model handling mechanism, not to a
universal mask interpreted by the training tasks. Remove `valid` from task
samples and prepared batches. Do not replace it with `input_valid`, optional
validity fields, or automatic finite-pixel filtering.

Keep target availability separate: segmentation can have ignored labels and
regression can have unknown target values. Augmentation-created target pixels
need the same task-specific treatment. FixMatch confidence and MIM patch
selection are algorithm masks, not input-validity masks.

Proposed supervised segmentation contract, declared in its `data.py`:

```python
class Sample(TypedDict):
    inputs: dict[str, torch.Tensor]
    target: torch.Tensor


@runtime_checkable
class SampleDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Sample: ...
```

Each method owns its concrete sample type and dataset protocol. A source can
remain an ordinary torch Dataset; the protocol requires its output without
introducing a storage wrapper or converter. Supervised datasets return
`inputs` and `target`; unlabeled datasets return only `inputs`.

The sample docstring states dtype, shape, target unknown representation and
whether tensors have a batch axis. A Protocol checks the Python output type
statically; it does not validate tensor shape or dtype at runtime.

## Coverage

Read all six `data.py`/`module.py` pairs and their corresponding data/module
tests, plus `ml/transforms`, ML builders and metrics, segmentation calibration
and callbacks, LightningCLI, model input/transform specifications, and training
documentation/template consumers. Existing working-tree edits were used as
the baseline. Notebooks were excluded.

| Pair | DataModule responsibility | LightningModule responsibility |
| --- | --- | --- |
| `classification/supervised` | Collate chips, augment pixel inputs, preserve scalar class targets, apply configured input transforms. | Chip classification loss and metrics. Its current pixel validity is unused by scoring. |
| `segmentation/supervised` | Move pixel inputs and class labels together; handle invented labels and labeled CutMix. | Dense loss, class metrics, callback logits/labels, one consistent ignore value. |
| `regression/supervised` | Move inputs and physical target values together; preserve unknown target values. | Score known target entries consistently in every split. |
| `segmentation/fixmatch` | Compose labeled/unlabeled loaders; shared weak geometry and strong photometric views. | Pseudo-label confidence, unlabeled loss weighting and supervised evaluation. |
| `regression/mean_teacher` | Compose labeled/unlabeled loaders and aligned views. | Teacher construction, consistency, ramp-up and EMA; evaluation currently uses the student. |
| `pretraining/mim` | Collate unlabeled inputs, augmentation and configured input transforms. | Patch selection, input hiding, reconstruction and deterministic validation masking. |

## Findings

### Repeated preparation around an already-used ImageAugmenter

All six DataModules already use `ImageAugmenter`. Four DataModules independently
load the spec, construct per-input transforms, create the same DataLoaders and
reshape temporal tensors. Supervised segmentation and regression expose private
`_augment` and `_finish` methods which their semi-supervised children call with
different mask conventions. Classification combines both operations in its own
`_finish`; MIM repeats them directly in its transfer hook.

The duplication lies around the augmenter rather than in an alternative Kornia
execution implementation. Renaming these methods would leave that coupling.

### Input validity and silent repairs conflict with the chosen contract

Supervised segmentation/regression move inputs and targets but keep the original
validity geometry. Classification/MIM move the validity mask. Evaluation of dense
tasks applies it while labeled training ignores it. Both semi-supervised methods
discard strong-view validity and use only weak-view validity for consistency.

The `_finish` paths also replace nonfinite model inputs implicitly. Infinities
become extreme finite values through default `torch.nan_to_num`, even when the
pixel is excluded from a loss. These paths should disappear under the confirmed
Dataset contract, rather than be consolidated into a universal validity module.

### Method invariants are incompletely enforced

Both semi-supervised DataModules accept geometric `strong_augmentations` despite
requiring aligned weak/strong views. Each pixel input gets a separate strong
pipeline call, so geometry can differ between inputs as well as views.

Segmentation has three independently configurable ignore values: DataModule,
Module and criterion. Criterion arguments can override the Module's value while
metrics and evaluation masking still use the Module's value.

Entirely ignored segmentation targets, entirely unknown regression targets and
entirely excluded Mean Teacher consistency tensors produce NaN losses. MIM
already specifies zero when no selected pixels are scored. A target availability
policy is needed even after removing input validity, because sparse supervision
and target padding remain meaningful.

Mean Teacher updates EMA in `on_train_batch_end`, whereas its ramp-up uses
optimizer `global_step`. With gradient accumulation those are different clocks.
This was traced in code; an accumulation Trainer probe was not run.

## Proposed placement

```text
ml/
  transforms/
    augmenter.py    ImageAugmenter: Kornia construction and joint tensor geometry
    inputs.py       InputTransform: configured transforms for named pixel inputs
    mixing.py       CutMix: one sampled box applied to all supplied tensors
  tasks/
    _data.py        Shared DataLoader construction, including worker context
    <head>/<method>/
      data.py       Sample/protocol, split loaders, target semantics, hook order
      module.py     Loss, metrics, optimization and method state
```

These are ownership proposals, not implemented classes.

1. **Deepen ImageAugmenter at the existing seam.** Move repeated joint temporal
   reshape/restore into its implementation. Images in `(B, ..., C, H, W)` retain
   their leading axes and share spatial geometry across instants and named pixel
   inputs. Keep explicit `data_keys` for multiple tensors. Keep target encoding
   and decoding with the method; keep model/spec knowledge outside ImageAugmenter.
   Preserve its existing box/keypoint behavior. This concentrates the repeated
   shape knowledge and improves locality without expanding into replay/inverse
   or a new task framework.
2. **Replace `_finish` with explicit InputTransform.** Its interface is
   `transform(inputs) -> inputs`. Apply the spec's named input transforms using
   ImageAugmenter, with time folded into the batch for per-band normalization.
   Preserve context tensors. Do not infer availability or fill NaN/Inf. Input
   transforms must preserve the task's spatial alignment; random geometry belongs
   to the joint augmentation call. The implementation lives in `ml/transforms`,
   and the task chooses when it runs.
3. **Give mixing its actual implementation.** A CutMix callable samples one box
   and pastes the same donor area in every image and target. It owns broadcast
   across temporal/band axes. Segmentation opts into it; ImageAugmenter does not
   learn segmentation loss semantics. Keep mask-free samples throughout.
4. **Keep transfer hooks readable.** Training augmentation, optional mixing and
   configured input transforms appear in their actual order. Evaluation applies
   input transforms only. The DataModule delegates tensor mechanics to transforms
   and retains task-specific target handling. This preserves the existing
   augmentation-before-preprocessing behavior.
5. **Keep LightningModules concrete.** Keep supervised inheritance within each
   head type for FixMatch/Mean Teacher. Remove reads of `valid` in evaluation,
   consistency and reconstruction. Share genuinely repeated target scoring within
   a head type; keep pseudo-label confidence, teacher EMA and patch masks with
   their methods. Common optimizer builders already own optimization construction.

For ordinary segmentation tensors, a caller already uses this existing seam:

```python
image, labels = augmenter(image, labels, data_keys=["input", "mask"])
```

The proposed maintainer flow is the same for multiple rasters and temporal
inputs: call ImageAugmenter jointly, restore task target representation, apply
CutMix if configured, then call InputTransform. No inherited `_finish` contract
needs to be understood by a child method.

For unlabeled methods, weak/strong views contain only their named inputs.
Strong augmentation is photometric; shared spatial augmentation runs once
before splitting the views. FixMatch averages confidence-gated pixel loss over
the full batch/spatial domain. Mean Teacher consistency scores its full output.
MIM scores its selected patches. Any padding coverage required by a geometric
augmentation is local to that augmentation/method, not a source-validity field.

### Alternatives

Moving `_augment`, `_finish` and `_mix` unchanged to shared helper functions
would shorten files but retain their shapes/masks/ordering interface. A generic
task base class would require switches for scalar labels, dense class labels,
regression values, teacher views and reconstruction. The proposed transforms
give leverage across written callers while keeping those task differences local.

## Checks and observed results

```bash
.venv/bin/python -m pytest tests/ml/tasks tests/ml/transforms -q --disable-warnings
```

Result: **113 passed, 37 warnings in 16.78s**. This verifies the current behavior;
it does not verify the proposed refactor.

Additional CPU probes of current code:

| Probe | Observed result |
| --- | --- |
| Nonuniform validity plus deterministic horizontal flip | Segmentation/regression masks do not follow the flip; classification/MIM masks do. |
| Crop from an 8x8 chip to 4x4 | Supervised segmentation fails when combining its unchanged 8x8 validity with 4x4 inputs. |
| Strong-only horizontal flip | Both semi-supervised methods accept it and misalign the strong view. |
| Strong Normalize with zero standard deviation | Both methods retain all 128 weak-valid pixels while strong inputs reach float32 maximum magnitude. |
| Entirely ignored segmentation batch | NaN training loss. |
| Entirely unknown regression targets | NaN training loss. |
| Mean Teacher with no unlabeled pixels selected | NaN training loss. |
| Task ignore 255, criterion ignore -1, invalid evaluation pixel | `IndexError: Target 255 is out of bounds.` |
| Concrete runtime-checkable SampleDataset Protocol in LightningCLI | Configured source Dataset instantiates successfully. |
| `Dataset[Sample]` annotation in the same CLI probe | Parser raises AssertionError in the installed environment. |
| Existing ImageAugmenter with temporal images, DEM and labels | Joint flip stays aligned, restores temporal axes, and requires no input-validity argument. |

The proposed implementation needs behavioral tests through each real transfer
hook and a small Trainer/CLI run: samples without `valid`, multiple rasters,
temporal axes, context preservation, target padding, joint CutMix, aligned
weak/strong views, and dense scoring without automatic input repair. Protocol
tests need source datasets with the correct method targets, not segmentation
fixtures reused as classification/regression/unlabeled datasets.

## Breaking changes and remaining decisions

- Removing required `valid` changes Dataset, view and batch interfaces, and dense
  evaluation semantics. Update all six task pairs, direct consumers, tests,
  templates and documentation together; do not add a compatibility alias.
- Removing implicit input sanitization exposes source/model contract errors.
  That is intentional; acceptable input values remain model-specific.
- Resolve one segmentation ignore value for targets, criterion, metrics and
  augmentation padding. Conflicting configuration should fail at construction.
- Decide the behavior of wholly unscored target batches. Proposed behavior is a
  differentiable zero loss and no metric update for that batch. Metrics for an
  entirely unscored epoch still need an explicit policy.
- Align EMA updates with optimizer steps and verify gradient accumulation.
- Calibration thresholds are owned by the LightningModule but native publishing
  exports its model, and threshold sweep/distributed behavior lacks coverage.
  This existing issue is separate from the data-preparation change.

Sources checked: installed torch Dataset/DataLoader, Lightning transfer hook and
Kornia container code; [PyTorch Dataset documentation](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.Dataset)
and [Kornia augmentation containers](https://kornia.readthedocs.io/en/latest/augmentation.container.html).
