# ML task data implementation plan

> Completed implementation record. The execution ledger and test counts below
> describe that refactor, not the current suite totals. Current sample contracts
> and transforms are documented in [Training contracts](../../concepts/training.md).


> Execute inline with the execution and test-driven development skills; use one fresh review after integration.

Goal: make all six task pairs consume model-ready Dataset outputs without an
input-validity field, and concentrate repeated tensor mechanics in transforms.

Design: [task structure review](../audits/2026-10-13-ml-task-structure.md).

Constraints: preserve existing edits; no compatibility paths; no new dependency;
keep Sample and concrete Dataset protocols in their methods; keep DataLoader
construction local (user correction); ignore notebooks.

## Tasks

- [x] 1. Add failing behavior tests for validity-free task samples, temporal joint
  augmentation, configured input transforms without repair, and synchronized
  CutMix. Run them against current code and record failures.
- [x] 2. Deepen `ml/transforms/augmenter.py` to restore batched temporal images and
  mask shapes. Add `InputTransform` in `ml/transforms/inputs.py` with
  `forward(inputs: dict[str, Tensor]) -> dict[str, Tensor]`: transform each named
  input per instant and preserve other inputs. Add `CutMix.forward(*tensors)` in
  `mixing.py`, preserving every tensor's batch/spatial alignment.
- [x] 3. Refactor all six DataModules: method-local runtime-checkable output
  protocols, direct ImageAugmenter calls, explicit input transforms and CutMix.
  Remove validity and implicit input repairs. Preserve target unknown handling;
  require photometric strong transforms so view geometry stays shared.
- [x] 4. Refactor LightningModules and tests: remove input-validity scoring,
  preserve target-only scoring and algorithm masks; make segmentation ignore
  configuration consistent. Test zero loss for wholly unscored targets and no
  metric update. Align teacher EMA updates with optimizer steps and test
  gradient accumulation.
- [x] 5. Update templates and documentation, run task/transform/CLI/workspace
  tests, Ruff, scoped type checks and repository tests as practical. Obtain one
  independent review of this change against the captured baseline.

## Review focus

- Temporal and multiple pixel inputs must move together, retaining context.
- Spatial-size-changing augmentation must restore correct tensor shapes.
- Unknown targets and augmentation-created targets must retain their meaning.
- Strong views must share geometry and native Kornia transform parameters across
  pixel inputs. Native per-image noise realizations may differ.
- Datasets without `valid` must work through real Trainer and LightningCLI paths.

## Execution ledger

- Approved for implementation by user on 2026-10-13.
- Baseline: `/tmp/geosave-ml-task-baseline`, captured before implementation.
- User correction: no shared DataLoader construction; native DataLoader already
  provides the abstraction. No shared task base will be introduced.
- Workspace: current dirty checkout, preserving existing edits; no commits or
  branch operations are part of this task.
- Deferred from this refactor: calibration publishing/distributed threshold
  ownership and multi-variable metric pooling. No unrelated fixes.

- RED: 6 initial tensor/data tests, 4 semi-supervised view tests, 4 loss/contract
  tests and one accumulation test failed for the intended pre-refactor behavior.
- GREEN: focused ML/CLI/workspace checks: 167 passed, 2 deselected, 1 xfailed.
- All six real Dataset-to-Trainer smoke runs passed.
- Final review found one distributed metric logging regression; two CPU ranks
  reproduced the failure before the fix. Fixed by reducing scored-target counts
  through Lightning's strategy before deciding whether every rank logs metrics.
  Empty local targets still do not update metrics. Segmentation and regression
  training/validation: 2 slow distributed tests passed (6 normal cases deselected).
- Criterion docstring's old override description was corrected.
- Final scoped BasedPyright (`ml/tasks`, `ml/transforms`): 0 errors, warnings or
  notes. Ruff on `src/geosave/ml` and `tests/ml` passed. Diff whitespace passed.
- Documentation build passed; generated API reference freshness passed.
- Default repository suite, excluding notebook tests: 1972 passed, 30 deselected,
  4 xfailed. Initial collection errors came from generated test bytecode retaining
  old checkout paths; removed only the proven stale chain-test cache files, then
  all 105 chain tests passed. A sandboxed geodata run stalled and was stopped;
  the successful broad run used local runtime/socket access.
- User correction retained: no shared DataLoader module or task base introduced.
- Native Kornia noise realizations may differ per image; shared spatial geometry
  and native augmentation gates/parameters remain the contract. No extra noise
  replay machinery was added.
- Remaining limitations: tests used CPU, not GPU; calibration ownership and
  multi-variable metric pooling remain outside this refactor.
