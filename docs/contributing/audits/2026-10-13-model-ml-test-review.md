# Model, ML, CLI and flow test review — 2026-10-13

Reviewed every current Python test/fixture in `tests/model`, `tests/ml`, `tests/cli`, and `tests/flows`: 56 original files and 431 test functions. Review used the live source and preserved the dirty working tree, including the recently added task Sample contracts. Original test snapshots are under `/tmp/geosave-docs-tests-audit/tests`; immediate pre-edit copies are under `/tmp/geosave-docs-tests-audit/model-ml-before`. No library source or notebooks were edited.

## Changes and ownership

Builder tests now mirror `criterion.py`, `optimizer.py`, and `scheduler.py`; shared `BuildSpec` tests live under `model/test_factory.py`. Model spec tests now follow `call.py`, `stage.py`, `rasters.py`, `stac.py`, and `model.py`. Encoder context tests follow the two encoder owners, and the Transformers reload regression follows release ownership. Generated workspace checks follow `cli/core/workspace.py`; scaffold installation follows `cli/commands/make.py`. Model namespace checks use `test_init.py`.

`ModelSpec` acquisition, preprocessing/context, and prepare tests share `test_model.py`, with small domain fixtures in `spec/conftest.py`. The scene used by both encoder context and published-model tests lives in `model/conftest.py`. The existing cross-method Dataset-to-Trainer and two-rank unknown-target contracts stay together in `ml/tasks/test_contracts.py` because they exercise the integration of each method's module/data pair.

The local STAC HTTP fixtures moved out of root conftest into `flows/conftest.py`, their sole caller scope. They still serve real optical COGs through native STAC parsing and raster loading. Removed unused fake catalog search methods, configured `StacSource` wrappers, and an unused terrain COG: the HTTP handler only needs Collection/Item documents and every endpoint regression requests optical. The handler retains requested-collection parsing and selection. Root retains other shared fixtures and owns independent cleanup there.

## Exact deletion and consolidation evidence

- MIM `test_reconstruction_uses_the_patch_mask_without_input_validity` duplicated the 0.5 parameter case of `test_only_hidden_pixels_are_scored`: identical constructor, `(2, 3, 2, 8, 8)` ones and expected loss 1.0. Retained the parameterized 0.5 and 0.0 cases.
- `test_one_box_moves_image_and_label_together` performed mixing with `torch.where` in the test itself. The retained `test_cutmix_pastes_one_donor_box_across_temporal_inputs_and_targets` executes production `CutMix` and checks donor selection, identical geometry, temporal/channel broadcasting, images, a second input and targets.
- Segmentation `test_template_preserves_task_optimizer_configuration` duplicated `cli/core/test_workspace.py::test_generated_entrypoint_keeps_task_optimizer_ownership`. The retained test executes generated `main.py` and asserts `auto_configure_optimizers=False` plus both subclass modes.
- Context-only `test_context_recipe_round_trips` duplicated persistence covered by `test_inputs_round_trip_references_and_calls`, which checks the context `CallSpec`, raster `Ref`, input resolution, configured transforms and saved/restored equality. Both context-scope rejection tests moved intact to `test_model.py`.
- `test_invalid_reference_paths_fail` moved into the owner `test_call.py` matrix. Six overlapping invalid paths were consolidated; its two unique values (`two names`, `3a`) remain present alongside existing key-syntax cases.
- Cycle diagnostic `test_a_cycle_names_the_key_joining_each_pair_of_steps` used identical Forward/Backward classes as the retained cycle-membership test. The retained test now checks both cycle members and both key-labelled links.
- Two paired encoder context tests split by encoder ownership. Each reader choice retains identical bad-time/missing-time/missing-grid parameters; the Prithvi encoder-by-temporal matrix remains four cases. Model-context and bucket-label assertions each split into two independent cases, accounting for two additional collected cases without additional computation.
- Stage validation and runtime explicit-rebinding regressions both remain; the runtime test was renamed `test_stage_runs_explicit_rebinding` to prevent duplicate names after consolidation.

## Lean execution and stale assumptions

Classification and regression lifecycle tests use one epoch and one train/validation batch (classification also one test batch, no duplicate sanity validation). Their existing metric lifecycle and parameter-change assertions remain. Multi-step semi-supervised and accumulated-gradient EMA regressions remain unchanged.

`ModelSpec.prepare` rejects text `keep` and an existing output before reading its raw item table. Those two tests now pass an empty native GeoDataFrame instead of writing two complete COG/GeoParquet samples; actual prepare persistence and lazy-before-write tests still read and write real assets. `find_labels` path discovery uses empty files; label table caller-column/ID checks no longer write unrelated raster assets because `read_labels` validates table metadata without opening those assets.

The classification missing-class-count regression previously constructed `torch.nn.Identity`, which fails earlier on its unannotated model contract. It now constructs a valid annotated terminal without `num_classes` and requires the class-count diagnostic.

The fresh-process stage-registration regression previously parsed the last printed DataFrame line as the obsolete dictionary API and failed with `SyntaxError`. It now validates all ten `(stage, name)` identities from the current native DataFrame index, keeping fresh-process auto-registration coverage.

## Per-file coverage ledger

Counts below are original test functions, before parametrization. Every listed file was read and traced to the stated live source owner; unchanged files retain their behavior.

| Original file under tests | Functions | Live source owner under src/geosave unless noted | Current test location under tests | Evidence / decision |
| --- | ---: | --- | --- | --- |
| `model/chain/test_chain.py` | 37 | `model/chain/chain.py` | `model/chain/test_chain.py` | Combine cycle membership and key-diagnostic assertions into one identical setup; preserve native terminals and gradients. |
| `model/chain/test_published.py` | 16 | `model/chain/published.py` | `model/chain/test_published.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/chain/test_step.py` | 38 | `model/chain/step.py` | `model/chain/test_step.py` | Keep annotation/AST/invocation contracts; drop four unused pixel fixture requests. |
| `model/release/test_huggingface.py` | 12 | `model/release/huggingface.py` | `model/release/test_huggingface.py` | Keep weights/spec/card/mode/error roundtrips; receive Prithvi context release regression. |
| `model/spec/test_acquisition.py` | 5 | `model/spec/{stac,model}.py` | `model/spec/test_stac.py; test_model.py; conftest.py` | Endpoint fallback and ModelSpec selection remain separate contracts. |
| `model/spec/test_call.py` | 14 | `model/spec/call.py` | `model/spec/test_call.py` | Keep call/reference behavior; consolidate invalid path matrix with two unique cases from test_model. |
| `model/spec/test_context.py` | 3 | `model/spec/model.py` | `model/spec/test_model.py` | Context scope validation retained; context persistence covered by richer existing roundtrip. |
| `model/spec/test_cuts.py` | 8 | `model/spec/cuts.py` | `model/spec/test_cuts.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/spec/test_examples.py` | 2 | `cli/core/workspace.py; templates/workspaces/segmentation` | `cli/core/test_workspace.py` | Bundled spec roundtrip and generated walkthrough retained. |
| `model/spec/test_execution.py` | 6 | `model/spec/{stage,model}.py` | `model/spec/test_stage.py; test_model.py` | All execution/order/rebinding/no-side-effect assertions retained. |
| `model/spec/test_model.py` | 23 | `model/spec/model.py` | `model/spec/test_model.py` | Retain declarations, lazy preprocessing, context, persistence; consolidate additional model-owned tests. |
| `model/spec/test_prepare.py` | 4 | `model/spec/model.py` | `model/spec/test_model.py` | Real roundtrip and lazy-before-write retained; early validation uses empty native table. |
| `model/spec/test_rasters.py` | 16 | `model/spec/rasters.py` | `model/spec/test_rasters.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/spec/test_stac.py` | 7 | `model/spec/stac.py` | `model/spec/test_stac.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/spec/test_stage.py` | 5 | `model/spec/stage.py` | `model/spec/test_stage.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/stages/encoder/test_clay.py` | 3 | `model/stages/encoder/clay.py` | `model/stages/encoder/test_clay.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/stages/encoder/test_context.py` | 12 | `model/stages/encoder/{clay,prithvi}.py; model/release/huggingface.py` | `model/stages/encoder/test_clay.py; test_prithvi.py; model/release/test_huggingface.py` | All temporal/grid/context/reordering and release assertions retained. |
| `model/stages/encoder/test_prithvi.py` | 2 | `model/stages/encoder/prithvi.py` | `model/stages/encoder/test_prithvi.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/stages/head/test_classification.py` | 5 | `model/stages/head/classification.py` | `model/stages/head/test_classification.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/stages/head/test_regression.py` | 4 | `model/stages/head/regression.py` | `model/stages/head/test_regression.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/stages/head/test_segmentation.py` | 7 | `model/stages/head/segmentation.py` | `model/stages/head/test_segmentation.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/stages/monolith/test_ibm_granite_biomass.py` | 1 | `model/stages/monolith/ibm_granite_biomass.py` | `model/stages/monolith/test_ibm_granite_biomass.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `model/test_exports.py` | 2 | `model/__init__.py` | `model/test_init.py` | Public import identities and unknown-name errors retained. |
| `model/test_factory.py` | 1 | `model/factory.py` | `model/test_factory.py` | Keep missing-attribute behavior; add existing selector validation and registered-name tests from builder file. |
| `model/test_registry.py` | 25 | `model/registry.py` | `model/test_registry.py` | Keep registration/build/publication contracts; migrate and repair fresh-process native DataFrame registration regression. |
| `ml/builders/test_builders.py` | 10 | `ml/builders/{criterion,optimizer,scheduler}.py; model/factory.py` | `ml/builders/test_criterion.py; test_optimizer.py; test_scheduler.py; model/test_factory.py` | Move factory selector validation to its owner; keep builder behavior. |
| `ml/metrics/test_multiclass.py` | 5 | `ml/metrics/multiclass.py` | `ml/metrics/test_multiclass.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/metrics/test_regression.py` | 2 | `ml/metrics/regression.py` | `ml/metrics/test_regression.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/classification/supervised/test_data.py` | 1 | `ml/tasks/classification/supervised/data.py` | `ml/tasks/classification/supervised/test_data.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/classification/supervised/test_module.py` | 3 | `ml/tasks/classification/supervised/module.py` | `ml/tasks/classification/supervised/test_module.py` | Repair misleading class-count stub; one train/validation/test batch retains metric lifecycle. |
| `ml/tasks/pretraining/mim/test_data.py` | 2 | `ml/tasks/pretraining/mim/data.py` | `ml/tasks/pretraining/mim/test_data.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/pretraining/mim/test_module.py` | 7 | `ml/tasks/pretraining/mim/module.py` | `ml/tasks/pretraining/mim/test_module.py` | Remove exact repeated mask-loss case; retain parameterized no-mask/partial-mask behavior and patch/temporal/lifecycle regressions. |
| `ml/tasks/regression/mean_teacher/test_data.py` | 4 | `ml/tasks/regression/mean_teacher/data.py` | `ml/tasks/regression/mean_teacher/test_data.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/regression/mean_teacher/test_module.py` | 6 | `ml/tasks/regression/mean_teacher/module.py` | `ml/tasks/regression/mean_teacher/test_module.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/regression/supervised/test_data.py` | 2 | `ml/tasks/regression/supervised/data.py` | `ml/tasks/regression/supervised/test_data.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/regression/supervised/test_module.py` | 6 | `ml/tasks/regression/supervised/module.py` | `ml/tasks/regression/supervised/test_module.py` | One train/validation batch retains optimizer movement and regression metric lifecycle; native end-to-end smoke retained. |
| `ml/tasks/segmentation/fixmatch/test_data.py` | 7 | `ml/tasks/segmentation/fixmatch/data.py` | `ml/tasks/segmentation/fixmatch/test_data.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/segmentation/fixmatch/test_module.py` | 4 | `ml/tasks/segmentation/fixmatch/module.py` | `ml/tasks/segmentation/fixmatch/test_module.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/segmentation/supervised/test_data.py` | 16 | `ml/tasks/segmentation/supervised/data.py` | `ml/tasks/segmentation/supervised/test_data.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/segmentation/supervised/test_module.py` | 25 | `ml/tasks/segmentation/supervised/module.py` | `ml/tasks/segmentation/supervised/test_module.py` | Retain training builders, class-count/ignore-index, gradients, checkpoints, CLI and scoring; template assertion covered by generated workspace test. |
| `ml/tasks/segmentation/test_calibrate.py` | 1 | `ml/tasks/segmentation/calibrate.py` | `ml/tasks/segmentation/test_calibrate.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/segmentation/test_callbacks.py` | 4 | `ml/tasks/segmentation/callbacks.py` | `ml/tasks/segmentation/test_callbacks.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/tasks/test_contracts.py` | 2 | `ml/tasks/*/*/{data,module}.py` | `ml/tasks/test_contracts.py` | Cross-method Dataset-to-Trainer integration and distributed sparse-target regressions retained. |
| `ml/test_cli.py` | 2 | `ml/cli.py` | `ml/test_cli.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/transforms/test_augmenter.py` | 13 | `ml/transforms/augmenter.py` | `ml/transforms/test_augmenter.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/transforms/test_inputs.py` | 3 | `ml/transforms/inputs.py` | `ml/transforms/test_inputs.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `ml/transforms/test_mixing.py` | 4 | `ml/transforms/mixing.py` | `ml/transforms/test_mixing.py` | Remove test-side torch.where demonstration; retain production CutMix geometry/donor/temporal/target assertions and box contracts. |
| `cli/commands/test_create.py` | 3 | `cli/commands/{create,make}.py` | `cli/commands/test_create.py; test_make.py` | Create variants/errors and make scaffold installation retained. |
| `cli/core/test_templates.py` | 1 | `cli/core/templates.py` | `cli/core/test_templates.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `cli/core/test_workspace.py` | 6 | `cli/core/workspace.py` | `cli/core/test_workspace.py` | Keep generated task/config/entrypoint checks and strict placeholder-Dataset xfail; receive walkthrough tests. |
| `flows/configs/test_anchor.py` | 3 | `geosave_flows/configs/anchor.py` | `flows/configs/test_anchor.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `flows/conftest.py` | 0 | `geosave_flows/ingest.py` | `flows/conftest.py` | Prefect harness retained; root local-STAC server fixtures moved here and simplified. |
| `flows/tasks/test_labels.py` | 7 | `geosave_flows/tasks/labels.py` | `flows/tasks/test_labels.py` | Keep directory/table/ID validations and real raster reading; remove raster writes from path/table-only checks. |
| `flows/tasks/test_sample.py` | 13 | `geosave_flows/tasks/sample.py` | `flows/tasks/test_sample.py` | Keep aligned-grid, atomic failure cleanup, encoding forwarding and GeoTIFF/Zarr/time roundtrips. |
| `flows/test_cli.py` | 5 | `geosave_flows/cli.py` | `flows/test_cli.py` | Retain current distinct behavior and error regressions; reviewed against live owner. |
| `flows/test_ingest.py` | 6 | `geosave_flows/ingest.py` | `flows/test_ingest.py` | Keep real STAC raster loading, native anchor, spec/recipe validation, existing-output rejection and Prefect lifecycle. |

## Checks

- Before edits: scoped Ruff clean; focused builder, stage and classification/regression behavior smoke: **23 passed**, 6 warnings, 20.56s. Classification lifecycle call was 4.32s.
- After moves and cleanup: focused specs/builders/encoders/releases/CLI/task/mixing/chain/labels suite: **363 passed, 2 deselected, 1 xfailed**, 14 warnings, 20.52s. Classification lifecycle call was 0.10s. Runtime measurements are diagnostic snapshots, not controlled benchmarks.
- Fresh-process registration, flow/label/Step smoke: **75 passed**, 1 warning, 26.30s, including real local STAC and Prefect flow lifecycle.
- Final simplified local-STAC fixture endpoint smoke: **5 passed, 1 slow deselected**, 1 warning, 1.76s.
- Scoped Ruff and scoped `git diff --check` passed. Integration/default and full slow suites remain coordinated by the root audit; these focused results do not claim repository-wide success.

## Remaining risks

No GeoSave production defect required a source change in this scope. The existing strict workspace xfail remains: segmentation starter data still names placeholder `datasets.MyDataset`; resolving the dataset source is product work, not stale-test deletion. Existing externally authored uncommitted changes were preserved. Fresh-process registration and distributed/Prefect tests keep their existing `slow` selection where applicable.


## Distributed slow-suite follow-up

The full local slow suite exposed an ordering dependency in the existing two-rank sparse-target regression: both cases passed in isolation (**2 passed**, 13.02s), but failed after the Prefect flow lifecycle (**1 passed, 2 failed**, 23.30s). Both Gloo ranks initialized and training reached the one-epoch limit before a worker exited with code 1; PyTorch had no saved worker exception.

A temporary diagnostic pytest plugin outside the repository wrapped `multiprocessing.process.BaseProcess._bootstrap` and captured the hidden failure. The installed Prefect queue service registers `_at_exit` through `threading._register_atexit` (`prefect/_internal/concurrency/services.py:123`). Its `os.register_at_fork` child callback resets active services, including `_done_event=None` (`services.py:84-90`), while the already registered thread shutdown callback remains inherited. During multiprocessing child teardown, `threading._shutdown` invokes `_at_exit -> drain -> _drain`, which asserts `_done_event is not None` (`services.py:241`). Python's `popen_fork` finally block then calls `os._exit` with the initial failure code and suppresses that traceback. Diagnostic evidence is in `/tmp/geosave-docs-tests-audit/distributed-worker-errors.log` and `distributed-instrumented.log`.

Changed only the regression's native `DDPStrategy` start method from `fork` to `forkserver`. Installed Lightning's multiprocessing launcher supports this method, and the Python forkserver starts workers from a separate clean server instead of inheriting the pytest process's live Prefect services and callbacks. Kept the two devices, 15-second collective timeout, one epoch, source datasets, and both train/validation finite-metric assertions unchanged. No skip, xfail, retry, assertion weakening, production patch, or diagnostic hook was added to repository code.

Validation with Prefect immediately before both distributed cases: **3 passed**, 1 parent warning, 38.16s (`distributed-forkserver-after-prefect.log`). Workers also emit existing Lightning/TorchMetrics warnings while computing synchronized metrics on the rank with no known targets; assertions remain intact. Scoped Ruff and diff whitespace checks pass. The root audit reruns the full local slow suite after this final code change.
