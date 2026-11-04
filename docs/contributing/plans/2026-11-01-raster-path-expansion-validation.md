# Raster path expansion validation

## Changes

- Removed io/raster/layout.py, Planned, PlannedScene and layout.save.
- Added paths.expand_paths for one Dataset's destinations and lazy selections.
- GeoDataTree expands groups and detects canonical path collisions globally.
- GeoTIFF owns suffix/grid/dimension/time/fill preflight for every output.
- Scene owns grouped eager/deferred writes; existing writer-returned headers
  and Scene return values remain available.
- STAC observation validation now belongs to the Item translator.
- Moved layout tests beside their implementation owners; replaced fake-write
  and duplicate path-option checks with real Scene/pixel round-trip tests.

## Fresh verification

- Full configured suite: `.venv/bin/python -m pytest -q --tb=short`.
  **38 failed, 1365 passed, 37 deselected, 3 xfailed** in 155.34 seconds.
- Compared exact failed test names with the preceding recorded full run:
  **38 before, 38 after, no new failures and no resolved failures**.
- Before the final review fix, raster/core/STAC/catalog regressions:
  **388 passed, 11 deselected**. The final invalid-date, fill and alias-path
  preflight checks passed separately and were included in the final full run.
- Catalog guide executed end to end: **3 Items, 2 samples, readable chips**.
- Ruff source/tests/scripts and formatting checks passed; git diff --check passed.
- BasedPyright on changed raster accessors, path expansion, Scene, GeoTIFF and
  STAC Item translation: **0 errors, 0 warnings**.
- Zensical build passed; generated API freshness and rendered coverage:
  **1142/1142**. Existing non-fatal docstring warnings remain.
- Zarr/schema-dependent tests used permitted local runtime/network access;
  a restricted-runtime Zarr stall was interrupted and rerun, not treated as
  a product failure or worked around in source.

## Review findings fixed

A later invalid fill-value group used to leave an earlier COG on disk. All
selected COGs are now validated before eager/deferred execution. Canonical
path checks reject relative and absolute aliases of one output. Independent
review then found NaT metadata loss and malformed-date failures occurring
only after an earlier tree group wrote; failing tests reproduced both and
GeoTIFF metadata preflight now rejects them before any writes.

## Remaining risks and compatibility

Fluent writer signatures and documented return/destination semantics are
unchanged. The removed layout module and planning types have no forwarding
aliases. Tree to_netcdf/to_zarr still take their documented destination folder;
to_catalog and to_cog use asset templates. Multi-file writes are not
transactional against later I/O/runtime failures. Existing ModelSpec/template
migration failures remain outside this refactor; notebooks were untouched.

## Exact remaining full-suite failures

- `tests/cli/core/test_workspace.py::test_walkthrough_runs_without_model_or_external_service`
- `tests/model/spec/test_model.py::test_inference_reads_an_ingested_sample_through_preprocess`
- `tests/model/spec/test_model.py::test_training_samples_pair_the_ingested_table_with_a_label_table`
- `tests/model/spec/test_model.py::test_prepared_pixels_are_the_preprocessed_raw_pixels`
- `tests/model/spec/test_model.py::test_prepare_without_a_path_writes_a_temporary_catalog`
- `tests/model/spec/test_model.py::test_prepare_keeps_pixels_lazy_until_written`
- `tests/model/spec/test_model.py::test_prepare_names_a_missing_kept_collection`
- `tests/templates/tasks/classification/supervised/test_data.py::test_training_batches_move_pixels_and_preserve_class_targets`
- `tests/templates/tasks/classification/supervised/test_module.py::test_trainer_consumes_model_ready_samples`
- `tests/templates/tasks/pretraining/mim/test_data.py::test_training_batches_are_augmented_then_transformed`
- `tests/templates/tasks/pretraining/mim/test_data.py::test_input_values_are_not_repaired`
- `tests/templates/tasks/pretraining/mim/test_module.py::test_trainer_consumes_model_ready_samples`
- `tests/templates/tasks/regression/mean_teacher/test_data.py::test_strong_augmentations_cannot_move_geometry`
- `tests/templates/tasks/regression/mean_teacher/test_data.py::test_labeled_values_and_unlabeled_views_are_finished_together`
- `tests/templates/tasks/regression/mean_teacher/test_data.py::test_evaluation_batches_are_supervised_ones`
- `tests/templates/tasks/regression/mean_teacher/test_module.py::test_trainer_consumes_model_ready_samples`
- `tests/templates/tasks/regression/supervised/test_data.py::test_target_values_move_with_the_pixels`
- `tests/templates/tasks/regression/supervised/test_data.py::test_values_an_augmentation_invents_are_unknown_not_zero`
- `tests/templates/tasks/regression/supervised/test_module.py::test_trainer_consumes_model_ready_samples`
- `tests/templates/tasks/segmentation/fixmatch/test_data.py::test_strong_augmentations_cannot_move_geometry`
- `tests/templates/tasks/segmentation/fixmatch/test_data.py::test_strong_views_perturb_the_weak_view_without_moving_it`
- `tests/templates/tasks/segmentation/fixmatch/test_data.py::test_labeled_chips_are_finished_as_in_supervised_training`
- `tests/templates/tasks/segmentation/fixmatch/test_data.py::test_shared_geometry_is_applied_before_splitting_views`
- `tests/templates/tasks/segmentation/fixmatch/test_data.py::test_evaluation_batches_pass_as_supervised`
- `tests/templates/tasks/segmentation/fixmatch/test_module.py::test_trainer_consumes_model_ready_samples`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_the_test_split_needs_its_dataset`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_augmentations_move_image_and_target_together_while_training`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_cutmix_pastes_the_same_box_into_pixels_and_labels`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_augmentations_do_not_run_outside_training`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_transforms_run_in_every_split_after_augmentation[True]`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_transforms_run_in_every_split_after_augmentation[False]`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_input_transforms_preserve_nan_values`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_a_time_axis_survives_augmentation_and_transforms`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_a_pixel_an_augmentation_invents_is_ignored`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_nonfinite_input_values_are_preserved`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_multiple_pixel_inputs_share_geometry_and_preserve_context`
- `tests/templates/tasks/segmentation/supervised/test_data.py::test_lightning_cli_builds_each_split_dataset`
- `tests/templates/tasks/segmentation/supervised/test_module.py::test_trainer_consumes_model_ready_samples`
