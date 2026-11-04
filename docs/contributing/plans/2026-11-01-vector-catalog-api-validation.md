# Vector/catalog simplification validation

## Implemented behavior

One GeoVector accessor serves ordinary features, Item rows and sampled Item
frames. GeoFrame and GeoCatalog are removed without aliases. Sample algorithms
live under core/vector; Item-row conversion lives in item.py. STAC persistence
is explicit and ordinary vector exports refuse silent Item metadata loss.

DataArray, Dataset and DataTree expose to_catalog(path, ...), composing the
existing file layout, format writers and STAC translators. Paths name raster
assets; templates support collection, time and band fields. Catalog output is
a native GDF. Source reuse without a supplied path is explicitly unsupported.

## Fresh checks

- Focused raster/catalog/vector/reader/writer/explorer regression command:
  `.venv/bin/python -m pytest tests/geodata/core/vector tests/geodata/core/test_register.py tests/geodata/core/ctor/test_catalog.py tests/geodata/core/raster/test_base.py tests/geodata/io/vector tests/geodata/io/test_openers.py tests/geodata/io/raster/test_layout.py tests/geodata/viz/explorer -q --tb=short`
  Result: **238 passed, 9 deselected** using the configured marker exclusions.
- Full configured suite: `.venv/bin/python -m pytest -q --tb=short`.
  Result: **38 failed, 1355 passed, 37 deselected, 3 xfailed** in 145.53 seconds.
  No ad-hoc test exclusions. Tests ran with permitted local runtime/network
  access; restricted-runtime Zarr stalls are not reported as product failures.
- New catalog guide executed end to end: **3 Items, 2 samples, readable chips**.
- Ruff: `ruff check src tests scripts` passed; `git diff --check` passed.
- BasedPyright on core, vector I/O, changed explorer modules, ModelSpec and
  registration typing: **0 errors, 1 warning**. The warning concerns the
  existing lazy stac_geoparquet export in io/vector/__init__.py.
- Zensical build passed. Generated API freshness passed; rendered coverage
  **1149/1149**. Existing Griffe docstring warnings remain non-fatal.

## Full-suite failures

The failed paths exercise the preceding refactor's unfinished ModelSpec and
training-template migration: sampled accessors are still indexed or counted as
old sample wrappers; ModelSpec.prepare passes removed target= to an Item row;
templates and walkthrough YAML still require the removed ModelSpec.chips.
These paths are outside this API implementation's no-redesign scope. Their
obsolete call patterns were inspected before implementation, but a full
before-change baseline suite was not run; this report does not claim a
reproduced pristine baseline or a green complete suite.

Exact failed tests:

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

## Breaking changes and limits

- Import GeoVector instead of GeoFrame or GeoCatalog. GDF accessor typing no
  longer has a schema generic parameter.
- Persist Item tables with to_stac_geoparquet; to_geoparquet is for features.
- path=None cannot reuse a source until unchanged-source verification exists.
- Each published Item represents one observation; multi-date stores require a
  time template. Mixed output formats in one template mapping are rejected.
- Filename collisions, missing grids and multi-date observations fail before
  writes. A later writer or metadata translation failure can leave assets
  already written; publication is not a multi-file transaction.
- Existing notebooks were intentionally left untouched.
