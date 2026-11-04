# Unified GeoParquet validation

## Implementation

One GeoVector.to_geoparquet method delegates to io.vector.geoparquet.write.
Presence of attrs['stac-geoparquet'] selects STAC encoding at call time;
ordinary features retain GeoPandas encoding. Catalog metadata and required
columns are checked before output, and complete Item-shaped rows without a
marker cannot silently become ordinary features. Selected-column reads retain
STAC metadata, so incomplete catalog rewrites fail rather than lose Collections.
The separate to_stac_geoparquet accessor is removed without an alias.

Collection copying/extents, relative hrefs, bbox reconstruction, Item fields,
codec settings and storage options remain covered by round trips. Shared
Parquet options cannot override managed metadata. Both encoders reject
store_schema=False because it removes required GeoParquet metadata. Catalog
row_group_size uses the existing dependency's public Arrow-stream input.

## Fresh checks

- Vector/catalog/reader regressions before review fixes: **175 passed,
  6 deselected**.
- GeoParquet module after review fixes: **24 passed, 2 deselected**.
- Full configured suite after all changes: `.venv/bin/python -m pytest -q --tb=short`:
  **38 failed, 1381 passed, 37 deselected, 3 xfailed**, 155.75 seconds.
- Exact failed-test-name comparison against the recorded pre-task run
  (/tmp/geosave-paths-full.txt): **38 before and 38 after, no new or resolved failures**.
- Ruff (`ruff check src tests scripts`), touched-file formatting and
  `git diff --check` passed.
- BasedPyright on vector/core I/O, raster catalog accessor and ingest caller:
  **0 errors, 1 existing lazy-export warning** in io/vector/__init__.py.
- Zensical build passed. Generated API freshness and rendered coverage passed:
  **1141/1141**.
- Executed the catalog guide: **3 Items, 2 samples and readable chips** through
  the unified public writer.
- Tests needing Zarr synchronization/schema downloads used permitted local
  runtime/network access. Default slow/integration marker exclusions remained.

## Regression evidence and review

Nine new initial tests failed against the old common writer: metadata was not
encoded as STAC, malformed/incomplete/unmarked catalogs were accepted, and
selected-column reads lost Collections. Those tests now pass. Independent
review then reproduced loss of all required footer metadata from store_schema
and rejection of row_group_size. Failing regression tests reproduced both;
metadata guards and the Arrow-stream path made them pass. Review also checked
caller migration, write-time marker routing and selected-column retention.

## Compatibility and remaining work

Callers must use to_geoparquet instead of to_stac_geoparquet. Unmarked
Item-shaped frames require explicit catalog construction; invalid catalogs do
not downgrade to feature encoding. GeoJSON/GeoPackage continue rejecting loss
of catalog metadata. Collections overrides apply only to marked catalogs.
No notebooks, dependency changes, compatibility aliases, raster lifecycle
redesign or ModelSpec redesign were included.

The following ModelSpec/template migration failures match the pre-task run:

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
