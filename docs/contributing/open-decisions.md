# Open decisions

Unsettled design questions. Delete an item when it is built or dropped. Other
design history lives in git.

## Models and prediction

- **Head-owned prediction and calibration.** Calibrated thresholds needed for
  production interpretation must survive native save/load; today
  `ThresholdCalibrator` lives on the Lightning module. Define head ownership with
  a native release round-trip smoke before adding new head types. Do not make
  every head use one merger or assume a chain attribute named `head`.
- **Detection.** "Settle the chosen detector's box convention/decoder and
  all-empty completion persistence" before a detection flow.
- **Release functions and serving.** The shipped release is `GeoSaveModel`
  (Transformers). Free functions `save_model` / `load_model` / `publish_model`,
  a generated model card, and a LitServe adapter are unbuilt. "LitServe dynamic
  batching is deferred until a per-sample request interface and concurrent remote
  runner exist."
- **Postprocessing in the spec.** `references: {logits: inference}` was rejected
  as ambiguous: "it does not say which model output, representation, dimensions,
  or raster grid is referenced."
- **ModelSpec implementation rework.** Keep the YAML's structure (rasters,
  preprocessing, chips, inputs, context, transforms), refine it, and rework the
  implementation behind it. Spec `transforms` stay the inference-time
  preprocessing (the Hugging Face `preprocessor_config.json` role); training
  augmentation stays in the DataModule.
- **`in_channels`** is typed by hand in the starters' `train.yaml` although the
  model spec lists the variables.

## Training outputs

- **Prediction and chip outputs back to rasters and tables.** Not built; the
  first attempt (`UnlabeledDataset`, `RasterMerge`, `write_rasters`,
  `write_classes`) was removed. Modules have no `predict_step`.
- **Per-source Datasets.** DataModules take one Dataset per split returning
  their `Sample`; the library ships none, so the segmentation and regression
  starters' `train.yaml` name a placeholder `datasets.MyDataset`.
- **Whole-raster evaluation.** Validation scores chips; scoring rebuilt rasters
  could return as a callback for chipped datasets.

## Data

- **Prepared normalization statistics** are not computed from catalogs yet.
  `ModelSpec.prepare`, `preprocess`, `model_inputs`, and `load_rasters` are built;
  Prefect orchestration through `prepare_dense_data` remains deferred and is
  not a CLI command. Unused Clay-specific sensor statistics were removed.
- **Catalog per collection.** Approved, not built: a catalog is a folder with one
  item table per Collection, so replacing labels rewrites only `label.parquet`.
  Today `to_geoparquet` rejects a suffix-less path and `open_vector` and
  `StacClient.open` reject a folder. See
  `specs/2026-10-09-catalog-per-collection-design.md`.
- **Remote catalog writes** go through `io.vector.geoparquet.write` and were not
  smoke-tested.
- **Single-band COG without a stored band name** (for example `<scene>/B04.tif`)
  reads as `band_1`; a file-stem fallback is deferred.
- **Raster explorer.** A HoloViz/Panel explorer over native xarray objects
  (attributes, variable selection, spatial display, time scrubbing, lazy
  metadata) is not built; `geodata/viz` has static plots and `explore()`.

## Examples

- Notebooks 3 to 5 and the Ship part (6 to 8) of
  `specs/2026-10-09-examples-series-design.md` are deferred. Naming question:
  `transform.crop.crop` shares its name with `.gs.crop`.

## Deferred by the user for this cleanup

- Starting a new run from earlier checkpoint weights.
- Catalog folders with one table per collection.
- A CI doctest smoke job.
