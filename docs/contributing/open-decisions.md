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
- **ModelSpec on hold; processing-code publication.** User direction,
  2026-10-17: defer ModelSpec redesign until the logistics of publishing
  preprocessing and postprocessing code are settled. Decide how code is
  packaged, versioned, reviewed, and trusted at execution, and which processing
  belongs in installed library implementations versus the model repository.
  Keep current behavior available; do not expand the processing language or
  commit to preserving its YAML structure in the training-template migration.
- **Postprocessing in the spec (deferred with ModelSpec).** `references: {logits: inference}` was rejected
  as ambiguous: "it does not say which model output, representation, dimensions,
  or raster grid is referenced."
- **ModelSpec implementation rework (on hold).** The earlier direction was to
  preserve and refine the YAML structure (rasters, preprocessing, chips, inputs,
  context, transforms). That preservation requirement is now provisional pending
  the processing-code publication decision above. Required inference preparation
  must remain reproducible; training augmentation stays project-owned.
- **Dataset owns ModelSpec preparation.** User direction, 2026-10-16: use a
  standard PyTorch Dataset to hold the ModelSpec and coordinate sampling,
  framing, chipping and raster reads. Keep catalog, frame and chip
  GeoDataFrames as the Dataset's preparation state, available for inspection
  and prediction merging. Prepare the tables in `__init__`; open requested
  rows and produce tensors in `__getitem__`. Use the same preparation for
  labelled training and unlabelled inference. Apply this direction when
  reworking ModelSpec instead of introducing another preparation flow or
  execution framework. Notebook 3 demonstrates the Dataset/table boundary
  without ModelSpec; keep ModelSpec out of the basic notebooks while its API
  remains in development.
  The model's input grid belongs at `chip(..., target=grid)`, independently
  of collection membership. Spatial sampling groups equal footprints and chooses no grid. Preserve the grid policy
  for unlabelled inference; read each chip with `target=row.gs.geobox` to
  request alignment. Chips retain their full layout grid for merging. A
  model producing a different output resolution also needs an explicit
  output-grid contract; the current merger expects chip-sized outputs.
- **`in_channels`** is typed by hand in the task templates' configs although the
  model spec lists the variables.

## Training outputs

- **Prediction and chip outputs back to rasters and tables.** Not built; the
  first attempt (`UnlabeledDataset`, `RasterMerge`, `write_rasters`,
  `write_classes`) was removed. The segmentation Module returns logits from `predict_step`;
  the DataModules have no prediction loader.
- **Per-source Datasets.** DataModules take one Dataset per split returning
  their `Sample`; the library ships none, so the segmentation and regression
  task templates' configs name a local `modules.<task>.<method>.datasets.MyDataset` stub.
- **Whole-raster evaluation.** Validation scores chips; scoring rebuilt rasters
  could return as a callback for chipped datasets.

## Data

- **Prepared normalization statistics** are not computed from catalogs yet.
  `ModelSpec.prepare`, `preprocess`, `model_inputs`, and `load_rasters` are built;
  computing statistics from a catalog is not.
- **Catalog per collection.** Approved, not built: a catalog is a folder with one
  item table per Collection, so replacing labels rewrites only `label.parquet`.
  Today `to_geoparquet` rejects a suffix-less path and `open_vector` and
  `StacTable` reject a folder. See
  the git history of `specs/2026-10-09-catalog-per-collection-design.md`.
- **Remote `StacTable` credentials.** rustac reads a table through DuckDB,
  which takes DuckDB secrets (`duckdb=`), not the fsspec `storage_options`
  every other remote read and write takes. Translating `S3StorageOptions` into
  a DuckDB secret is not built. See
  the git history of `specs/2026-10-12-stac-structure-design.md`.
- **Remote catalog writes** go through `io.vector.geoparquet.write` and were not
  smoke-tested.
- **Single-band COG without a stored band name** (for example `<scene>/B04.tif`)
  reads as `band_1`; a file-stem fallback is deferred.
- **Explorer environments and computed overviews.** The native Panel explorer
  uses independent raster callbacks and native GeoViews elements. Its table
  remains below the map while sidebar selection changes. Full-extent computed
  rasters can still execute every source chunk; no pyramid is generated.
  Static plot spacing
  and VS Code's missing static copy control remain separate follow-ups.

## Examples

- [The examples plan](specs/2026-10-09-examples-series-design.md) follows the
  existing `00_basics`, `01_model`, `02_sources`, and `03_inputs` notebooks.
  `03_inputs` stays basic: TIFF paths, Dataset-owned preparation and a
  model built inside the LightningModule. `04_recipe` introduces ModelSpec
  after its API is settled, with preprocessing, prepared data and the supervised training
  task. Pretrained models, multiple sources, workspaces and model releases
  follow as separate planned lessons. Orchestration and deployment remain
  deferred. See [the examples guide](../examples.md) for runnable lessons.

## Deferred by the user for this cleanup

- Starting a new run from earlier checkpoint weights.
- Catalog folders with one table per collection.
- A CI doctest smoke job.
