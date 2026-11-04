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
- **ModelSpec YAML execution iteration (2026-11-04).** User authorized implementation
  and end-to-end iteration. v3 uses installed calls for preparation, encoding,
  shared tensor transforms, chip decoding, and merged-output postprocessing.
  Plain mappings retain per-sample state; no processor wrapper or code-upload
  mechanism was introduced. See [the implementation plan](plans/2026-11-04-model-processing.md).
  This supersedes the earlier hold. Scope inference across scenes/windows/chips,
  general intermediate persistence, augmentation/context policies, and publication
  dependency/trust rules remain open. The current input boundary is an explicitly
  selected sample with sufficient spatial/temporal support.
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
- **stac-geoparquet writer deviations.** stac-geoparquet 0.8.2 and 0.8.3 write
  `version: "1.0.0"` beside `collections` (a 1.1.0 field), an EPSG:4326 rather
  than OGC:CRS84 geometry CRS, `geometry_types: []`, and the `type` column the
  spec does not recommend. All come from its `to_parquet`; rewriting its
  metadata on our side would be glue. Dropping `type` was tried and reverted:
  rustac's table search and mixed old/new tables then yield Items pystac
  rejects. Raise upstream or wait for a release that fixes them.
- **Remote `StacTable` credentials.** rustac reads a table through DuckDB,
  which takes DuckDB secrets (`duckdb=`), not the fsspec `storage_options`
  every other remote read and write takes. Translating `S3StorageOptions` into
  a DuckDB secret is not built. See
  the git history of `specs/2026-10-12-stac-structure-design.md`.
- **Remote catalog writes** go through `io.formats.geoparquet.write_geoparquet` and were not
  smoke-tested.
- **Single-band COG without a stored band name** (for example `<scene>/B04.tif`)
  reads as `band_1`; a file-stem fallback is deferred.
- **Explorer environments and computed overviews.** The native Panel explorer
  uses independent raster callbacks and native GeoViews elements. Its table
  remains below the map while sidebar selection changes. Full-extent computed
  rasters can still execute every source chunk; no pyramid is generated.
  Static plot spacing
  and VS Code's missing static copy control remain separate follow-ups.
- **Explorer controls that need a UI design.** User direction, 2026-10-24:
  attempts were reverted for their design, and the behavior below is built
  without a control. Design each before adding it to the sidebar.
  - *Place search.* `utils.geocode.search_places(query)` returns
    `{name: BoundingBox}` in EPSG:4326, and `Explorer.zoom(bbox)` frames one.
    Nominatim's policy allows searching on submit only, about once a second.
    Rejected: results as a menu list under the box (it read as a layer list),
    and a "Matches" select under the box.
  - *Axes besides time.* A raster with another non-spatial axis (a level or a
    depth) shows its first step. `RasterLayer.axes` holds each axis's labels
    and `RasterLayer.steps` the position shown; setting `steps` redraws.
  - *Startup loader.* On a cold tile cache a served page can sit under the
    Material UI template's loader for its 30-second fallback: tiles requested
    before the map moves to another zoom load but are never drawn, and Bokeh
    counts only drawn tiles as finished. Measured: 32 s cold, 3 s cached,
    2.5 s with no basemap. Candidate:
    `#loader:not(.error) { display: none !important; }` in the page's own
    `config.raw_css`.

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
