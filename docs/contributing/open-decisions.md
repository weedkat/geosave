# Open decisions

Library status checked on 2026-11-15. Implemented contracts live in
[architecture](architecture.md); this page records remaining decisions and
migration work. Design rationale lives in [direction](design/direction.md).

## Publication and catalog loading

Implemented: `RasterStac.from_dataset`, `merge`, `to_cog`,
`Dataset.gs.to_catalog`, native Item/table conversion, catalog concatenation,
row `to_item`, table `to_dataset`, and ODC loading on chosen or native grids.
The old STAC-specific DataTree accessors have been removed. General trees have
caller-defined hierarchies; the staging wrapper owns Item/asset structure.

Remaining questions and limitations are maintained in
[publish and load](design/publish-load.md#gaps-to-settle):

- Item identity across repeated observations, fused observations and grids.
- Scalar-time/timeless publication; static variables currently repeat per Item.
- Original time labels for periods, property selection and arbitrary Dataset attrs.
- Derived Collection identity and extent updates.
- Native-grid return shape and concatenation across differing href bases.
- Deferred publication, classification metadata, alternate asset formats and
  preflight validation of all destinations.

DataArray `to_catalog`, raster `to_item` and Dataset `to_cog` remain stubs.
Use `Dataset.gs.to_catalog` for dated assets, or `io.write_cog(dataset, path)`
for one named-band file. Catalog sampling, vector rasterization and row pixel
reading remain unavailable. Explorer catalog pixel viewing depends on that
row reader; loading selected rows into a Dataset is available independently.
GCP placement remains unsupported by `gs.geobox`.

## Sampling and stores

Implemented: raster temporal `.gs.sample`, spatial `ChipLayout`, lazy
`RasterChipper`, and dense `RasterMerger(chipper.layout)`. See
[the sampling reference](../reference/chips.md).

Remaining work:

- Stream merges beyond memory; current accumulators occupy full output arrays.
  `RasterMerger.is_complete` and `discard` free a raster once its last chip is
  merged, so only rasters in progress are held.
- Vector outputs: classification footprints, detection coordinates and duplicate
  removal, instance reconstruction.
- Persist sample rows, source identity and prepared-store record indices.
- Decide store invalidation when preprocessing changes.
- Support additional temporal selection policies only when required; current
  windows select the first or last observation per nonempty slot.
- Multi-grid sample preparation and labels with independent timelines.

A library preloading helper is not required by the current contract. Project
Datasets may read on demand or use a prepared store. Preserve grid and time
metadata with records; see [samples and chips](design/samples-chips.md).

## Processing, release and inference

Implemented 2026-11-16:

```text
ml/processing/raster/pipeline.py    RasterPipeline(model, preprocessor, predictor, postprocessor, ingestor=None)
                                    predict(rasters), predict_to_zarr(rasters, destination, chunks)
ml/processing/raster/ingestion/      base.py RasterIngestor; sentinel2.py Sentinel2Ingestor ("sentinel2_l2a")
ml/processing/raster/preprocessing/  base.py RasterPreprocessor; dense.py DensePreprocessor ("dense"): bands, decode, fill, mean/std
ml/processing/raster/prediction/     base.py RasterPredictor; dense.py DensePredictor ("dense"): RasterChipper -> model(**inputs) -> RasterMerger
ml/processing/raster/postprocessing/ base.py RasterPostprocessor; dense.py SegmentationPostprocessor ("segmentation"),
                                     RegressionPostprocessor ("regression"); thresholds.py
ml/release                  ChainConfig.preprocessor/predictor/postprocessor/ingestor, from_chain(chain),
                            from_pipeline(pipeline), load_pipeline(repo)
```

A release names only library components. A library sub-processor is stored by
name; a project's own by class path, loading where the project's code is installed.

Hugging Face repos, one audience each (decided 2026-11-15):

```text
org/<name>-data       dataset repo   raw/ processed/ labels; not LitData stores
org/<name>-training   model repo, private   workspace code + artifacts/<run>/*.ckpt, one commit per run
org/<name>            model repo, public    config.json + model.safetensors + model card
```

LitData stores are preloads tailored to one model; they stay local or in a
bucket. Later: record the data and training commits in the release's model
card (lineage), once configs name the data repo revision.

`RasterPipeline` orchestrates inference over rasters. Every step around the
model is an interchangeable sub-processor, named by its input and entered by
`__call__`; `@chain_step()` on it declares the values handed between steps. Each step
is a package holding its abstract role (`base.py`) and its implementations, named by task
(`dense.py` for per-pixel segmentation and regression); subclass where a
model needs more.
The contract between steps carries no fixed names: the ingestor returns a
DataTree of source rasters, the preprocessor returns a DataTree whose nodes
are the model's pixel inputs, and the model is called with the batch by name.
A release stores the sub-processors as `{name, init_args}` each, the ingestor
optionally; `ARTIFACT_FORMAT_VERSION` is 3. This supersedes the earlier plain-function
registries, the proposal to reuse ModelSpec as a declaration, and a registry
of main processors. `workspace/dynamicworld` runs it end to end: train,
`scripts/publish.py`, `scripts/predict_region.py`.

`predict_to_zarr(rasters, destination, chunks)` predicts chunk by chunk.
Chunks start on multiples of the predictor's stride and each is read with a
halo (`sampling.find_chip_span`: the pixels of every chip touching it, plus
what a reflection reads at the raster's end), predicted, cropped back to the
chunk and written into its region of one Zarr store (`create_zarr`,
`write_zarr_region`). The chunk's chips are then the whole raster's, so the
store equals `predict` for padding modes; `shift`, `drop` and `wrap` layouts
are not reproduced exactly. Memory holds one chunk's output. `predict` is in
memory with no chunk concept; `predict_path` and `predict_region` were
removed, and the caller runs `processor.ingestor`; `RasterIngestor` is an
abstract class whose `ingest_to_zarr` writes each acquisition to a store first.

Deployment (2026-11-16): the whole processor runs on the server. A release
accepts a project's sub-processor by `class_path` (`check_deployable_spec`);
it rebuilds wherever the project's code is installed, so the server image is
built from the workspace. The segmentation template's `scripts/.../serve.py`
is a LitServe API taking `{"rasters": store, "destination": store}` and
calling `predict_to_zarr`. LitServe is not a dependency of geosave. Open:
the other templates have no `serve.py`; a region request that ingests first;
loading a release without the project's code (shipping `processing.py`). A LitData preload of input chips was tried and
removed: it did not lower memory, and in deployment the model sits behind
LitServe, tensors in and tensors out. Open: nothing checks that hand-built rasters match what the
ingestor would return (bands, packing, resolution); one acquisition per store,
no time cube yet; chunks are predicted one after another; serving the model remotely (`DensePredictor.predict` reads the
model's device); the templates' DataModules still call their own preprocessor's `normalize`, which the library no longer has;
missing inputs/time
steps. `monitor` was replaced 2026-11-16 by the `geosave_flows.predict.predict_region`
flow, whose tasks are plain `@task` functions; progress inside one raster is not a task. See
[release and inference](design/release-inference.md).

`calibrate_thresholds` remains a stub. The built-in segmentation postprocessor
accepts class names/colors and uses argmax; it does not yet accept thresholds.
A future calibrated decoder needs its thresholds persisted with its settings.
Decode merged predictions; do not average integer class ids.

## Template and notebook migrations

ModelSpec is unavailable. Template DataModules now build explicit preprocessors
and take project Datasets; they no longer import ModelSpec. Most task Datasets
remain placeholders. Supervised segmentation has a paired raster Dataset and
scene-merging evaluation, but its Dataset/DataModule supply flat named samples
while Module steps still read nested `inputs`; complete that integration before
claiming a full training run. See [recipe status](../reference/training.md).

Its publication script supports separate release and private training repos;
other generated publication scripts still combine workspace archives with the
release. Migrate them to the [three-repository workflow](../mlops/index.md).

The current notebook source still references removed `geosave.model` in
lesson 1 and removed model/chip APIs in lessons 3 and 4. They remain separate
migration work; see [examples](../examples.md). Do not treat the notebook
series as verified by executable Markdown tutorials.

## Explorer and architecture

- Decide whether native DataTrees need their own overview figure; plot selected
  Dataset nodes today.
- Standalone map export is implemented through `Explorer.snapshot()` and
  Export map; it captures current pixels. Time scrolling, point/region series
  and separation of pixel reading from display state remain design topics.
- Declare and test the dependency direction inside `geodata`; public package
  direction is already `geodata <- ml` and `geosave <- geosave_flows`, with
  inference isolated from training support inside `ml`.
- Per-Item band metadata and provider-specific decoding need a contract when
  scale/offset differs along time. Loading currently refuses conflicting packing.

## Workspaces

`workspace/dynamicworld` (checked 2026-11-15) registers its ingestor and
preprocessor in `modules/processing.py`, measures normalization statistics with
`scripts/compute_statistics.py`, preloads chips into a LitData store,
and releases and predicts through `scripts/publish.py` and
`scripts/predict_region.py`. Its validation still merges chips in the Module
with its own `RasterMerger`; the agreed pattern is the segmentation template's:
`TensorChipper` and `TensorMerger` in the step, with the release
`DensePredictor`'s chip settings.

`workspace/gedi-biomass` still calls the removed `gs.Chips`
(`modules/preparation.py`, `modules/prediction.py`, its Dataset); migrate it to `RasterChipper` and a processor, or archive it.

Older timing tables and completed implementation lists have been retired here;
[direction](design/direction.md) retains historical observations explicitly as such.
