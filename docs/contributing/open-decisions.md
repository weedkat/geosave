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

Implemented 2026-11-15:

```text
ml/processing/roles.py      Ingestor, Preprocessor, Predictor, Postprocessor; one Registry each
ml/processing/ingestors.py  Sentinel2Ingestor ("sentinel2_l2a")
ml/processing/preprocessors.py  Reflectance ("sentinel2_reflectance")
ml/processing/predictors.py ChipPredictor ("chips"): RasterChipper -> forward -> RasterMerger
ml/processing/postprocessors.py ClassMap ("class_map")
ml/processing/processor.py  Processor(model, ingestor, preprocessor, predictor, postprocessor)
                            predict_region, predict_path, predict_raster; ProcessorSpec, build_processor
ml/release                  ChainConfig.processor, from_chain(chain), from_processor(processor), load_processor(repo)
geosave_flows/monitor.py    monitor(processor): entry points become flows, sub-processor calls tasks
```

A release names only library components and sub-processors, so it loads with
`pip install geosave`; deploying a model means adding its parts to geosave.

Hugging Face repos, one audience each (decided 2026-11-15):

```text
org/<name>-data       dataset repo   raw/ processed/ labels; not LitData stores
org/<name>-training   model repo, private   workspace code + artifacts/<run>/*.ckpt, one commit per run
org/<name>            model repo, public    config.json + model.safetensors + model card
```

LitData stores are preloads tailored to one model; they stay local or in a
bucket. Later: record the data and training commits in the release's model
card (lineage), once configs name the data repo revision.

There is one main `Processor`, the common pattern of using a model; every
step around the model is an interchangeable sub-processor. A release stores
the four sub-processors as `{name, init_args}` each;
`ARTIFACT_FORMAT_VERSION` is 2. This supersedes the earlier plain-function
registries, the proposal to reuse ModelSpec as a declaration, and a registry
of main processors. `workspace/dynamicworld` runs it end to end: train,
`scripts/publish.py`, `scripts/predict_region.py`.

Open: offline inference (preload ingest, preprocessing and chips into LitData,
then predict from the store); `Processor.forward` passes the model `image` only
(named versus stacked tensors, time and location inputs); missing inputs/time
steps; progress inside one raster is not visible to `monitor`. See
[release and inference](design/release-inference.md).

Per-class thresholds are postprocessor `init_args`, found after training by
`calibrate_thresholds` on the best checkpoint's validation predictions; it is
a stub. Decode merged predictions; do not average integer class ids.

## Template and notebook migrations

ModelSpec is disabled. Its template data workflows need replacement while
retaining project-owned Modules and sample types. Migration work includes the
template `data.py`/`datasets.py`, recipe files, configuration keys and tests that
import their helpers. `segmentation/supervised` no longer imports ModelSpec in
its modules; the other methods' `data.py` still do, so their DataModules raise
`NotImplementedError` when constructed.

The earlier notebook audit identified removed `gs.write_grid` calls in lessons
0 and 1, and old chip APIs in later lessons. They remain separate notebook
migration work; see [examples](../examples.md). Do not treat those notebooks as
verified by the executable Markdown tutorials.

## Explorer and architecture

- Decide whether native DataTrees need their own overview figure; plot selected
  Dataset nodes today.
- Static export without a live kernel, time scrolling, point/region series and
  separation of pixel reading from display state remain design topics.
- Declare and test the dependency direction inside `geodata`; public package
  direction is already `geodata <- model <- ml` and `geosave <- geosave_flows`.
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
`ChipPredictor`'s chip settings.

`workspace/gedi-biomass` still calls the removed `gs.Chips`
(`modules/preparation.py`, `modules/prediction.py`, its Dataset); migrate it to `RasterChipper` and a processor, or archive it.

Older timing tables and completed implementation lists have been retired here;
[direction](design/direction.md) retains historical observations explicitly as such.
