# Open decisions

Library status checked on 2026-11-15. Implemented contracts live in
[architecture](architecture.md); this page records remaining decisions and
migration work. Design rationale lives in [direction](design/direction.md).

## Publication and catalog loading

The [GeoDataTree specification](design/raster-tree.md) records the implemented
separation: general raster leaves, independent grids, effective inherited attrs,
variable grouping, dimension split, merge, concat and ungroup. The default
`raster_leaves` policy requires leaf-owned grids; crop/reprojection uses native
tree mapping without parent-coordinate repair. Numerical channel conversion permits
different units and requires explicit dtype preparation when necessary.

`stac.publish_item(item, assets=..., root=...)` writes one observation's explicit
COG assets and returns a completed native Item. Native catalog composition and
GeoParquet round trips remain separate. RasterStac and automatic Dataset catalog
conversion are removed. Timeless assets and conflicting times/periods are rejected;
explicit membership cannot prove provenance when timestamps were copied.

Remaining design work is explicit, not missing foundation implementation:

- Standalone tree batch-file writer returning node-path-to-href mapping: not
  implemented here; its public entry point/destination policy is not settled.
  Publisher mapping uses explicit asset keys and does not substitute for that API.
- Per-leaf dtype/layout arguments on tree tensor conversion: not added without
  an agreed argument shape; prepare each Dataset explicitly first.
- Dedicated catalog-to-tree loading is deferred by agreement; compose existing
  loading with native DataTree construction. ODC native-grid grouping remains by
  CRS/resolution and can align differing pixel origins.
- Reference-product publication, alternate formats, deferred/transactional writes,
  and derived Collection extent updates require their own contracts. No fake dates
  or new provenance metadata are introduced.

Catalog sampling, vector rasterization and row pixel reading remain unavailable.
Explorer catalog pixel viewing depends on the row reader. GCP placement remains
unsupported by `gs.geobox`.

## Sampling and stores

Implemented: temporal `gs.iter_samples(rasters, freq)` with `missing="skip"`
or `"raise"`, matching a tree's dated nodes by time bucket; iterable
`RasterChipper` with pixel or fractional spatial overlap; `iter_batches`,
which reads one block of chunks at a time; and `RasterMerger`, which places each
prediction by its chip location's source, leaving NaN predictions to
overlapping chips. See
[the sampling reference](../reference/chips.md).

Remaining work:

- Same-resolution dense logits use the chipper's locations and input grids.
  Placement for model outputs at a different resolution remains open.
- Large output grids: implemented. `RasterMerger(store=...)` keeps a raster's
  sums and weights in Zarr, chips in any order; a dense postprocessor given a
  store uses `<store>/<window index>.zarr`. A Sentinel-2 tile merges in 0.6 GB
  of memory at about 27 ms per 256-pixel, 9-class chip, since each chip
  rewrites the up to four chunks it touches; a write-back chunk cache would cut
  that if staged merging becomes the bottleneck.
  Without a store a merger holds the whole raster, about 4.8 GB plus 4.3 GB for nine classes on a 10980² Sentinel-2
  tile. The pipeline passes `<run store>/product`.
- A window's product is dated by its chips' `source.timespan`: its bucket bounds
  when `write_time_bounds` stated them, otherwise its first and last
  acquisition.
- A merged prediction declares its source raster's extent (`ChipSource.extent`,
  written by `RasterMerger.merge`). Open: how a vector product, such as
  `vectorize` output, keeps that extent; a GeoDataFrame has no attr group for it.
- A window too small for one chip under `mode="drop"` yields no chip, so it
  has no product; a pipeline must notice it.
- Pixels on a raster's outer edge are only covered by chip rims, so they get
  context from one side only.
- Vector outputs: classification footprints, detection coordinates and duplicate
  removal, instance reconstruction.
- Persist sample rows, source identity and prepared-store record indices.
- Decide store invalidation when preprocessing changes.
- Support additional temporal selection policies only when required; current
  windows select the first or last observation per nonempty bucket. Choosing an
  observation by quality, such as the least cloudy, belongs to ingestion or
  preprocessing.
- Windows anchored on an event or a dated label, rather than time buckets,
  beyond selecting the label's period with `.sel(time=...)` first.
- Chips centred on sparse labels (GEDI footprints, field plots), skipping chips
  outside an area of interest, non-square chips, inputs at several
  resolutions, and model output at another resolution.
- Built-in time context (`time`, `day_of_year`, ...) needs one timeline; a
  project with several sensors passes its own context reader function.

A library preloading helper is not required by the current contract. Project
Datasets may read on demand or use a prepared store. Preserve grid and time
metadata with records; see [samples and chips](design/samples-chips.md).

## Processing, release and inference

`GeoRegion -> Ingestor -> collection subtree` is the implemented handoff.
`ml.pipeline.Pipeline` names the five stages; the contracts after ingestion
are designed one stage at a time. Each source owns its observation dates and grid.
The model author supplies the matching and preparation policy.
[Pipeline data flow](design/release-inference.md).

`pipeline(region, store=None)` is the one way to make a product. Ingestors
return their collections lazily; the pipeline loads them into memory, or, with
a store folder, stages every collection in `<store>/sources.zarr` in one
computation and reopens it lazily. The postprocessor stages its product under
`<store>/product`. Zarr staging is internal to a run; publication comes after
the pipeline. A run over several regions gives each region its own folder. An
existing store is refused. Reusing one for a retry needs a completion marker
written after the last chunk: a store left partial by an interrupted write
opens without error and reads its missing chunks as fill.

Open decisions include prompt inputs, detection outputs, pipeline input from
files, resuming a run from its store and the publisher. Native
`ChainModel.from_chain` releases remain available for trained model weights.

## ML frameworks

Of the five pipeline stages, only the predictor depends on an ML framework:

```text
ingest      GeoRegion          -> raw DataTree              framework-free
preprocess  raw DataTree       -> prepared rasters          framework-free
predict     model, prepared    -> scores on the input grid  per framework
postprocess scores             -> product                   framework-free
publish     product, Location  -> STAC Item                 framework-free
```

Each framework is one subpackage of `ml`. `ml.torch` exists; scikit-learn
would be a sibling holding its predictor (pixel table through
`xr.apply_ufunc`, no chipping) and release (skops). A prototype pixel
predictor returned the same lazy `(band, y, x)` scores the torch chip predictor
returns, and the segmentation template's postprocessor accepted them unchanged.

Open:

- The pipeline is standalone and releases stay model-only: the caller loads
  the release (`ModelChain.from_pretrained(repo)`) and passes the model to
  `TorchPredictor`. Loading a release from a pipeline YAML file is open. The
  prepared tree's raster node paths are the model's input names.
- Training per framework: the torch path is Lightning on chips; a scikit-learn
  path would sample a pixel table at labels with the same preprocessor.
- The install extra is still named `ml`, not after its framework.

## Template and notebook migrations

Geospatial workspaces now ship their editable Pipeline stages. Portable
Pipeline export/loading, project-specific readers, and the unfinished inference
stages remain open. Supervised segmentation trains on paired raster chips.
Its validation and test hooks require an evaluation input contract.
See [kit status](../reference/training.md) and the generated README.

Lessons 0 and 1 place predicted class maps on the input image's grid with
odc-geo's `wrap_xr`. Lessons 3 and 4 use `gs.RasterChipper` and the same
project scene function. Lesson 4 releases the chain and label scheme and
keeps the processor in Python. [Examples](../examples.md).

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
  scale/offset differs along time. The raster stamp refuses conflicting published packing.

## Workspaces

Dynamic World and the supervised segmentation template train with named model
inputs and an integer target. Both await a whole-scene evaluation contract.
The Dynamic World publisher still uses the retired raster pipeline.

`workspace/gedi-biomass` still calls the removed `gs.Chips`
(`modules/preparation.py`, `modules/prediction.py`, its Dataset); migrate it to `RasterChipper` and a processor, or archive it.

Older timing tables and completed implementation lists have been retired here;
[direction](design/direction.md) retains historical observations explicitly as such.
