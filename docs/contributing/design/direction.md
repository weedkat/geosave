# Direction: architecture, contracts and data flow

Updated against library source on 2026-11-16. Implemented contracts live in
[architecture](../architecture.md). This page retains design questions and
rationale; proposed APIs are not promises of available behavior. Details:
[publish and load](publish-load.md), [samples and chips](samples-chips.md),
[release and inference](release-inference.md).

## 1. Goal

One modular path for developing a geospatial model, whatever the task
(segmentation, regression, detection, classification, pretraining) and
whatever the model needs (one date or a time series, one sensor or several
stacked, with or without time and location context):

```text
explore ─▶ operate ─▶ prepare ─▶ train ─▶ release ─▶ predict ─▶ publish
  EDA      2-D raster   samples   Lightning  Hub       catalog in   catalog out
```

## 2. Data flow and the objects that cross each boundary

```text
provider STAC -> Item rows -> ODC Dataset -> preparation -> one-grid DataTree
Dataset -> RasterStac -> COGs + Item rows -> GeoParquet
prepared tree -> temporal windows -> RasterChipper -> project Sample -> DataLoader -> Module
region or raster -> processor (ingest, preprocess, RasterChipper, model, RasterMerger, postprocess) -> product Dataset
```

Native Item rows, Datasets, DataTrees and task-owned sample dictionaries cross
these boundaries. `ChipLayout` records the raster and placement of each chip;
GeoBoxes state exact grids. A release stores the processor that runs its
model; the `dense` predictor is the library-owned prediction loop.

## 3. Workflow, stage by stage

| Stage | Current route | Remaining work |
| --- | --- | --- |
| Search | `StacClient.open(provider).search(query)` | provider-specific preparation |
| Load | `rows.gs.to_dataset(geobox=grid, chunks={})` or `stac.load_dataset(items, ...)` | native-grid return shape remains a design question |
| Open and operate | native xarray, raster accessors, features and visualization | see G1–G9 |
| Publish | `ds.gs.to_catalog(path)`; `RasterStac` for several grids | identity and metadata limitations in publish-load |
| Cut and merge | `RasterChipper([tree])`, `RasterMerger(chipper.layout)` | streaming output, vector outputs |
| Plan sample rows | project pandas operations | persisted sample-catalog contract |
| Store | project preparation | store invalidation and record identity |
| Train | project Dataset, DataModule, Module and LightningCLI | implement placeholder Datasets and finish template sample integration |
| Release | `ChainModel.from_chain(chain)` or `ChainModel.from_pipeline(pipeline)` | project-owned data/experiment lineage |
| Predict | `load_pipeline(repo).predict(rasters) / predict_to_zarr(rasters, destination)` | catalog orchestration, streaming output and vector outputs |

## 4. Design groups

Each group: its contract, the pattern in use or suggested, alternatives
worth arguing, and flags (coupling, complexity, bugs).

### G1 Raster object and accessors (`core/raster`)

- **Contract**: a Dataset/DataArray on one GeoBox in ODC layout → the same,
  lazy, with metadata preserved. Structural requirements are stated as rules:
  `ds.gs.validate(rule, ...)`.
- **Pattern**: accessor (facade) on the native object + rule functions
  (strategy as a plain callable). `GeoRaster` ABC shares the grid contract
  across the three accessors.
- **Alternatives**: (a) free functions `gs.reproject(ds, ...)` and no
  accessor; (b) a wrapper class owning the Dataset. Both lose native objects
  in user code; not suggested.
- **Flags**:
  - Many methods exist on both `GeoDataset` and `GeoDataArray`.
    Accepted so far: no type branches, one concrete type per
    function. Discuss whether persistence (`to_zarr`, `to_netcdf`, `to_cog`)
    and viz (`plot`, `explore`) belong on both, or only on Dataset.
  - The accessor reaches `io`, `viz` and `stac` through lazy imports. That is
    the facade working as intended, but the direction inside `geodata` is not
    declared or tested (see G16).

### G2 Metadata (`attrs`, `conventions.py`)

- **Contract**: a set of attrs that mean one value together ⇄ one validated
  object: `CFPacking.from_attrs(band.attrs)`, `stamp_attrs(band, packing)`.
- **Pattern**: Pydantic value object per standard (`CFPacking`, `CFFlags`,
  `EOBand`, `ACDD`, `StacCollection`, `StacProperty`, `Display`).
- **Alternative**: read through cf-xarray only. It covers CF, not STAC, EO
  or display, so the groups stay.
- **Flag**: none structural. This group is the cleanest in the library.

### G3 Raster and vector files (`io`)

- **Contract**: `Location` (path, URL, UPath) ⇄ Dataset, DataTree or
  GeoDataFrame. Every writer: locate → check destination → encode → write
  now or deferred. Every reader runs it back.
- **Pattern**: one module per format; `Codec` pairs (`encode`/`decode`)
  listed per format (`zarr.CODECS`); reader chosen by suffix (`find_opener`).
- **Alternatives**: a format class hierarchy (ABC `Format` with
  `open`/`write`); not suggested, the modules share steps, not state.
- **Flags**:
  - Resolved 2026-11-15: `open_raster` returns a Dataset for the root or an
    explicit `group=`; `open_raster_tree` returns a DataTree. Zarr and NetCDF
    also have separate Dataset and tree openers.
  - `open_raster(**options: Any)` documents three backends' options in one
    docstring (about 200 lines). The typed per-format functions
    (`open_geotiff`, `open_zarr`, `open_netcdf`) already exist; `open_raster`
    could accept only what all formats share.
  - `codecs/gdal.py` owns substantial GeoTIFF metadata translation.
    It is the first place to look when
    GeoTIFF round trips break.

### G4 Catalog: Item rows (`core/vector`, `stac/catalog/`, `stac/search`)

- **Contract**: STAC Items ⇄ GeoDataFrame rows, Collections in
  `attrs["stac-geoparquet"]`; asset hrefs are absolute in the table and in
  the file, and a relative one is resolved once, on read.
  Selection is plain pandas. A table states what it guarantees through rules:
  `rows.gs.validate(stac_items, one_collection, dated)`.
- **Pattern**: accessor + rule functions; `StacQuery` frozen dataclass as a
  query value.
- **Alternatives**: a `Catalog` class wrapping the frame (loses pandas);
  pystac `ItemCollection` as the public object (not columnar, no pandas
  selection).
- **Flags**:
  - `StacClient` (API) and `StacTable` (file) implement the same five
    members by convention. A `Protocol` would state it; today nothing does.
  - `search(query: StacQuery | SearchParams)`: a union parameter serving two
    callers.
  - Collections ride in pandas `attrs`, which `merge`, `gpd.sjoin` and a
    `concat` of differing frames drop (checked). `concat_catalogs` exists
    because of this, and sample planning (G10) joins rows, so it must carry
    the Collections across itself. Hrefs are absolute, so a dropped footer
    loses Collection links only.
  - Catalog sampling, row `read_pixels` and vector `rasterize` remain unavailable.
    `to_dataset` is implemented; `to_datatree` and `GeoSamples` are removed.

### G5 Loading and metadata stamps (`stac/load.py`, `stac/stamps`)

- **Contract**: Items + ODC options → Dataset; what the STAC documents state
  is written onto it without reading pixels.
- **Pattern**: ordered pipeline of named steps from a registry:
  `load_dataset(items, stamps=[*DEFAULT_STAMPS, "sentinel_2"])`,
  `Stamp = Callable[[Dataset, LoadSource], Dataset]`.
- **Rules kept from the stamp design** (2026-11-13): a stamp reads only STAC,
  never another stamp's output; names only, no callables in the call; a later
  stamp overrides attrs; each stamp checks its own consistency; an unknown
  name raises before loading; only scalar property values become coordinates;
  a coordinate is a property exactly when it carries `stac_property`.
- **Alternatives**: fixed private helpers (the previous state: no place to
  add provider fixes); callables passed directly (not selectable from YAML).
- **Flags**:
  - **Repeated time labels.** `groupby="id"` gives one step per Item, so two
    tiles of one pass share a label (`time` index not unique). `RasterStac.from_dataset` then
    raises `two Items format to the id`, because the id is formatted from the
    whole Dataset's place and time. See G6.
  - `load_native_datasets` returns `(crs, resolution, dataset)` tuples; the
    Dataset states both.
  - Not yet written: the `sentinel_2` stamp; per-step shared properties.

### G6 Publication (`stac/raster.py`, `stac/publish.py`)

- **Implemented**: `RasterStac.from_dataset` groups a Dataset into an Item/asset
  tree. Collection metadata stays on the wrapper. `merge` joins by Item id;
  `to_cog` writes pixels and returns Items; `create_catalog` produces rows.
  `Dataset.gs.to_catalog` is the one-grid shortcut. Loading returns Datasets.
- **Identity remains open**: formatting ids from time and footprint can collide
  for repeated observations or differ between grids intended for one Item.
  Options: carry source identity in a marked coordinate; allow property fields
  in formatting; retain explicit formatting and reject collisions. Carrying
  source identity also needs a rule for fused observations and derived products.
  Inferring footprints from valid pixels would require reads during staging.
- **Limitations**: an Item dimension is required; static variables repeat per
  Item; unmarked Dataset attrs are not published as Item properties. See
  [publication boundaries](publish-load.md#what-round-trips).
- **Pattern**: native DataTree as an inspectable publication layout, owned by
  `RasterStac`; native PySTAC Item builders. General DataTrees have no STAC rules.

### G7 Places and grids (`core/region.py`, odc-geo `GeoBox`)

- **Contract**: `GeoRegion` = geometry + timespan, no pixels. `GeoBox` =
  exact pixel placement. No operation chooses a grid implicitly.
- **Pattern**: frozen dataclass value with named constructors
  (`from_bbox`, `from_geometry`, `from_coordinates`).
- **Flag**: `GeoRegion` also formats ids (`format`), computes MGRS and
  geocodes over the network (`locate`). Three jobs on one value; `format` is
  the one publication depends on.

### G8 EDA and derived features (`features`, `core/raster/utils/statistics.py`)

- **Contract**: Dataset → pandas table (`statistics`, `class_percent`) or
  Dataset → DataArray (`features.ndvi`, cloud and shadow masks).
- **Pattern**: pure functions over one Dataset; results are native objects
  that plot and explore like any other.
- **Suggested addition**: statistics over a catalog without loading it
  whole (per-Item statistics as columns of the rows), the same columns the
  sample plan filters on (G10).

### G9 Visualization (`viz/plot`, `viz/explore`)

- **Contract**: `obj.gs.plot()` → static HoloViews figure;
  `obj.gs.explore()` → `Explorer`; explorers compose with `*`.
- **Pattern**: model–view with Param. `Layer` (Parameterized) holds data and
  display state and returns plain values (`draw`, `identify`, `swatches`);
  `page/` components bind to that state; `Explorer` owns layer membership,
  selection and **the one time state** (`period`, `freq`).
- **Settled** (2026-11-03): HoloViews push of view-sized images; no tile
  server.
- **Features toward a GIS app.** Each is a method on `Layer` returning plain
  values plus a page component bound to existing Explorer state. Control
  layout is the maintainer's design; only state and data contracts are
  proposed here.

  | Feature | State and data contract | Native piece |
  | --- | --- | --- |
  | Time scroller | steps `Explorer.period` forward and back, plays; no new state (a discrete period selector already exists) | `pn.widgets.DiscretePlayer` on `explorer.param.period` |
  | Time series at a point | new `Layer.read_series(x, y)` → DataArray along `time` for the shown bands, masked like `identify`; chart marks the current `period`; clicking the chart sets `period` | `hv.Curve` + `hv.VLine`, `Tap` stream |
  | Time series over a drawn region | the same over `Explorer.region`, reduced by mean | `ds.gs.crop(region).mean(spatial_dims)` |
  | Spectral profile at a point | `identify` values in band order with `EOBand` wavelengths on x | `hv.Curve` |
  | Histogram of the view | pixels already read for the view | `hv.Histogram` |
  | Sample and prediction review | sample rows and prediction rows are Item rows, so they already draw as footprints; catalog pixel view returns with W2 | existing `VectorLayer` |
  | Static export (implemented) | `Explorer.snapshot()` and Export map produce a standalone current view; new raster detail still requires Python | `hv.save(..., resources="inline")` |

- **Flags**:
  - `RasterLayer` owns reading, the pixel-window cache, stretch,
    styling and legend. A point series reads differently from a view
    (all times, one pixel), so reading and caching should separate from
    display state before `read_series` is added.
  - Catalog pixel viewing is paused on `row.gs.read_pixels` (7 tests
    skipped); it becomes `stac.load_dataset(items, geobox=view)` after W2.
  - A long series over remote COGs is one request per step; the chart needs
    a lazy, cancellable read.

### G10 Sample planning and chips ([samples-chips](samples-chips.md))

- **Contract**: Item rows + spec → **sample rows**: Item rows whose `proj:*`
  state the chip GeoBox, whose period is the time window, and whose
  properties list the source Items per time step.
- **Pattern**: `ChipLayout` owns grids and chip indices; `RasterChipper` cuts trees
  and `RasterMerger` reconstructs by the same layout. Proposed catalog planning
  uses pandas (`sjoin`, `Grouper`); "meaningful" is a
  filter over measured columns (`valid_fraction`, class counts, `split`).
- **Alternatives**: torchgeo-style samplers over an R-tree index (random
  access, no persisted plan, so no reproducible sample catalog); a virtual
  cube with a custom lazy backend (set aside 2026-11-12: a second laziness
  mechanism beside dask and ODC).
- **Open**: the rule filling a time step when several Items fall in it;
  candidate is a named callable (strategy) only if the release must state it.

### G11 The Sample and data loading ([samples-chips](samples-chips.md))

- **Contract**: tree tensor conversion → named tensor dict: pixel tensors,
  optional `transform` float64 (6,), `crs` int64 EPSG code and explicitly
  requested `TENSOR_CONTEXT` entries. Collates with
  torch's default; the GeoBox rebuilds from the batch.
- **Pattern**: TypedDict per task; one torch Dataset per data source,
  project-owned; the store (LitData) is an optimization holding the same
  dict, not a second contract.
- **Alternatives**: a Sample dataclass (does not collate); carrying only a
  row index (every consumer needs the table); torchgeo's `GeoDataset`
  (brings its own CRS/bounds indexing parallel to ODC).
- **Current**: `TENSOR_CONTEXT` computes requested time/location entries from
  the raster at tensor conversion. Geometry-changing augmentation does not
  update that context; see [training contracts](../../concepts/training.md).

### G12 Model construction (`ml/components`, `ml/chain`)

- **Contract**: a mapping of stage specs → `ModelChain` (`nn.Module`) with
  named inputs: `model(image=..., temporal_coords=...)`.
- **Pattern**: registry + factory (`register_model`, `build_model`,
  `BuildSpec`); dataflow routing by declared names (`@chain_step`);
  constructor wiring through `Published` attributes.
- **Alternatives**: (a) fixed composition with three Protocols
  (`Encoder → Decoder → Head`): simpler and type-checkable, but optional
  context inputs and multi-source encoders need special cases; (b) plain
  `nn.Module` per model with no chain: no reuse of decoders and heads.
- **Flags**:
  - Step outputs are discovered by parsing method source with `ast`
    (`chain/step.py`). It is the cleverest mechanism in the library;
    `outputs=(...)` on the decorator already states the same thing
    explicitly and could be required.
  - `chain/` owns routing. Worth it only if models with optional
    context and several inputs are a goal. They are (Prithvi-TL, Clay,
    S1 + S2), so keep, but simplify discovery.
  - Detection has no head or stage yet; `ImageAugmenter` already handles
    boxes.

### G13 Training (`ml`, `templates`)

- **Contract**: batches of `Sample` → Lightning Module; metrics, criterion,
  optimizer and scheduler built from config; LightningCLI entry point.
- **Pattern**: Lightning's own (Module, DataModule, Callback, CLI) + small
  factories by spec. The library holds reusable parts; the project owns the
  Module, DataModule and Dataset.
- **Ownership settled**: reusable parts stay in `ml`; the project owns its
  Module, DataModule and Dataset. Source templates are copied into the project.

- **Flags**:
  - Templates copy whole Modules per method. Deliberate
    ("editing one method must not change another"), but the methods drift
    apart. DataModules now build explicit preprocessors; most Datasets remain
    placeholders. Supervised segmentation's flat Dataset/DataModule samples
    do not yet match its Module's nested inputs.
  - Threshold calibration is a stub.
  - Most publication scripts still combine training archives with releases;
    supervised segmentation supports separate repositories. See
    [recipe status](../../reference/training.md).

### G14 Release, processing and inference ([release-inference](release-inference.md))

- **Implemented**: `ml/processing/raster` owns one abstract role and Registry per
  role plus `RasterPipeline`. `ChainConfig` stores each sub-processor's
  spec (`ARTIFACT_FORMAT_VERSION` 4); `load_pipeline` rebuilds
  them around the model. The dense predictor supports named pixel inputs and
  requested time/location context. The ModelSpec declaration, main processor
  registry and plain-function sketches are superseded.
- **Pattern**: registry + factory by role (strategy objects behind abstract roles);
  the pipeline runs them in order and checks the values handed between them.
- **Open**: different normalization per input, missing steps/sources, context
  after augmentation and streaming decoded products. Projects own statistics
  provenance and experiment/data revision links.

### G15 Outputs per task

- **Contract**: model output for one chip + its GeoBox → a georeferenced
  native object.
- **Outputs**: project postprocessors decode merged dense predictions today;
  vector reconstruction needs its own processors:

  | Kind | Result | Reconstruction |
  | --- | --- | --- |
  | segmentation | uint8 DataArray + `CFFlags` | merge logits by chip index, argmax |
  | regression | float DataArray + units | merge values by chip index |
  | detection | GeoDataFrame | chip pixels → map through the affine, NMS across overlaps |
  | instances | GeoDataFrame | `gs.vectorize`, dissolve across overlaps |
  | chip classification | GeoDataFrame per chip footprint | none |
  | embeddings (pretraining) | DataArray at patch resolution or rows | Open |

- **Current behavior**: dense merging preserves the source GeoBox, floors
  weights at covered borders and uses NaN for uncovered pixels. Accumulators
  still occupy full output arrays in memory.

### G16 Package boundaries (cross-cutting)

- **Settled**: `geodata ← ml`; `geosave ← geosave_flows`; inference inside
  `ml` does not import training support; no torch at
  `geodata` import; tested in `tests/test_layering.py`.
- **Flag**: inside `geodata` nothing is declared. Top-level imports form
  cycles (`io ↔ stac`, `core.vector → stac → core`), broken only by lazy
  imports.
- **Suggested**: declare and test one direction,
  `utils, errors, attrs ← core ← io ← stac ← viz`, with accessor methods
  the only code allowed to import upward, lazily.

### G17 Orchestration and workspaces (`geosave_flows`, `cli`, `templates`)

- **Contract**: a flow takes a validated config and produces one catalog
  (Settled 2026-10-30). A workspace is generated project code that composes
  library APIs.
- **Pattern**: plain `@flow`/`@task` functions over a pipeline; Pydantic configs
  (`ConfigModel`, region configs); template copying for workspaces.
- **Current**: `geosave_flows` holds the `predict_region` flow and its region
  configs. `monitor` and label reading were removed 2026-11-16.
  Broader preparation/catalog orchestration remains project-owned.

## 5. Pattern summary

| Pattern | Where | Not where |
| --- | --- | --- |
| Accessor facade | G1, G4 | no wrapper classes around xarray or pandas |
| Rule functions | G1, G4 | no rule classes |
| Pydantic value / declaration | G2 attrs, G17 configs, `BuildSpec` | not for values that never leave the process |
| Frozen dataclass | `GeoRegion`, `StacQuery`, `LoadSource` | |
| TypedDict | G11 Sample; option bags mirroring a dependency's kwargs | not for library-internal structure |
| Named registry + factory | G5 stamps, G12 stages, processing roles, builders | only when names come from config |
| Pipeline of named steps | G5 stamps | preparation is a Prefect flow, not a library pipeline class |
| Codec pair | G3 | |
| Builder | G6 `StacItem` | |
| Strategy table | G15 decoders; G10 step-fill rule if needed | |
| Protocol | processing roles; G4 catalog sources (suggested) | |
| ABC | `GeoRaster` only | none for Datasets or tasks: torch and Lightning define them |
| Composite | `xr.DataTree` as publish layout | no GeoSave tree class |

Set aside, with reasons: a virtual cube over the catalog (second laziness
mechanism); `select_times`/`select_items` (patches for a time axis holding
every acquisition); a tile server for the explorer; a recipe executor.
Calendar compositing exists as a raster operation; selecting a training sample
from several source observations remains project policy.

## 6. Remaining work

Historical W1–W13 labels in older discussion refer to the following areas.
Publication, row Dataset loading and chip layout/merging are implemented;
their obsolete implementation checklists have been retired.

| Area | Remaining work |
| --- | --- |
| W1–W3: publication/load | Item identity, loss boundaries and native-grid return shape; see publish-load |
| W4: dense chips | streaming accumulation and other output kinds |
| W5–W7: samples/store | task-owned sample contracts, sample rows, record identity and invalidation |
| W8: release | processor, named inputs and context implemented; open: lineage records and differing normalization per input |
| W9–W10: prediction | threshold calibration, vector outputs, streaming output and catalog orchestration |
| W11: templates | implement placeholder Datasets and align flat/nested sample contracts |
| W12: Explorer | catalog pixel reads, reading/display ownership, time series tools |
| W13: architecture | decide and test dependency direction inside geodata |

## 7. Code still on old designs

Source templates now use preprocessors, but their integration and data adapters
still need work. Example notebooks 1, 3 and 4 reference removed model/chip
APIs; the notebook suite has not been rerun for this documentation review.
`workspace/gedi-biomass` still calls the removed `gs.Chips`. See
[open decisions](../open-decisions.md#template-and-notebook-migrations).

## 8. Findings kept from earlier work

Historical prototype observations; timings and removed-API measurements below
are not performance claims for the current implementation.

| Finding | Consequence |
| --- | --- |
| `Dataset.load()` computes in place | docs and templates use `.compute()` |
| Collections mixed in one parquet union every asset key | one stac-geoparquet file per collection; read the directory |
| DataLoader (forkserver) and LitData (spawn) re-import the script | Datasets live in modules; nothing runs at import |
| A 200k-row catalog pickles to 302 MB | workers open the catalog by path |
| Dask `chunks={}` was 3–6× slower than crop-then-load for view-sized reads of Sentinel-2 | view and sample reads use `chunks=None` |
| `expand` of 100 Items × 6 assets: 0.8 s, 702 nodes; `combine`: 0.5 s | the tree is fine as a layout, unnecessary for reading |
| Per-sample read, 4 steps × 2 bands + label on another grid, local COGs: about 100 ms either way | no virtual cube needed |

## 9. Questions to settle first

1. Item identity: carried as a coordinate, formatted, or both (G6)?
2. How should context change after geometric augmentation (G14)?
3. How should different pixel inputs select different normalization (G14)?
4. How a time step is filled when several Items fall in it, and whether the
   release states it (G10)?
5. Should publication accept scalar-time or timeless rasters (G6)? Static
   variables currently repeat in every Item.
6. The declared direction inside `geodata` (G16)?
