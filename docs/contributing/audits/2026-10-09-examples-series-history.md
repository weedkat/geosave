# Historical examples-series design and experiments

> Archived on 2026-10-16. The [current examples plan](../specs/2026-10-09-examples-series-design.md) replaces this roadmap.
> Historical design and experiment notes. The original numbering and API
> changes below are superseded in part. The current files are `00_basics`,
> `01_model`, `02_sources`, and `03_inputs`; see [the examples guide](../../examples.md)
> for the supported learning path. Remote timings and measurements below
> record their original experiments, not current performance guarantees.


Date: 2026-10-09

## Goal

Replace the feature-by-feature notebooks in `examples/` with one project taught
end to end. The task never changes: predict the Dynamic World land-cover class
of every pixel. Each notebook opens with a problem the previous one left and
adds one thing to the same project.

The series doubles as a capability test of the library. Writing each notebook
is expected to expose API weaknesses; those are fixed before the notebook is
written, not worked around in it.

## Reader

Someone who has trained an image model before and may know nothing about
geospatial data. Notebook 0 uses no geospatial vocabulary. Terms arrive only
when the project needs them.

## Rules for every notebook

- `import geosave as gs` is the one alias for data. Model names come
  from `geosave.model`.
- Around five code cells. One sentence of markdown states the problem before
  each cell. Every cell moves the project forward; a cell that only displays
  an object, or builds something the notebook never uses, is cut.
- One or two "what the library did for you" callouts, each one line, each
  smoke-tested before it is written down.
- Explain a property only when the project needs it. No tours of `crs_name`,
  `resolution` and similar for their own sake.
- The attrs models are never named. The reader sees their effect only.
- Runs offline unless the lesson is about fetching data.

## Structure

| Part | # | Title | Opens with | Teaches |
|---|---|---|---|---|
| Train | 0 | Basics (`00_basics`) | A folder of pictures and label pictures | A tile reads like any image. Tiny CNN encoder / decoder / head on whole tiles, plain Lightning, prediction beside label |
| Train | 1 | Model (`01_model`) | The loop works but is bare | `chain_step` and `Published` on the same three stages, augmentations, TensorBoard |
| Data | 2 | A raster is more than pixels | Where did those tiles come from? | Array, raster, stack. Each carries place and date. `explore()` on a real map. `write_rgb`, `write_nodata`, legends for flag data. Laziness note |
| Data | 3 | Working with rasters | RGB confuses water with shadow | Scale to reflectance, NDVI, cloud and shadow mask, choosing the band set, aligning two sources, save and reopen |
| Data | 4 | Where data comes from | Three tiles is not a dataset | STAC and why, anchor, `source.load(anchor)`, labels from the benchmark, a year of scenes |
| Data | 5 | Fitting data to a model | A year of scenes over a large area is not a model input | Tiling, time windows, composites, pretrained encoder from the registry |
| Ship | 6 | One recipe for every input | Deferred | Model spec, `prepare`, GeoSave's `Module` / `DataModule` |
| Ship | 7 | From notebook to project | Deferred | Workspace, config, CLI, release |
| Ship | 8 | Running it | Deferred | Prefect flows, remote storage, monitoring, outside RGB |

The Ship part is a placeholder. It is designed after notebooks 0 to 5 exist,
using what they taught about the API. It gets its own spec.

## Notebooks 0 to 5

### 0 · Basics (`00_basics.ipynb`)

Data: `data/dynamic_world/<split>/{images,labels}/<id>.tif`, the layout most
segmentation datasets use (see "Dataset" below). No catalog table.

1. Open one image and its label, show them side by side. Bands are named here:
   four of them, three of which the file marks as red, green and blue.
2. A plain torch `Dataset`: open both files, `.gs.to_tensor()`.
3. Label 0 (unlabelled) is `ignore_index` in the loss, as CV code does it.
3. `Encoder`, `Decoder`, `Head` as three ordinary `nn.Module`s, called in
   sequence. Convolutional, so whole tiles go in with no input size.
4. A plain `LightningModule` (cross entropy, AdamW) and `Trainer.fit`.
5. Predict one tile, plot it beside its label.

Not in this notebook: catalog, chips, STAC, spec, CRS, `chain_step`.

### 1 · Model (`01_model.ipynb`)

1. The same three stages gain `chain_step`; `Encoder` publishes its channel
   width with `Published`.
2. `ModelChain` wires them; a decoder that also reads `image` swaps in without
   glue. `build_model` widens the encoder in one place.
3. The trained model is a recipe by registered name, at notebook 0's width, so
   the flips added next are the one change between the two runs.
4. Augmentations on image and label together; compare the runs in TensorBoard.

The pretrained encoder (DINOv3) waits for notebook 5, where it is trained.

Callout: sizes published by one stage reach the next without being repeated.

### 2 · A raster is more than pixels

1. Open the four-band source of a tile from notebook 0: one band is an array,
   the bands on one grid are a raster.
2. Image and label together are a stack.
3. Place: `explore()` draws the tile on a web map.
4. Date: where a raster's date comes from.
5. Telling a raster what its numbers mean, on a label read raw from disk:
   `write_rgb` names the bands that are red, green and blue, `write_nodata`
   names the value that means "no measurement", and a legend attached to a
   flag band gives each code its class name and colour. Each is shown by
   what it changes: the plot, the statistics, the saved file.

Callouts: nothing is read until values are asked for; a label's plot takes its
class names and colours from the data.

### 3 · Working with rasters

1. Stored integers to reflectance with `mask_and_scale()`.
2. NDVI as a new band computed from two.
3. Cloud and shadow mask.
4. Choose the band set the model reads; retrain notebook 1's model on it.
5. Two rasters on different grids: `gs.tree` keeps them as they are,
   `.gs.align()` brings them onto one. Save and reopen.

Callouts: the scale is read from the data; `align()` picks the finest grid by
default. `gs.tree` does not align on its own, and the notebook says so.

As built (2026-10-10), notebook 3 (`03_bands`) covers reflectance, NDVI and
NDWI from `gs.features`, and choosing bands; retraining on six bands. Cloud
and shadow masking and aligning two sources move to notebook 4: the dataset's
four bands carry nothing a cloud detector reads, and loading from STAC is
where clouds, SCL and second sources appear.

### 4 · Where data comes from

1. What STAC is: search one collection.
2. An anchor is a place and a time span, taken from a label or a polygon.
3. `source.load(anchor)` returns imagery on the anchor's grid.
4. Ask for a year: many scenes come back.
5. Save samples and list them in a catalog table.

Needs network. Callout: the loaded image already matches the label pixel for
pixel.

### 5 · Fitting data to a model (deferred)

Deferred on 2026-10-10 by the user: tiling (chips), time windowing (frames)
and their naming come after the examples and the API they exercise. Planned
content, unchanged:

1. Tiling in space: chips as table rows.
2. Windowing in time: frames.
3. Composites: many scenes to one.
4. A pretrained encoder from the registry replaces the tiny one, now that
   inputs have a fixed size.

Open naming question for this notebook: `transform.crop.crop` (a pixel window
with halo padding, used only by `cuts.select_pixels`) shares its name with
`.gs.crop` (cut to a geographic region). Recommended: move it into `cuts` as a
private `_window`.

## Revised plan for notebooks 2 and 3 (2026-10-12)

The files are `02_inputs.ipynb` and `03_sources.ipynb`; the numbered sections
above predate them.

- **2 · Inputs** splits the work into `make_features` (raster → raster:
  reflectance plus spectral indices) and `preprocess` (raster → tensor: NaN
  filled, each band standardized with a mean and spread from 20 training
  tiles).
- **3 · Make a dataset** builds a small dataset (3 labels) in notebook 0's
  layout plus a catalog. Each step answers the problem the last one left:
  where a label's picture comes from (anchor → search → load); which date
  (the clearest by `SCL`); saving it for every label (`to_cog` +
  `create_items`); splits; feeding one pair through `make_features` →
  `preprocess` → a model instance; labels that are polygons (GeoJSON →
  `GeoAnchor.from_geometry` → `rasterize`); sharing (`to_stac`). No training.
- **Composites** move to a later notebook that combines Sentinel-1 and
  Sentinel-2.

## API changes

Each is test-driven and lands before the notebook that needs it.

| # | Change | Needed by | Tests |
|---|---|---|---|
| 1 | `.gs.to_tensor(dtype=None)` on array, raster and stack, beside `to_numpy`. `torch` is imported inside the method so `import geosave` stays torch-free. `ml/inputs.py` is deleted; `model_inputs` moves to its one caller, `ml/segmentation/supervised/data.py` | 0 | Move `tests/ml/test_inputs.py` cases to the three accessor test files; add a test that importing `geosave.geodata` does not import torch |
| 2 | `gs.cuts` and `gs.stac` on the root's lazy loader | 4, 5 | Root exposes both; importing the root loads neither |
| 3 | `geosave.model` exports `ModelChain`, `Published`, `chain_step`, `build_model` | 1 | Import test |
| 4 | `build_model` accepts a class as a stage, beside a spec dict | 1 | Published wiring through classes; `stage_specs` still round-trips |
| 5 | `.gs.explore()` on array and raster: a display copy reprojected to Web Mercator, drawn over map tiles with hvplot. Returns the HoloViews overlay | 2 | Returns an overlay holding tiles and the raster; source raster unchanged; categorical raster keeps its legend colours |
| 6 | Done: a dated raster always opens with a `time` axis | 2 | GeoTIFF read tests |
| 7 | `source.load` records the collection's scale and offset, so `mask_and_scale()` works on loaded imagery | 4 | Loaded band carries the scale; `mask_and_scale` changes its values |

Change 1 relaxes the rule "`geodata` imports no torch" to "`geodata` imports no
torch at import time". `CLAUDE.md` is updated to say so.

Change 5 smoke-tested: reproject to EPSG:3857 and overlay on OSM tiles works
with the bokeh backend and no new dependency. Tiles do not draw under the
matplotlib backend, so `explore()` selects bokeh for its own output.

## Dataset

Built by `examples/make_dataset.py` from the extracted Dynamic World release
(`train/Experts` only) and Sentinel-2 L2A on Planetary Computer:

- 300 expert tiles, at least 90% labelled and at most 1% cloud, spread over
  all 14 biomes; train, val and test hold 4:1:1, each spanning the biomes.
- Images: B02, B03, B04, B08 as uint16, scale 1e-4, nodata 0, RGB marked.
  Labels: classes 0 to 9 (Dynamic World's codes minus one), 255 unlabelled,
  names and colours in each file and in `classes.yaml` at the root.
- Every file carries its date and opens with a `time` axis of one.

Published as `fatmur/dynamicworld-300` (public, CC BY 4.0). It is both the
examples' data and a small benchmark; notebooks download it next to
themselves with `snapshot_download`.

Notebook conventions settled on 2026-10-10: short file names (`00_basics`,
`01_model`, then `02_rasters`, ...); Lightning `Trainer` with defaults;
loaders as named variables; `ignore_index=255`; metrics as a torchmetrics
`MetricCollection` cloned with a prefix, as `supervised.Module` does; model
structure shown with the printed model and Lightning's `ModelSummary`;
TensorBoard opened in the notebook; plain matplotlib for predictions.

## Done since the first cycle

- Rasters and stacks open with `gs.open_raster` / `gs.open_tree` (were
  `read_*`); vectors keep `read_vector`, as pandas and geopandas do.
  `io.<format>.open_raster|open_vector` and `io.safe.open_tree` are unchanged.
- A dated raster always has a `time` axis, length one for one date; undated
  rasters have none (API change 6).
- `build_model` takes a class, a built module, or a stage specification.
- Opening a dated GeoTIFF stays lazy (it used to load the pixels).
- `.gs.explore()` and `.gs.write_legend()` on raster and band.

## Weaknesses found, not yet scheduled

Triage when the notebook that meets each one is written.

- A label's class names come from `label.gs.legend.classes`, but the map
  includes `255: 'unlabelled'`, so a head's class list needs a filter.
- Reading one chip takes a catalog filter, `to_tree` and `cuts.select`.
- `cuts.merge` needs a hand-built `DataArray`; its result has no band
  coordinate or legend.
- A missing `Published` source reports "missing a required argument", not
  which stage should have published it.
- Raw Sentinel bands are stored as float32 with no nodata.
- A raw label reads as `band_1` with no class names; shipping the legend with
  the example label files avoids an API change.
- `raster.gs.to_cog("x.tif")` refuses a file name; writing one file needs
  `gs.io.geotiff.write_cog`.
- Every plot asks Nominatim for a place caption. It degrades to no caption
  offline, but a plot reaching the network is a surprise.
- Lightning's tips and warnings fill the training cell's output.
- A chain holding a stage given as a built module has no recipe, so it
  cannot be exported as a release; the error names the stages.
- `ImageAugmenter` wants a float mask with a channel axis, so labels are
  reshaped going in and coming out, and it requires `size` even for flips.
  Agreed 2026-10-12: replace it with Kornia transforms built by LightningCLI
  `class_path`, data keys fixed at construction, `keepdim=True`, augmentation
  in `DataModule.on_after_batch_transfer`, and `valid` moved as a second mask
  (pass it as int64: Kornia casts every mask to the last mask's dtype). Needs
  a plan with tests. The notebooks already use plain Kornia.
- A model's repr prints the data flow followed by every layer; there is no
  short view of the flow alone.
- Importing the library logs a `gcsfs experimental features` INFO line.
- `.gs.explore()` works but its UI looks poor (user, 2026-10-11); not yet
  diagnosed.
- `to_cog("name")` writes an undated raster as `name.tif` but a dated one as
  `name/name_<timestamp>.tif`.
- `NotGeoreferencedWarning` from rasterio's in-memory warp appeared twice
  during `reproject` and could not be reproduced since; silence it only with
  a reproducing test. Seen again 2026-10-12 in notebook 2's `gs.explore` of a
  saved COG and notebook 3's `month.load`.
- `to_tree` on a remote frame opens every data asset's header before any
  pixel is read: about 38 s cold for 4 Sentinel-2 items (16 assets each),
  cached afterwards.
- The interpreter, pytest included, prints "Error in sys.excepthook" lines
  at exit; not yet diagnosed.

## Fixed during the notebook 4 smoke test (2026-10-12)

- Variables read through STAC items are named by asset key: a one-band asset
  is its key, band `i` of a multi-band asset `key.i` (odc-stac's notation).
  `eo:bands` facts become attrs (`band_attrs`). Reading promises band names,
  not their order; models select bands by name. GeoSave's own writers
  (`ModelSpec.prepare`) write one file per band so names survive.
- `cogs.open_raster` names bands from the catalog record (Planetary Computer's
  files have no band descriptions).
- `merge_bands` ignores the CRS coordinate's placeholder value (0 from
  rioxarray, the EPSG code from odc-geo).
- `open_raster` detects the format of a signed URL from its path.
- `create_item(s)` take `collection="label"`.
- A read warns once about bands lacking a fill, not once per band per chip.
- `chunks={}` opens a tiled GeoTIFF in its own tile blocks: one remote chip
  went from 760 s to 24 s (4 s warm).
- `merge_bands` spells x/y once from the shared grid (`write_cf_axes`):
  sliced bands came bare from rioxarray, warped ones with odc-geo's CF attrs,
  and every merge warned `DroppedAttrsWarning`.
- `gs.tree` takes a band (DataArray) as a one-variable group.

## Notebook 4 flow, smoke-tested 2026-10-12

```python
labels = pd.concat([gs.stac.create_item({"label": p}, id=p.stem, collection="label") for p in paths])
scenes = pd.concat([source.set_query(datetime=month, filter="eo:cloud_cover <= 20").search(anchor) for ...])
windows = gs.cuts.stacks(pd.concat([labels, scenes]), on="label")
chips = gs.cuts.chips(gs.cuts.frames(windows, 4, tolerance="10D"), 224, overlap=32)
image = chips.iloc[0].gs.to_tree()["sentinel-2-l2a"].dataset
pixels = image[["B02", "B03", "B04", "B08", "SCL"]].gs.to_tensor(dtype="float32")  # (4, 5, 224, 224)
```

Offline, the dataset's own tiles as items give 32 chips per 2 tiles and
`cuts.merge` puts chip outputs back on the 510 px grid. Open question before
writing the notebook: train on remote chips (about 4 s each warm) or save
chips locally first.

## Use cases found

Each notebook adds the user goals it meets; they are the input for the
`model_spec` and flow design after the examples.

- **Predict a raster and get a map back** (notebooks 0, 2): run a trained
  model on a tile and receive a georeferenced, legended raster to plot or save.
  Today: a batch from the loader, `argmax`, then `gs.raster(..., labels)`,
  which passes the label's legend on to a band of the same name.
- **Fetch imagery that matches my labels** (notebook 3, `03_sources`):
  `label.gs.anchor`, `source.search(anchor)` and `source.load(anchor)` return
  Planetary Computer pixels identical to the dataset's own. Planetary Computer
  publishes no `raster:bands` scale, so the notebook states it with
  `set_config(dtype="uint16", nodata=0)` and `write_packing(1e-4)`; API
  change 7 is dropped. `set_query(datetime=..., filter=...)` widens the window
  without changing the anchor. Reads are network-bound at about 3-12 s per
  scene, so the notebook composites one month (10 scenes), computed once.
- **A clear picture from a cloudy month** (notebook 3): SCL mask, median in
  reflectance, saved as a COG.

## Build order

One notebook at a time, each a full cycle:

1. Smoke-test the notebook's flow as a script.
2. Land the API changes it needs, tests first.
3. Write the notebook.
4. Add its test to `tests/examples/test_notebooks.py`: `slow` when offline,
   `integration` when it needs network.

Order: change 1, then notebook 0; changes 3 and 4, then notebook 1; changes 5
and 6, then notebook 2; notebook 3; changes 2 and 7, then notebook 4;
notebook 5.

## Existing notebooks

`00_quickstart` to `05_train` stay in place until the notebook covering their
content lands, then are removed in the same change. `examples/configs/` stays
for the Ship part. `tests/examples/test_notebooks.py` drops a test when its
notebook is removed.

## Out of scope

- The Ship part (6 to 8), workflow and Prefect design, remote storage.
- Other heads (regression, classification, detection) and the Granite biomass
  model: one "where next" list at the end of the series.
- Remote training data with LitData.
