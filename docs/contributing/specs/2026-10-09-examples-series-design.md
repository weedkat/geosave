# Examples series: one land-cover project

Updated: 2026-10-16. Notebooks 0–3 exist; notebook 4 is next. Later lessons
are planned and must be smoke-tested before their designs are settled.

This plan follows the current files in `examples/`. The
[examples guide](../../examples.md) describes the runnable lessons. Earlier
numbering, API proposals and experimental measurements are preserved in
[historical notes](../audits/2026-10-09-examples-series-history.md); they are not
requirements for the remaining lessons.

## Goal and reader

Teach one land-cover mapping project from image files to a reusable model.
The reader knows some Python and image modelling but may know no geospatial
terminology. Each lesson starts with a problem the previous one leaves and
produces a visible result using the same data and splits.

The first four notebooks establish the workflow with native xarray and
GeoPandas objects, a caller-owned torch Dataset, and ordinary Lightning.
Notebook 4 extends the input declarations and introduces GeoSave's training tasks as ways
to express work the reader already understands.

## Teaching rules

- Use `import geosave as gs` and current public APIs.
- Keep the `DynamicWorld(Dataset)` name and tensor contract across notebooks
  0, 1 and 3: `(image, label)` for training, an image for prediction.
  All three accept image paths and optional label paths. Notebook 3 builds
  a catalog, samples, frames and chips inside its constructor, and opens
  and aligns each requested chip internally. No ModelSpec in these basic lessons.
- Set up train, validation and test explicitly. Avoid dictionaries of split
  rules, generic dispatch helpers and custom orchestration in these examples.
- Show whole tables and native objects. Select columns or bands when an
  operation needs them, rather than trimming every display.
- Introduce geometry, dates, grids and metadata when they solve a concrete
  problem. Follow explanations with code and useful output.
- Use xarray directly: `image.assign(ndvi=ndvi)` adds a derived band already
  on the image's grid. Align independent rasters explicitly before combining
  their pixels.
- Save raster files before creating Items that point to them. Explain that
  asset hrefs are stored relative to the Parquet catalog, resolved when read,
  and can also be published as ordinary STAC JSON.
- Keep plain DataLoaders and the notebook's `Segmenter` through notebook 3.
  Introduce task Modules, DataModules and their sample contract in notebook 4.
- Start each notebook in a fresh kernel. Files carry forward; variables do not.
- Reuse local data after acquisition. Identify remote queries and weight
  downloads, and distinguish cached checks from live remote execution.
- Use short training runs to demonstrate behavior, validation for comparisons,
  and held-out test data for final evaluation. Estimate any normalization
  statistics from the training split only.
- Keep exercises after the completed workflow. Readable steps determine cell
  count; there is no arbitrary five-cell limit.

## Sequence

| # | File | Status | Question | Visible result |
| --- | --- | --- | --- | --- |
| 0 | `00_basics.ipynb` | Existing | Can I inspect these images and train a baseline? | Data summaries, validation curves, a prediction beside its label. |
| 1 | `01_model.ipynb` | Existing | How do I change the model while keeping the data fixed? | A composed model, augmentation and comparable training runs. |
| 2 | `02_sources.ipynb` | Existing | How do I build this dataset from labels or a geometry? | Saved image/label pairs, a season over a GeoJSON outline, NDVI and STAC catalogs. |
| 3 | `03_inputs.ipynb` | Existing | How do catalog rows become model inputs and maps? | Chips, plain Lightning training, daily scene predictions using the same Dataset, merged onto their grids and catalogued. |
| 4 | `04_recipe.ipynb` | Next | How do I save and automate the input recipe? | A saved ModelSpec, prepared catalog and a run using GeoSave's supervised task. |
| 5 | `05_pretrained.ipynb` | Planned | What changes when I use a pretrained encoder? | An explicit input contract and a controlled comparison with the small CNN. |
| 6 | `06_multisource.ipynb` | Planned | How do I combine observations from different sensors and dates? | Quality masks, temporal composites and explicitly aligned optical/radar inputs. |
| 7 | `07_workspace.ipynb` | Planned | How do I turn the working experiment into a repeatable project? | The same data recipe and model configuration run from a generated workspace. |
| 8 | `08_release.ipynb` | Planned | How can someone else load the trained model and its input recipe? | A local model release reloaded for prediction; optional Hub publication. |

## What the existing lessons establish

### 0 · Explore and train a baseline

Download the Dynamic World teaching dataset into
`data/dynamic_world/<split>/{images,labels}/`. Open and plot an image and label,
inspect reflectance and class balance, then use `DynamicWorld` and named
DataLoaders to train an encoder/decoder/head model with plain Lightning.
Put a prediction back on the label's grid with `gs.raster` and its legend.
The loss and metrics ignore label code 255.

No catalog, ModelSpec or GeoSave training task is needed here.

### 1 · Improve the model

Keep notebook 0's Dataset, splits and tensor contract. Introduce `chain_step`,
`Published`, `ModelChain`, `build_model` and registered parts through visible
changes to that model. Add joint image/label flips inside the notebook's
`Segmenter` and compare validation curves in TensorBoard.

Pretrained weights and new data sources wait for their own lessons.

### 2 · Build a dataset from STAC

A label supplies a grid and date. Search its day, load a matching scene,
save the pair, then create Items and a catalog. Build the train, validation
and test splits explicitly, combine their tables, and save
`data/my_dataset/catalog.parquet`.

Next, read a geometry from `khdtk.geojson`, create an anchor, and load a
season over that grid. Add NDVI with `assign` and explore it. Keep the
four-band imagery as the model input. Save the season in `data/khdtk/` and
show both GeoParquet and STAC JSON publication.

This lesson fetches and explores data; it does not introduce training tasks.

### 3 · Turn catalog rows into inputs and maps

Read notebook 2's catalog and collect each split's image and label TIFF
paths. Pass them to `DynamicWorld(images, labels=None)`. Its constructor
builds a catalog from existing file headers, groups by label time with
`tolerance="1D"`, frames by day and chips at 128 pixels. With no labels or an
empty label list, it prepares image-only inference. Its `table` property
returns the chip GeoDataFrame; `frames` retains the original merge grids.
`__getitem__` opens and aligns the requested row and returns tensors.
`Segmenter.__init__` builds the model, evolving the earlier external-builder
pattern. Keep ModelSpec out of this basic notebook while it is in development.

Train with ordinary DataLoaders and a `Segmenter` following notebook 1.
Evaluate the test split, predict chips over the saved geometry season, merge
the scores onto each acquisition day's grid, and save the maps before cataloguing them.
Keep input scenes in `data/khdtk/` and prediction rasters and their
GeoParquet catalog in `data/khdtk_pred/`. Show prediction, score indexing
and merging in separate cells with the familiar evaluation loop. STAC JSON
publication remains in notebook 2.

This completes the basic workflow. Frames, chips, geometry ingestion and data
publication are already taught; later lessons use them rather than
reintroduce them as missing steps.

## Remaining lessons

### 4 · One recipe for model inputs — next

Use the standard PyTorch Dataset as the boundary for ModelSpec management.
The Dataset owns the sequence of sample, frame and chip calls; its native
GeoDataFrames are preparation state retained for inspection and merging.
Keep table construction in the constructor and raster reads in `__getitem__`.
Extend notebook 3's Dataset rather than inventing a separate preparation
flow. The ModelSpec declares requirements; the Dataset uses them for both
training and inference.

Start from notebook 3's working preparation and training code. The problem is
keeping raster requirements, band selection, preprocessing and tensor inputs
consistent when the same model is trained and used over another place.

1. Show the preparation steps and the named input the existing model needs.
2. Introduce `ModelSpec` after its API is settled, including preprocessing and named inputs. Distinguish the raster/input
   recipe from the model architecture built with `build_model`.
3. Compare a manually prepared sample with the declared preparation. Check
   band order, tensor values, nodata and the resulting grid before proceeding.
4. Use `prepare` to save the prepared rasters, catalog and spec. Reopen them
   and show what was preserved.
5. Introduce the supervised task's `Module` and `DataModule`, showing which
   training behavior they take over. Make the Dataset contract change
   explicit: `(image, label)` becomes a `Sample` containing named `inputs`
   and `target`. Do not hide this change behind an adapter.
6. Train briefly and predict over the existing geometry with the same input
   recipe. Save the spec and a checkpoint for subsequent lessons.

Keep the same optical data, splits and class definitions. Add no pretrained
encoder, second sensor, workspace or deployment flow in this lesson.

### 5 · Use a pretrained encoder

Reuse notebook 4's prepared data and task setup. Select one supported
pretrained encoder and make its required band order, spatial size and
normalization explicit. Select the needed bands from the existing imagery,
adapt the input recipe, and inspect a forward pass before training.

Compare with the small CNN using the same geographic splits and a stated
training budget. Show validation metrics and maps; a short teaching run is
not a benchmark. Weight download is the new external dependency, with later
runs using its cache.

### 6 · Combine sources and dates

Extend the dataset-building workflow with optical quality information and
Sentinel-1 observations over the same places. Inspect each source before
combining it: cloud/shadow validity, acquisition dates, variable names,
units and native grids.

Show quality masking, a temporal composite and calendar pairing as explicit
operations. Keep sources in separate DataTree groups until a shared grid is
needed, then align explicitly. Display the result, save and catalogue the
new inputs, and update the model's input recipe. Combine optical and radar
into a model input only after their band and normalization choices are
visible. Compare against the optical-only run on the same splits.

### 7 · Move the experiment into a workspace

Start from a working notebook configuration. Generate a workspace, place the
Dataset code with the task, and carry the existing ModelSpec, model recipe
and training settings into its configuration. Run the same experiment through
LightningCLI, inspect its logs, and demonstrate checkpoint resume.

The workspace composes GeoSave and Lightning; it does not introduce a second
implementation of preparation or training. No Prefect flow is needed here.

### 8 · Save and load a model release

Distinguish a checkpoint for resuming training, a catalog of raster outputs,
and a model release containing architecture, trained weights and an input
recipe. Use registered model parts from the preceding lessons.

Save a release locally, reload it in a fresh process, and compare its prediction
with the trained model on the same input. Then prepare and predict another
place with the loaded recipe, preserving the output grid and class legend.
Make Hub upload an optional, explicitly authorized final step; local export
and reload complete the lesson without publishing to an account.

## Later directions

Prefect orchestration, remote training data, serving and monitoring need their
own use cases and designs after the local workflow is complete. They have no
assigned notebook numbers yet. Other prediction tasks and large benchmark
studies are separate learning paths; this series remains about land cover.

## Implementation and verification order

Build notebook 4 next, then revisit each later outline using what that lesson
reveals. The roadmap does not authorize implementing all remaining notebooks
at once.

For each lesson:

1. Inspect the current public APIs and the notebook it builds on.
2. Smoke-test the complete flow as a small script before settling its design.
3. Fix any necessary library defect in its owning module, with focused tests;
   do not put workarounds into the teaching code.
4. Write the standalone notebook with visible outputs and exercises.
5. Execute its teaching cells in a fresh kernel, check its saved artifacts,
   and add or update its test in `tests/examples/test_notebooks.py`.
6. Validate notebook structure, formatting, and links from the examples guide
   and README. Mark a lesson existing only when its file and checks exist.

Exercise persistence and lazy-execution promises used by the lesson. Record
whether verification used cached data or live services. Keep completed lessons
stable when adding advanced task APIs; update their descriptions only when
their actual behavior changes.
