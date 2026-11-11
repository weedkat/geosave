# Examples series: one land-cover project

Updated: 2026-11-06. This is the active teaching plan for `examples/`.
The [examples guide](../../examples.md) describes the runnable lessons.

## Goal and reader

Teach one land-cover mapping project from labeled images to a model another
person can use. The reader knows some Python and image modelling but may know
no geospatial terminology. Each lesson starts with a problem the preceding
lesson leaves and produces a visible result using the same data and splits.

Library coverage is an authoring check, not the table of contents. A notebook
about storage, vectors, or chips in isolation is not a continuation of this
project. Introduce each operation when the project needs it: save an image to
reuse it, align a label so training is meaningful, merge predictions to map a
forest, preserve a recipe so a later observation receives the same preparation.

## Teaching rules

- Use real project inputs and current public APIs. Native xarray and GeoPandas
  objects remain visible; a project-owned Dataset supplies ordinary tensors.
- Start with the question, the files carried forward, and the result the reader
  will make. Explain the next decision briefly, then show code and its output.
- Retain the same geographic splits, band order, class IDs, and missing-pixel
  policy unless the lesson explicitly makes one of them the experiment.
- Introduce one substantial project problem at a time. Do not add a different
  dataset just to demonstrate another feature.
- Make each step easy to follow. Inspect and convert one prepared image/label pair
  before introducing the Dataset that reads it. Keep split selection visible before
  constructing loaders. Use intermediate values and explicit loops; some
  repetition is preferable to an abstraction that obscures the work.
- Separate meaningful steps with blank lines inside cells, functions, and loops.
  Give distinct responsibilities their own cells with connecting prose. Judge
  simplicity by whether readers can follow the work, not by line count.
- Keep acquisition, alignment, and missing-data preparation outside training
  Datasets. Use prepared inputs in the basics lesson. Scale imagery, never class
  IDs. Introduce preparation when acquired data makes it necessary, rather than
  adding defensive checks and repair modes to every sample read.
- Pass changing inputs and configuration into functions explicitly. Imports,
  other definitions, and genuine project constants may be referenced directly;
  a client, query, collection, or selected band list must not be captured from
  unrelated notebook cells. Do not disguise changing values as constants.
- Keep one route to a completed project result in the main lesson. Alternative
  decoders, model widths, model discovery, and NDVI review belong in optional
  extensions with a stated reason to try them.
- Start in a fresh kernel; files carry forward, variables do not. Keep setup
  explicit rather than executing or extracting another notebook's cells.
- Keep acquisition, sample grouping, alignment, tensor conversion, model
  execution, and output interpretation explicit. Do not hide these in a generic
  framework or infer a grid from matching array shapes.
- Use short runs to demonstrate a workflow. Compare models on validation data;
  reserve held-out test data for final evaluation. Label cached checks and live
  remote checks separately.
- Finish the workflow before **More things you can do**. Supplements suggest
  plausible extensions of this project and may include a short runnable example.
  No quizzes, difficulty ladders, hidden solutions, or empty answer cells.
- Browser/server tools are optional. Run All must not launch a desktop browser
  or start TensorBoard merely to complete the lesson.

## Current sequence

| # | File | Project question | Result carried forward |
| --- | --- | --- | --- |
| 0 | `00_basics.ipynb` | Can I learn land cover from these labeled images? | Baseline predictions and training logs; optional per-class review. |
| 1 | `01_model.ipynb` | Which model changes are worth investigating on these same tiles? | A composed model, paired augmentation, and a second validation run. |
| 2 | `02_sources.ipynb` | How do I obtain the right imagery for my labels and my forest? | Aligned image/label pairs with fixed splits, six forest observations, and catalogs. |
| 3 | `03_inputs.ipynb` | How do I train on those pairs and make complete forest maps? | Merged dated predictions, held-out evaluation, and the trained architecture/weights/classes. |
| 4 | `04_recipe.ipynb` | How do I map another observation without changing what the model's inputs mean? | A saved preparation recipe, equivalence against the existing inputs and scores, and a georeferenced review image/map pair. |

### What 0–3 already establish

Notebook 0 uses eight complete training pairs and two validation pairs from the
pinned Dynamic World subset, ordinary Dataset and DataLoader objects, and plain
Lightning. Labels are expert annotations with ten
classes including cloud and nodata 255; they are not the automated Dynamic World
product. Predictions use native xarray coordinates and the label legend.

Notebook 1 keeps the data fixed while introducing model composition and paired
augmentation. Its changes motivate controlled follow-up experiments, not claims
that one short run proves a better architecture.

Notebook 2 searches with `GeoRegion`, loads selected Items onto a native
`GeoBox`, and writes paired rasters and STAC GeoParquet. `to_catalog` writes the
assets it describes. The acquisition and metadata workflow is already taught;
later lessons reuse the files rather than repeating remote searches.

Notebook 3 groups catalog rows with `sample(collections=..., freq="D")` and opens
native DataTrees. It prepares imagery and categorical labels before constructing
matching chips. Its Dataset only returns tensor pairs. Non-overlapping evaluation
chips count each original pixel once, with padded labels ignored. Forest inference
uses overlapping chips and merges their keyed scores outside the training classes.
The caller restores spatial coordinates, dates, and observation masks. `gs.Chips`
reads windows on demand; the notebook explicitly loads its small source collection.

Its handoff in `data/landcover/` contains `model.yaml`, `weights.pt`, and
`classes.yaml`. These are the actual trained chain, not a new random model.
Lightning checkpoints remain separate artifacts for resuming training.

### 4 · Keep the preparation with the trained model

The motivating problem is input drift when the model is reused: weights alone
cannot recover band order, scaling, or missing-pixel handling. This lesson
records notebook 3's existing preparation using the **current** ModelSpec v3;
it does not redesign ModelSpec or introduce a library training abstraction.

1. Reload notebook 3's architecture, trained weights, and class definitions.
2. Select a 128-pixel validation window and show the familiar manual preparation.
3. Save the same preparation as a recipe. Keep decoded reflectance without
   adding new normalization to an already-trained model.
4. Deliberately reorder source variables, then compare the recipe's tensor and
   scores with the manual path. Preserve missing-pixel masks separately.
5. Reload the recipe and make a prediction for a review window in the latest
   saved forest observation. Keep its grid, date, class IDs, legend, and nodata.
6. Export the input crop and prediction together for inspection in a GIS.

This lesson uses a small review window to establish equivalence. It does not
replace notebook 3's full-scene chipping/merging workflow or claim new accuracy.
The project still owns file reads, observation selection, and augmentation.
Recipe execution calls installed Python operations; reviewed local recipes are
used throughout. Publication of arbitrary processing code remains unresolved.

## Planned continuations

These are project outlines, not implemented notebooks. Inspect and smoke-test
one complete flow before settling each lesson. Keep numbering provisional until
its prerequisite artifacts and current APIs have been verified.

### 5 · Improve the forest model with pretrained knowledge

Reuse the geographic splits and compare against the existing small CNN. Select
one supported encoder whose weights can actually be obtained and whose input
contract is known. The current TIFFs contain B02/B03/B04/B08: a six-band encoder
requires acquiring the missing bands, not pretending the four-band files fit.

Make the change in band order, normalization, spatial size, or context explicit
in a new recipe. Inspect a real forward pass, train for a stated budget, and
compare class metrics and the same validation maps. Keep weight-download cost
and access requirements visible. Save the trained run and recipe, not just an
architecture showcase.

### 6 · Make forest mapping more reliable through cloudy months

Start from dates where the optical imagery limits the project's map. Acquire
quality information and, if it answers the project question, radar over the
same places. Inspect dates, units, quality flags, and native grids first.

Apply quality masks before temporal reduction, retain valid-observation counts,
and align continuous and categorical layers with appropriate resampling.
Respect the labels' temporal meaning when using composites. Compare the new
inputs against the optical-only experiment on the same splits. The outcome is
a supported seasonal mapping experiment, not a tour of temporal operations.

### 7 · Rerun the experiment outside a notebook

Move a working experiment into an editable generated workspace, preserving its
actual Dataset behavior, recipe, model configuration, and evaluation protocol.
Current templates live under `modules/<task>/<method>/`; there is no `ml.tasks`
package. Explain any change from tuple batches to named `Sample` dictionaries
and test preparation equivalence before training.

Run through LightningCLI, inspect saved effective configuration and logs, and
resume a checkpoint deliberately. A template's default pretrained model or
normalization must not silently replace this experiment. No Prefect orchestration
is needed to demonstrate a repeatable local training project.

### 8 · Hand the trained forest model to a colleague

Package the actual trained registered chain, input recipe, class meaning, and
reference case. Reload in a separate process and reproduce the reference scores,
then map another saved observation while retaining output georeferencing.

Explain what the recipient needs installed and what the release does not include
(training data, optimizer state, arbitrary Python implementations). Local export
and reload complete the project. Hub publication is an optional explicit action,
not a cell that runs during normal notebook execution.

## Changes from the October plan

- Replace `GeoAnchor`, `gs.raster`, and the old sample/frame/chip table workflow
  with the current `GeoRegion`, native xarray, row grouping, and NumPy Chips APIs.
- Remove the proposed library-owned supervised task and hidden `Sample` adapter.
  Training implementations belong to generated projects, and ModelSpec redesign
  remains on hold. Notebook 4 uses existing v3 behavior only.
- Do not require `prepare` to persist arbitrary intermediate processing state.
  The project saves deliberate raster outputs, recipe files, and model weights.
- Remove exercises; supplements now extend the completed land-cover project.
- Reject the six standalone feature notebooks added during the first revision.
  Their replacement follows this sequence rather than renaming those notebooks.

## Implementation and verification

- [x] Remove the standalone additions and their test registrations and links.
- [x] Keep useful 0–3 API corrections; rewrite supplements around the project.
- [x] Smoke-test the saved recipe against real cached validation pixels and the
  trained notebook-3 checkpoint before writing notebook 4.
- [x] Add the trained-model handoff to notebook 3 and write notebook 4.
- [x] Execute notebook 4 in a fresh kernel with the actual trained handoff.
- [x] Verify input/score equivalence, then reopen the image and prediction to
  check pixels, grid, dates, class legend, and nodata.
- [x] Run the scoped notebook regression, formatting, structure, and link checks.

Further coverage should arise from these project decisions. Prefect scheduling,
remote training data, serving, and monitoring need concrete use cases after the
local project is complete; they are not independent feature lessons to append
just to cover exports.

### Verification for this revision

Notebooks 0, 1, 3, and 4 completed in fresh kernels using cached source data
and isolated output folders. These runs included training, evaluation, forest
mapping, and the notebook-3 handoff consumed by notebook 4. Notebook 0's later
revision uses eight complete training pairs and two validation pairs, also used
by notebook 1. Notebook 3 uses separate preparation, non-overlapping evaluation
chips, and an explicit forest prediction loop. No fresh acquisition was claimed.

Input tensors and model scores matched the manual path. Reopened review TIFFs
retained the exact source pixels and packing, matching grid/date, valid class
IDs, legend, and nodata. The local catalog/chip and paired-input regressions
passed (5 passed, 5 slow tests deselected), including image-only prediction and
helper execution without notebook configuration globals. All five notebooks passed structure,
syntax, and Ruff check/format checks; the scoped diff-whitespace check passed.
Full live notebook-2 acquisition and the complete integration suite were not
rerun.
