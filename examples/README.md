# Map land cover, one project at a time

Start with labeled satellite images. Train a baseline, investigate a different
model, acquire imagery for your own forest, and turn predictions into maps.
Each notebook tackles the next problem in that project and leaves files or
results you can use in the next lesson.

## Follow the project

| Lesson | The problem you solve | What you leave with |
| --- | --- | --- |
| [0 · Explore and train](00_basics.ipynb) | Can these images and labels teach a model to recognize land cover? | A baseline prediction and training logs; optional per-class review. |
| [1 · Compare models](01_model.ipynb) | What should I change when the baseline misses things? | A second training run and concrete experiments to compare. |
| [2 · Find a Sentinel-2 Item](02_sources.ipynb) | How do I find an observation and read just my region? | A Jakarta NDVI GeoTIFF on the source image grid. |
| [3 · Fine-tune Prithvi on image chips](03_inputs.ipynb) | How does Prithvi perform on held-out scenes? | Test metrics, a training checkpoint and a project predictor. |
| [4 · Build a land-cover pipeline](04_pipeline.ipynb) | How do I turn the trained model into maps someone else can make? | A released pipeline and a dated forest map. |

Notebooks 1 and 3 use the teaching data from notebook 0. Notebook 2 runs independently.
Notebook 4 restores notebook 3’s checkpoint and repeats its predictor definition before replacing it with the library's. Start a fresh kernel for
each lesson: files carry forward, Python variables do not.

The **Explore further** sections suggest ways to extend the project, such as
reviewing difficult classes or expanding the observation period.

## Before you start

From a repository checkout:

```bash
uv sync
```

Open the notebooks in Jupyter or VS Code, select the repository `.venv` as the
kernel, and use `examples/` as the working directory. Run cells top to bottom.
The teaching-data download is about 400 MB. CPU training is supported; the short
runs demonstrate the workflow rather than produce a validated operational map.
Notebook 2 searches over Jakarta and saves NDVI as a GeoTIFF. Notebook 4 finds the forest's least cloudy August observation and reads it with the library's Sentinel-2 ingestor.

Plots render in the notebook. TensorBoard and the Explorer server are optional
snippets you can run separately. Rerunning training starts a fresh run unless
you explicitly supply a checkpoint. Use a different logger version to keep
experiments separate. Cells with `overwrite=True` replace their example outputs.

## The data and what it means

The pinned teaching subset pairs Sentinel-2 imagery with **expert-annotated
Dynamic World human annotations**, not the automated Dynamic World product.
Labels keep their original classes (0 unlabelled, 1–9 land cover, 10 cloud); the
notebooks remap them in memory with the `dynamic_world` scheme in `labels.yaml`
to nine classes, with 255 ignored. Images hold B02, B03, B04, B08 as stored
digital numbers; each scene's SCL is a separate file.
`scene.parquet` lists every annotation with its split, paths and date; several
annotations can share one image. Each notebook links the dataset card and
original source; retain their attribution when sharing derived work.

Notebooks 0 and 1 use the same eight expert training pairs and two validation
pairs. Both use the same class lookup and fill missing image values with zero.
In notebook 3, `DynamicWorld` returns a whole lazy image Dataset and its
remapped label DataArray. `DynamicWorldTrain.read_sample` draws a random 128 × 128
window with `gs.random_crop` and reads the mask on that grid. Its
`__getitem__` converts the crop and mask to tensors.
The training loader stacks two crops per batch. Validation and test yield
one xarray pair at a time. `Segmenter` passes the image to `LandCoverPredictor`
and scores the merged logits against the mask, counting each labeled pixel once.

The predictor uses `RasterChipper` to read overlapping windows, passes up to
16 chips per call to Prithvi and blends their logits on the original grid with
`RasterMerger`. The merger holds one scene’s class logits in host memory.
Prithvi EO 2.0 Tiny TL uses each chip’s center and acquisition date. The first
run downloads about 130 MB of encoder weights. Notebook 4 builds a pipeline
step by step: `DensePreprocessor` prepares the bands as notebook 3's Dataset
did, the predictor merges chip logits and `SegmentationPostprocessor` decodes
them into a dated class map. It then swaps in `DensePredictor`, the library's
version of the same predictor, so the saved release names library steps only,
and applies the reloaded pipeline to forest imagery.

The notebooks use `original_class_names` and `original_class_colors` for the
stored annotation codes. After remapping, `class_names` and `class_colors`
map training IDs to names and colors in the same order as the model outputs.
Notebook 4 uses that same scheme for the postprocessor, so the saved pipeline
attaches the training legend to each map it produces.

Keep the train/validation/test splits fixed before cutting chips. Inspect
geographic overlap before claiming generalization. Compare experiments on
validation data and reserve the test split for final evaluation. The forest observation has no reference labels. Use its map to inspect how
the trained pipeline behaves on new imagery.

## Files that connect the lessons

| Folder | What it carries |
| --- | --- |
| `data/dynamic_world/` | Downloaded teaching images, labels, `scene.parquet` and `labels.yaml` |
| `data/jakarta/ndvi.tif` | Notebook 2’s vegetation index |
| `data/landcover_model.pt` | Notebook 3's PyTorch model checkpoint, restored in notebook 4 |
| `data/landcover_pipeline/` | Notebook 4's release of the model with its pipeline |
| `data/khdtk_maps/` | Notebook 4's dated forest map as a GeoTIFF and STAC Item |
| `lightning_logs/` | Training curves for each notebook and checkpoints for notebooks 0 and 1 |

`gs.RasterChipper` reads windows on demand;
merging predictions allocates full output maps. Prepare larger projects in batches.

## Where the project goes next

The [series plan](../docs/contributing/specs/2026-10-09-examples-series-design.md)
continues with improving this model using pretrained weights, handling cloudy
months with additional observations, moving the experiment into a workspace,
and handing the trained model to a colleague. These later lessons are planned,
not yet runnable notebooks. Each will keep the project question, data, and
comparison visible instead of teaching a library subsystem in isolation.
