# Examples

Start with a small task, inspect its output, then adapt it to your own data.
The first examples work offline after downloading their small sample files.

## Small complete tasks

| Example | You produce | What you need |
| --- | --- | --- |
| [Your first vegetation map](basics/first-raster.md) | NDVI plot and a GeoTIFF | Base package; 752 KiB image. |
| [Images and labels to chips](basics/training-chips.md) | Image/label plots and named tensors | ML extra; about 520 KiB for the pair. |
| [Run a training workspace](advanced/training.md) | Run configuration and checkpoint | ML extra; supplied Lightning boilerplate. |
| [Compare vegetation by area](advanced/vegetation-zones.md) | Zone summary, CSV and raster | Base package; the vegetation-map sample. |
| [Read overlapping training chips](advanced/overlapping-chips.md) | Aligned image/label samples and a plot | ML extra; the training pair. |
| [Store imagery as a catalog](basics/catalog.md) | Named COG assets and an observation index | Base package; the vegetation-map sample. |

![Satellite image and expert labels used in the real-data examples](assets/land-cover-labels.png)

*Reference annotations, not model predictions. Tait et al. (2021),
[Dynamic World training dataset](https://doi.org/10.1594/PANGAEA.933475), CC BY 4.0.
Contains modified Copernicus Sentinel data (2019).*

## Real land-cover project

The notebooks develop one land-cover project with a roughly 435 MB
[teaching dataset](https://huggingface.co/datasets/fatmur/dynamicworld-300).
It covers data inspection, model choices, acquisition, training and reuse.


| Notebook | Focus |
| --- | --- |
| [0 · Basics](https://github.com/weedkat/geosave/blob/main/examples/00_basics.ipynb) | Inspect image/label pairs and train a baseline. |
| [1 · Model](https://github.com/weedkat/geosave/blob/main/examples/01_model.ipynb) | Compose model stages and compare runs. |
| [2 · Find a Sentinel-2 Item](https://github.com/weedkat/geosave/blob/main/examples/02_sources.ipynb) | Query Sentinel-2 over Jakarta, crop one Item, and compare natural color with NDVI, and save a GeoTIFF. |
| [3 · Fine-tune Prithvi on image chips](https://github.com/weedkat/geosave/blob/main/examples/03_inputs.ipynb) | Train on random crops, predict up to 16 chips per batch, and merge logits on each image’s grid. |
| [4 · Build a land-cover pipeline](https://github.com/weedkat/geosave/blob/main/examples/04_pipeline.ipynb) | Build a pipeline step by step, move the experiment's predictor into the library, release it and map a forest observation. |

Run notebooks from `examples/` with a fresh kernel and the development
environment from `uv sync`. Lessons 0 and 1 compare models on the same tiles.
Lesson 2 saves a Jakarta NDVI GeoTIFF. Lesson 3 uses a Lightning DataModule
for the training and evaluation splits and defines its predictor inline. Lesson 4
restores the training checkpoint, checks the library's `DensePredictor` against that
predictor and releases a pipeline of library steps.

These short runs teach a workflow. Evaluate on independent geography or periods
before making claims about model quality or land-cover change.

## Organize your project

Use [MLOps](mlops/index.md) for repository layouts and MLOps,
[Advanced](advanced/index.md) for larger workflows, and
[Contributing](contributing/index.md) for the library's direction and open work.
