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
| [3 · Fine-tune Prithvi on image chips](03_inputs.ipynb) | How does Prithvi perform on held-out scenes? | Whole-scene test metrics, a reloadable model and its class legend. |
| [4 · Map land cover with the trained model](04_pipeline.ipynb) | How do I turn the released model into maps of a new place and season? | Your own four-stage pipeline, dated forest maps for the 2019 and 2023 dry seasons, and each 2023 map as a cloud-optimized GeoTIFF with a STAC Item. |

Notebooks 0, 1 and 3 share the same teaching-data download. Notebook 2 runs
independently. Run notebook 3 before notebook 4 to create the model release.
Start a fresh kernel for each lesson.

The **Explore further** sections suggest ways to extend the project, such as
reviewing difficult classes or expanding the observation period.

## Before you start

From a repository checkout:

```bash
uv sync
```

Open the notebooks in Jupyter or VS Code, select the repository `.venv` as the
kernel, and use `examples/` as the working directory. Run cells top to bottom.
The first training notebook downloads about 840 MB. Notebook 3 also downloads
about 130 MB of pretrained encoder weights on its first run. The training runs
are sized for a CPU: each uses part of every split and a few epochs, and takes a
few minutes. With a GPU, use every scene and more epochs; the comments beside
each setting say what to change. Notebooks 2 and 4 need network access to Microsoft
Planetary Computer for imagery over Jakarta and the KHDTK forest, respectively.

Plots and interactive maps render in the notebook. The optional TensorBoard
cells show how to compare runs. Each training run gets its own log directory.
Rerunning lesson 3 updates the model release, and the map writers replace
their previous output at the same path.

## The data and what it means

The pinned subset pairs Sentinel-2 L2A imagery with expert land-cover
annotations. Its nine-band images contain stored digital numbers. The models
use blue, green, red and near infrared (B02, B03, B04, B08) in that order.
The label TIFFs use codes 0 for unlabelled, 1–9 for land cover and 10 for cloud.
The `dynamic_world` lookup in `labels.yaml` maps those codes to training IDs
0–8, with 255 ignored. Starting from the images in each split leaves out the
four annotations without matching imagery. Test image filenames pair with
labels ending in `_consensus.tif`.
[Pinned dataset card](https://huggingface.co/datasets/fatmur/dynamicworld-300/blob/99e1222862a8100eecd8193210cb09b1b61b6c24/README.md) · [Original annotations](https://doi.org/10.1594/PANGAEA.933475).

Notebooks 0 and 1 use the first 64 of 198 expert training pairs and 16 of 49
validation pairs. Both use the same class lookup and fill missing image values with zero.
In notebook 3, `DynamicWorld(folder, count)` reads the first scenes of a split
folder as lazy image and label pairs. Prithvi reads 128 × 128 chips:
`RandomChips` draws several fresh windows from each training scene per epoch,
crops the label to the same grid and yields `(pixels, context, target)`, the
context being the chip's centre and acquisition date. The DataModule builds
each split in `setup`.

Validation and test take one whole scene per step. `Segmenter.predict_scene`
cuts it with `RasterChipper` into chips that overlap by a quarter, evaluates 16
per model call, and `RasterMerger` blends their logits on the scene grid with a
Hann window, so each labeled pixel is scored once. The same method predicts the
validation map. The trained chain is released with its `ModelHubMixin`
`save_pretrained`, beside the label scheme.

Notebook 4 builds the mapping pipeline one stage at a time. An ingestor
reads Sentinel-2 over the forest as a lazy tree whose `image` node feeds the
model. A `Preprocessor` decodes and fills the four bands and makes one window
per acquisition day. `TorchPredictor` loads the release and cuts each window
into chips; its layout records where each chip sits and when its window was
observed. `SegmentationPostprocessor` merges the chip scores into dated class
maps. The notebook checks those maps against notebook 3's scene predictor,
stages a run in Zarr for regions larger than memory, wraps the four calls in
its own `LandCoverPipeline` class and publishes the 2023 maps.

The notebooks use `original_class_names` and `original_class_colors` for the
stored annotation codes. After remapping, `class_names` and `class_colors`
map training IDs to names and colors in the same order as the model outputs.
Notebook 3 saves the label scheme beside the weights so notebook 4 can attach
the same legend to each map.

Keep the train/validation/test splits fixed before cutting chips. Compare
experiments on validation data and reserve the test split for final evaluation.
The forest observation has no reference labels. Its map lets you inspect how
the trained model behaves on new imagery.

## Files that connect the lessons

| Folder | What it carries |
| --- | --- |
| `data/dynamic_world/` | `labels.yaml` and prepared images, labels and SCL in `processed/{train,val,test}/` |
| `data/jakarta/ndvi.tif` | Notebook 2’s vegetation index |
| `data/landcover_release/` | Notebook 3’s ModelChain configuration, weights and label scheme, loaded by notebook 4 |
| `data/khdtk_run/`, `data/khdtk_2023/` | Notebook 4's staged runs: `sources.zarr` and the merged maps under `product/` |
| `data/khdtk_maps/` | Notebook 4's dated forest maps, each a GeoTIFF with a STAC Item |
| `lightning_logs/` | Training curves for each notebook and checkpoints for notebooks 0 and 1 |

## Where the project goes next

To move this experiment into a shared workspace, keep the training code,
data revision and released model linked so a colleague can reproduce the
map. [Model development workflow](../docs/mlops/index.md).
