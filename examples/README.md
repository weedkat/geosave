# Land-cover mapping examples

Run the notebooks in order. Use the repository environment (`uv sync`) as the
kernel and `examples/` as the working directory. Start each notebook in a fresh
kernel; files carry forward between lessons.

| Notebook | What you do |
| --- | --- |
| [0 · Explore and train](00_basics.ipynb) | Inspect images and labels, then train a small model. |
| [1 · Change the model](01_model.ipynb) | Swap model parts and compare training runs. |
| [2 · Build a dataset](02_sources.ipynb) | Fetch images, save image/label pairs, and create catalogs. |
| [3 · Train and predict](03_inputs.ipynb) | Make chips, train, predict without labels, and save maps. |

The first download is about 400 MB. Notebook 2 also needs Planetary Computer
access. Later runs reuse local files. Maps and TensorBoard need a browser.

## From files to model inputs

Every notebook passes TIFF paths to the `DynamicWorld` Dataset. With labels,
it returns `(image, label)` tensors. Without labels, it returns an image tensor.
Notebook 3 builds its catalog, frames and chips inside Dataset initialization;
callers continue to pass image and label paths.

Notebook 3 downloads the dataset at the beginning. Its Lightning DataModule
creates the split Datasets in `setup()` and supplies DataLoaders to Trainer.
The original training splits are preserved. `preprocess()` masks, scales and converts one chip
to tensors when requested.

The Dataset prepares its chip index using these operations:

```python
samples = catalog.gs.sample().iloc[:1]
frames = samples.gs.frame(
    windows={"sentinel-2-l2a": {"slots": ["0D"]}},
    reference="label",
)
chips = frames.gs.chip(128, target=grid)
```

`sample` groups scenes at the same place. `frame` chooses their time windows.
`chip` divides them into pixel windows on that sample's image grid. Each training
sample keeps its native CRS. For prediction, omit `reference="label"`; image
timestamps supply the reference times instead.

The Dataset's `chips` holds the chip rows, and `frames` holds their original maps.
Pixels are read when a chip is requested. `postprocess(logits, frames, chips)`
reconstructs dated label maps. `IGNORE_INDEX`, `CLASS_MAP` and `COLOR_MAP` are
loaded once from the dataset's class YAML and shared by preprocessing, training
and postprocessing. KHDTK is used only for inference.

## Saved files

- `data/dynamic_world/`: training data used by notebooks 0, 1 and 3.
- `data/my_dataset/`: train/validation/test pairs and catalog built in notebook 2.
- `data/khdtk/`: forest imagery and catalog for prediction.
- `data/khdtk_pred/`: predicted rasters in `land_cover/` and `catalog.parquet`.

Catalogs point to saved rasters using relative paths. Move the catalog and its
raster folders together.

These short runs demonstrate the workflow. Use validation to choose models
and keep test data for final evaluation. Training resumes the notebook's last
checkpoint; choose a new logger version and omit `ckpt_path="last"` for a fresh
experiment. Exercises at the end of each notebook are optional.
