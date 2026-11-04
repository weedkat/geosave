# Land-cover mapping examples

Run the notebooks in order. Use the repository environment (`uv sync`) as the
kernel and `examples/` as the working directory. Start each notebook in a fresh
kernel; files carry forward between lessons.

| Notebook | What you do |
| --- | --- |
| [0 · Explore and train](00_basics.ipynb) | Inspect images and labels, then train a small model. |
| [1 · Change the model](01_model.ipynb) | Swap model parts and compare training runs. |
| [2 · Build a dataset](02_sources.ipynb) | Fetch images, save and catalog image/label pairs, and pair them into samples. |
| [3 · Train and predict](03_inputs.ipynb) | Make chips, train, predict without labels, and save maps. |

The first download is about 400 MB. Notebook 2 also needs Planetary Computer
access. Later runs reuse local files. Maps and TensorBoard need a browser.

## From files to model inputs

Notebook 2 saves rasters with `to_catalog`, which writes the files and returns
their catalog; no STAC Item or Collection is built by hand. `catalog.gs.sample()`
groups catalog rows into samples by place and date without opening a file, and
each sample opens with `.gs.to_datatree()` as a lazy DataTree, `/{collection}/{res}`.

Notebook 3 uses one `DynamicWorld` Dataset for every split and for inference.
It takes DataTrees, each a sample opened from a catalog: image/label pairs from
notebook 2's `data/my_dataset/catalog.parquet`, or dated forest scenes from notebook 2's. Training reads complete tiles and applies random crops.
Setting `chip_size` cuts the trees with `gs.Chips` into fixed windows.

Validation and test collect chip scores and call `dataset.merge()` at epoch end.
Metrics count each valid pixel once on the restored grid. These loaders run in
Dataset order and cover every chip; the short validation split contains two
complete tiles.

For KHDTK inference:

```python
items = gs.open_vector("data/khdtk/catalog.parquet")
trees = []
for rows in items.gs.sample(pair=["sentinel-2-l2a"]):
    trees.append(rows.gs.to_datatree())
forest = DynamicWorld(trees, chip_size=128, overlap=64)
loader = DataLoader(forest, batch_size=8)
module.eval()
batches = []
with torch.inference_mode():
    for image in loader:
        batches.append(module(image.to(module.device)).cpu())
merged = forest.merge(torch.cat(batches).numpy())
forest_cover = postprocess(merged, forest.trees)
```

`postprocess()` adds dates, class IDs, colours and nodata from `classes.yaml`.
KHDTK is used only for inference.

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
