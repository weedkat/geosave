# Pixelwise regression

The segmentation workspace with a different head: the same encoder and
decoder predict one value per pixel for each variable the `regression` head
names. `configs/train.yaml` configures the Lightning Trainer, model, and data;
`configs/model_spec.yaml` declares the source bands, acquisition, and
preprocessing.

## Prepare training items

`spec.prepare` is not settled yet. Until it is, write each split's inputs and
target values as COGs and catalogue them in one item table per split:

```python
import geosave as gs

files = sample.gs.to_cog({"image": "data/train/s0/image/image_{time:%Y%m%d}.tif",
                          "target": "data/train/s0/target.tif"})
gs.stac.create_items(files, id="s0").gs.to_geoparquet("data/train.parquet")
```

The target raster's nodata becomes unknown and its stored scale and offset
are applied. Loss and metrics read only pixels with a known value.

## Train

```bash
uv run python main.py fit --config configs/train.yaml
```

A batch is `(model_inputs, target, ids, valid)`; `target` is `float32` shaped
`[batch, variable, H, W]`, NaN where unknown. Validation merges chip values
back onto each sample and scores every sample whole with MAE, RMSE and R².
