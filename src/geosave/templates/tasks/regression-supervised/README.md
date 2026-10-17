# Pixelwise regression

The segmentation workspace with a different head: the same encoder and
decoder predict one value per pixel for each variable the `regression` head
names. `configs/regression-supervised.yaml` configures the Lightning Trainer, model, and data;
`configs/model_spec.yaml` declares the source bands, acquisition, and
preprocessing.

For the learning path through raster preparation, training, and prediction
maps, start with the
[notebooks](https://weedkat.github.io/geosave-engine/examples/). The
[training contracts](https://weedkat.github.io/geosave-engine/concepts/training/)
describe the library's regression Module/DataModule pair.

## Connect your Dataset

`modules/datasets.py` holds a `MyDataset` stub that `configs/regression-supervised.yaml` names
for each split. Implement it, or point the config at your own importable Dataset. Your Dataset
owns raster reading, preparation, spatial cuts, and tensor conversion; the
DataModule batches its samples and applies augmentation and spec transforms.
Keep training and validation scenes or geographic regions separate.

```python
sample = {"inputs": {"image": image}, "target": target}
```

One sample's `image` is a float tensor shaped `(band, y, x)`. `target` is
`float32 (variable, y, x)`, with channels in the regression head's `variables`
order. The Dataset supplies inputs suitable for the selected model. Decode
stored target scale and offset explicitly and use NaN for unknown targets.
Every split scores finite target values. The method's `SampleDataset` protocol
declares `__len__` and `__getitem__(index: int) -> Sample`; source classes can
remain ordinary torch Datasets. Input availability is not a task-level field.

ModelSpec is still under development in Alpha. Its current `preprocess` and
`prepare` methods operate on rasters and catalogs when your preparation code
calls them. The DataModule does not execute raster preprocessing or cuts.
See the
[input preparation guide](https://weedkat.github.io/geosave-engine/guides/model-spec/).

Select variables before stacking tensor channels and match their order to
normalization and encoder settings. The default DINOv3 encoder downloads
pretrained weights; set `pretrained: false` in its `init_args` for an offline
architecture check.

## Train

```bash
uv run python main.py fit --config configs/regression-supervised.yaml
```

A batch keeps the sample dictionary keys and adds a batch axis. Validation and
test score chip batches using MAE, RMSE, and R² over the configured variables
together. They do not merge predictions into full scenes; overlapping chips
can score the same source pixel more than once.

For prediction, call the model with prepared named inputs and retain the
reference raster or window to place its values on a grid. Apply the same tensor
transforms as training. This task pair supplies no prediction DataLoader or
`predict_step` that unpacks its samples. The
[catalogs and cuts guide](https://weedkat.github.io/geosave-engine/guides/catalogs-and-cuts/)
covers caller-owned window reconstruction.

## Publish the dataset

A workspace holds one dataset. Upload `data/` with the Hugging Face CLI; it
needs no GeoSave command:

```bash
hf upload-large-folder your-account/your-dataset data --repo-type dataset
```

`hf upload` suits a few files; `upload-large-folder` resumes and batches the
many small files a Zarr store holds. Link the dataset repo from your model card.
