# Semantic segmentation

`configs/segmentation-supervised.yaml` configures GeoSave's supervised Module/DataModule pair,
model stages, optimizer, and Lightning Trainer. Before the first fit, supply
the Dataset classes named by that config. `configs/model_spec.yaml` declares
raster requirements, input bindings, chip size, and tensor transforms.

For a complete learning project using satellite image/label pairs and
prediction maps, follow the
[notebooks](https://weedkat.github.io/geosave-engine/examples/). The
[training guide](https://weedkat.github.io/geosave-engine/guides/training/)
shows a complete Dataset feeding the library task pair used by this starter.

## Prepare a raster

Run the local example from this workspace; it needs no downloads or model:

```bash
uv run python scripts/prepare_example.py
```

Input is a packed, lazy four-band Sentinel-2 `xarray.Dataset`. Output is a
sample-ready lazy Dataset with nodata masked and reflectance unpacked. The
example computes only the small result it prints. The spec calls
`optical.gs.mask_and_scale()` to decode nodata and packing together.

```python
from geosave.model.spec import ModelSpec

spec = ModelSpec.load("configs/model_spec.yaml")
image = spec.preprocess({"sentinel_2_l2a": optical})["image"]
```

Job areas, dates, destinations, and Prefect settings are caller-owned flow
parameters. Collection identity, endpoint priority, query/load options, source
requirements, and model-specific preparation stay with the model document.

## Connect your Dataset

The config references `modules.datasets.MyDataset` for `train` and `val`. It is
a stub in `modules/datasets.py`: implement it, or replace the `class_path` and
`init_args` for each split with your own importable Dataset. Keep training and validation scenes
or geographic regions separate before cutting chips.

Your Dataset reads each sample and returns this dictionary:

```python
sample = {"inputs": {"image": image}, "target": target}
```

One sample's `image` is a float tensor shaped `(band, y, x)`, and `target` is an
`int64 (y, x)` class map. The Dataset supplies inputs suitable for the selected
model. Class codes follow the head's `classes` list. Set unknown targets to
`255`, matching the Dataset, DataModule, Module and criterion ignore value.
Context inputs, when needed, also belong in `inputs`. The method's `SampleDataset`
protocol declares `__len__` and `__getitem__(index: int) -> Sample`; source
classes can remain ordinary torch Datasets.

The DataModule batches your Datasets and prepares their tensors on the device.
It does not discover files, read a catalog, preprocess rasters, or cut chips.
With the current ModelSpec implementation, those steps are explicit:

| Section | What it does | Where it runs |
| --- | --- | --- |
| `rasters` / `preprocessing` | Select named variables and decode or transform raster values. | Your preparation script or Dataset calls `spec.preprocess`. |
| `chips` | Declare spatial cuts. | Your code calls `spec.chips.cut` on a sample table. |
| `inputs` / `context` | Bind pixels and context to the model's argument names. | Your Dataset calls `spec.model_inputs` before tensor conversion. |
| `transforms` | Normalize pixel tensors. | The DataModule, on the device in train, validation, and test. |

ModelSpec is still being developed in Alpha. See the
[input preparation guide](https://weedkat.github.io/geosave-engine/guides/model-spec/)
for the current `preprocess`, `prepare`, and input-binding APIs.

`data.init_args.augmentations` in `configs/segmentation-supervised.yaml` run on the device while
training only, on the image and the target together, before `transforms`.

A collated batch keeps the same dictionary keys, adding a batch axis to the
tensors. Validation and test score supplied labels in chip batches, excluding
ignored targets; they do not reconstruct full scenes. Overlapping chips can therefore
score the same source pixel more than once.

To change the encoder, edit `model.init_args.model_chain.encoder` in
`configs/segmentation-supervised.yaml`. Enable the matching `context` recipe in the spec
for Prithvi TL or Clay. The default DINOv3 starter leaves context disabled.
Clay wavelengths and GSD are explicit encoder constructor settings.

Select variables explicitly before tensor conversion and match their channel
order to normalization statistics and encoder settings. The default DINOv3
encoder downloads pretrained weights; review its RGB channel adaptation when
using multispectral inputs. Set `pretrained: false` in its `init_args` for an
offline architecture check.

```bash
uv run python main.py fit --config configs/segmentation-supervised.yaml
```

## Release and inference

Publish the selected checkpoint's native inference graph, the processing
contract, and this workspace with the run's checkpoints:

```bash
python scripts/publish.py your-account/your-model \
    artifacts/segmentation-supervised/version_0/checkpoints/<best>.ckpt
```

The repository root holds `config.json`, `model.safetensors`, and
`model_spec.yaml`; `workspace/` holds the code, configs, and checkpoints for
resuming or fine-tuning. Only the files `scripts/publish.py` allows are
uploaded, so `.env`, data, and logs stay local.

Deployment loads the model and its spec at the printed commit:

```python
import geosave.model.release  # registers GeoSaveModel with AutoModel
from transformers import AutoModel
from geosave.model.spec import ModelSpec

model = AutoModel.from_pretrained("your-account/your-model", revision=commit).eval()
spec = ModelSpec.load(f"hf://your-account/your-model@{commit}/model_spec.yaml")
encoder = model.chain.encoder  # plain nn.Module for downstream tasks
```

Resume or fine-tune from the released workspace:

```bash
hf download your-account/your-model --include "workspace/*" --revision <commit> --local-dir release
cd release/workspace
python main.py fit --config configs/segmentation-supervised.yaml --ckpt_path artifacts/segmentation-supervised/version_0/checkpoints/last.ckpt
```

The spec owns raster requirements, acquisition, and preprocessing. The native
model accepts named tensor inputs and returns raw logits. Task-specific
inference code owns tiling, iterative execution, aggregation, and output
interpretation when those behaviors are needed.

Load and execute the saved spec explicitly: a release does not apply tensor
normalization automatically. Match the preparation used in training. A direct
named-input model call supports prediction; this task pair supplies no
prediction DataLoader or `predict_step` that unpacks its sample dictionary.

Cut windows from the item table through its `gs` accessor:
`items.gs.sample(reference="label")` lists one sample per label Item, and
`spec.frames.cut(...)` and `spec.chips.cut(...)` split it in time and space
without reading pixels. A row reads its own pixels with `row.gs.to_xarray()`;
`spec.model_inputs(rasters)` binds them to the model's inputs, and
`.gs.to_tensor()` turns each into a tensor.

Enable the encoder's `context` recipe in `model_spec.yaml` when the rasters
carry time labels and a georeferenced grid. The template leaves context
disabled for its default DINOv3 encoder. Context encoding is shared by
training and inference; cached values are an optional explicit argument, not a
different inference recipe.

For whole-scene reconstruction, keep the window identity alongside each
prediction and merge them with `windows.gs.merge(chips, outputs, bands=...)`. Blend logits before
interpreting class codes. See
[catalogs and cuts](https://weedkat.github.io/geosave-engine/guides/catalogs-and-cuts/)
for reading and merging windows; scene reconstruction is caller-owned.

`ImageAugmenter` remains available from `geosave.ml.transforms` for
YAML-configured native Kornia augmentation, including nested pipelines. The
DataModule owns joint image/target and temporal handling, passing `data_keys`
at the augmentation call. Multiple inputs require explicit keys; a single
image can omit them. The shared augmenter
also supports Pascal VOC `bbox_xyxy`, COCO `bbox_xywh`, and normalized
`bbox_yolo` boxes for custom training methods; YOLO class IDs use a separate
`class` or `label` tensor.

## Publish the dataset

A workspace holds one dataset. Upload `data/` with the Hugging Face CLI; it
needs no GeoSave command:

```bash
hf upload-large-folder your-account/your-dataset data --repo-type dataset
```

`hf upload` suits a few files; `upload-large-folder` resumes and batches the
many small files a Zarr store holds. Link the dataset repo from your model card.
