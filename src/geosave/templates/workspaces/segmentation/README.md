# Semantic segmentation

`configs/train.yaml` configures the Lightning Trainer, model, and data objects.
`configs/model_spec.yaml` accompanies the model and declares its required source
bands, STAC acquisition recipe, and lazy preprocessing.

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

## Prepare training items

`spec.prepare` is not settled yet and raises `NotImplementedError`. Until it
is, write each split's model inputs and labels as COGs and catalogue them in one
item table per split:

```python
import geosave as gs

files = sample.gs.to_cog({"image": "data/train/s0/image/image_{time:%Y%m%d}.tif",
                          "label": "data/train/s0/label.tif"})
gs.stac.create_item(files, id="s0").gs.to_geoparquet("data/train.parquet")
```

`supervised.DataModule` reads the prepared item tables through
`configs/model_spec.yaml`:

| Section | What it does | Where it runs |
| --- | --- | --- |
| `preprocessing` | nodata, unpacking, band order | once, before the catalog is written |
| `chips` | cuts every sample into model-sized chips | from the catalog, no file opened |
| `inputs` | names what the model chain takes | per chip |
| `transforms` | tensor transforms such as normalization | on the device, every split |

`data.init_args.augmentations` in `configs/train.yaml` run on the device while
training only, on the image and the target together, before `transforms`.

A batch is `(model_inputs, target, ids, valid)`. `model_inputs` holds the inputs the
spec lists; `target` is an `int64` class mask shaped `[batch, H, W]`, with pixel
code 255 ignored. Validation merges tile predictions back onto each sample and
scores every sample whole.

To change the encoder, edit `model.init_args.model_chain.encoder` in
`configs/train.yaml`. Enable the matching `context` recipe in the spec
for Prithvi TL or Clay. The default DINOv3 starter leaves context disabled.
Clay wavelengths and GSD are explicit encoder constructor settings.

```bash
uv run python main.py fit --config configs/train.yaml
```

## Release and inference

Publish the selected checkpoint's native inference graph, the processing
contract, and this workspace with the run's checkpoints:

```bash
python scripts/publish.py your-account/your-model \
    artifacts/my_model/version_0/checkpoints/<best>.ckpt
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
python main.py fit --config configs/train.yaml --ckpt_path artifacts/my_model/version_0/checkpoints/last.ckpt
```

The spec owns raster requirements, acquisition, and preprocessing. The native
model accepts named tensor inputs and returns raw logits. Task-specific
inference code owns tiling, iterative execution, aggregation, and output
interpretation when those behaviors are needed.

Library supervised batches contain `(model_inputs, target, ids, valid)`.
`ids` are the chips' positions in the dataset's `windows` table;
validation/test assemble logits and score each frame's original labels.
Validity excludes input nodata before zero filling. Augmentation stays in the
training DataModule.

Cut windows from the item table with `geosave.geodata.cuts`:
`cuts.stacks(items, on="label")` lists one window per label, and
`spec.frames.cut(...)` and `spec.chips.cut(...)` split it in time and space
without reading pixels. A row reads its own pixels with `row.gs.to_stack()`;
`spec.model_inputs(rasters)` binds them to the model's inputs, and
`.gs.to_tensor()` turns each into a tensor.

Enable the encoder's `context` recipe in `model_spec.yaml` when the rasters
carry time labels and a georeferenced grid. The template leaves context
disabled for its default DINOv3 encoder. Context encoding is shared by
training and inference; cached values are an optional explicit argument, not a
different inference recipe.

Native `tiler.Merger` accepts the row's integer tile ID and numpy prediction.
Convert tensors with `detach().cpu().numpy()` in ML. Keep tilers and halo widths
with the job and pass padding to `Merger.merge(extra_padding=...)`.

Supervised validation/test own completion and validity policy. They accumulate
logits and coverage using the same native taper and score original parent
labels only after every tile arrives.
Validation/test require a single device until scene partitioning across ranks
is supported. Prediction still returns indexed logits for caller-owned merging.

`ImageAugmenter` remains available from `geosave.ml.transforms` for
YAML-configured native Kornia augmentation, including nested pipelines. The
DataModule owns joint image/target and temporal handling, passing `data_keys`
at the augmentation call. Multiple inputs require explicit keys; a single
image can omit them. The shared augmenter
also supports Pascal VOC `bbox_xyxy`, COCO `bbox_xywh`, and normalized
`bbox_yolo` boxes for custom training methods; YOLO class IDs use a separate
`class` or `label` tensor.
