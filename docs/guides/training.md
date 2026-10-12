# Train a segmentation model

Train the library's supervised segmentation task on small, generated images.
This example runs on CPU with no data downloads or pretrained weights. Each
pixel has two input bands and a label: class `1` where the second band exceeds
the first, class `0` everywhere else.

## Describe the inputs

Save this as `model_spec.yaml` in a new working directory:

```yaml
schema_version: 2
rasters:
  optical:
    variables: [red, nir]
chips:
  size: 16
inputs:
  image: !ref optical
```

The spec names the two bands, the 16 × 16 chip size, and the model input
`image`. This example supplies tensors directly, so it needs no STAC recipe.
For a real raster source, your Dataset reads the chips and converts their
pixels to tensors; the DataModule does not read or cut rasters for you.

## Return one sample from a Dataset

Start `train.py` with:

```python
import torch
from lightning import Trainer, seed_everything
from torch import nn
from torch.utils.data import Dataset

from geosave_engine.ml.tasks.segmentation.supervised import DataModule, Module, Sample
from geosave_engine.model.registry import register_model


class Chips(Dataset):
    def __init__(self, size, start=0):
        self.size = size
        self.start = start

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> Sample:
        generator = torch.Generator().manual_seed(self.start + index)
        image = torch.rand(2, 16, 16, generator=generator)
        valid = torch.isfinite(image).all(dim=0)
        target = (image[1] > image[0]).long()
        target[~valid] = 255
        return {"inputs": {"image": image}, "target": target, "valid": valid}
```

`Sample` is a dictionary type. `inputs` contains the named model tensors;
`target` holds the class codes; `valid` is a boolean mask of pixels with usable
inputs. A sample has no batch dimension. The DataLoader adds one when it
combines samples:

| Tensor | One sample | Batch of four |
| --- | --- | --- |
| `inputs["image"]` | `(2, 16, 16)` | `(4, 2, 16, 16)` |
| `target` | `(16, 16)` | `(4, 16, 16)` |
| `valid` | `(16, 16)` | `(4, 16, 16)` |

Use integer class codes from `0` to `num_classes - 1`, with `torch.int64`
targets. Set unknown labels to `ignore_index`, here `255`. In **training**, the
Module ignores that label value; it does not apply `valid` to the target. Your
Dataset must also set labels to `255` wherever inputs are unusable. In
**validation and testing**, the Module additionally excludes pixels where
`valid` is false. Keep the Dataset, DataModule, Module, and loss aligned on the
same ignore value.

## Add a small model and fit

Append to `train.py`:

```python
@register_model("model", "tiny_segmentation")
class TinySegmentation(nn.Module):
    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(image)


seed_everything(7, workers=True)
data = DataModule(
    spec="model_spec.yaml",
    train=Chips(16),
    val=Chips(8, start=100),
    batch_size=4,
    num_workers=0,
    ignore_index=255,
)
module = Module(
    model_chain={"model": {"name": "tiny_segmentation"}},
    ignore_index=255,
    metrics=["iou.macro", "f1.macro"],
)
trainer = Trainer(
    accelerator="cpu",
    devices=1,
    max_epochs=1,
    limit_train_batches=2,
    limit_val_batches=1,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    enable_model_summary=False,
)
trainer.fit(module, datamodule=data)
trainer.validate(module, datamodule=data)
```

Run it from the directory holding both files:

```bash
python train.py
```

Example output, rounded for readability:

```text
`Trainer.fit` stopped: `max_epochs=1` reached.

Validate metric    DataLoader 0
val_f1_macro       0.3272
val_iou_macro      0.2432
val_loss           0.7155
```

This is a wiring example, so two training batches produce modest scores. A
smaller validation loss means better fit to the labels. IoU measures overlap
between predicted and true class regions; F1 combines precision and recall.
`macro` averages the class scores with equal weight. Remove the batch limits
and train longer when you move to a real task.

The model is ordinary PyTorch. Its named input `image` matches the Dataset,
and its `num_classes` tells the Module how many class scores it produces.
The registered name lets the Module construct it from `model_chain`.
A single model can serve as the whole chain; an encoder, decoder, and head
become useful when you need to reuse or swap stages. See
[Model composition](models.md).

The DataModule batches the Datasets and finishes tensors on the training
device. The Module owns loss, optimizer, and metrics; it defaults to cross
entropy and AdamW with learning rate `0.001`. Lightning's `Trainer` runs the
training and validation loops. Validation accumulates metrics over the chip
batches; it does not reconstruct a full scene before scoring it.

## Inspect predictions

For an interactive session, run `python -i train.py` instead of
`python train.py`. After fitting, enter:

```python
batch = next(iter(data.val_dataloader()))
batch = data.on_after_batch_transfer(batch, 0)
module.eval()
with torch.inference_mode():
    logits = module(**batch["inputs"])
    labels = logits.argmax(dim=1)
tuple(logits.shape), tuple(labels.shape)
```

Output:

```text
((4, 2, 16, 16), (4, 16, 16))
```

Logits are raw class scores. Taking `argmax` along the class dimension gives
one predicted class per pixel. The batch above stays on CPU; for a model on
another device, move its input tensors there first. Calling
`on_after_batch_transfer` after fitting applies the same input transforms as
validation, without training augmentation.

The supplied task pair has no prediction DataLoader or custom `predict_step`
that unpacks `Sample`. Use the named-input call above for model development.
To use `Trainer.predict`, supply a prediction loader and a `predict_step` that
accepts its batches and applies the required input preparation.

## Connect real data and configure runs

Replace `Chips` with a Dataset for your source while retaining its sample
contract. Keep training and validation regions or scenes separate, and choose
band order, target codes, and nodata handling explicitly. The
[catalogs and cuts guide](catalogs-and-cuts.md) explains how to list spatial
chips and read their rasters; [the raster basics](../basics/rasters.md) covers
native xarray objects and conversion to tensors.

Use [a workspace](workspaces.md) when you want an editable project and YAML
run configuration. The segmentation and regression starters provide
configurations for library task pairs; you still supply the Dataset classes.
They select larger encoders that may download pretrained weights.

Continue with [training contracts](../concepts/training.md) for augmentation,
other tasks, semi-supervised methods, and confidence calibration. See
[the training API](../reference/training.md) for constructor arguments.
