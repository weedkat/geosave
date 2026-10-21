# Train a segmentation model

For real satellite files, prediction maps, and GeoTIFF export, start with the
[notebook learning path](../examples.md). Those notebooks define ordinary
Lightning training code. This guide shows the generated project’s editable supervised
Module/DataModule pair and the sample dictionaries it consumes.

Train the project’s supervised segmentation recipe on small, generated images.
This example runs on CPU with no data downloads or pretrained weights. Each
pixel has two input bands and a label: class `1` where the second band exceeds
the first, class `0` everywhere else.

```bash
pip install 'geosave[ml]'
geosave create demo --task segmentation --method supervised
cd demo
```

## Describe the inputs

Save this as `model_spec.yaml` in the workspace:

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

from modules.segmentation.supervised import DataModule, Module, Sample


class Chips(Dataset[Sample]):
    def __init__(self, size, start=0):
        self.size = size
        self.start = start

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> Sample:
        generator = torch.Generator().manual_seed(self.start + index)
        image = torch.rand(2, 16, 16, generator=generator)
        target = (image[1] > image[0]).long()
        return {"inputs": {"image": image}, "target": target}
```

`Sample` is a dictionary type. `inputs` contains the named model tensors;
`target` holds the class codes. The Dataset supplies inputs suitable for the
model. A sample has no batch dimension. The DataLoader adds one when it
combines samples:

| Tensor | One sample | Batch of four |
| --- | --- | --- |
| `inputs["image"]` | `(2, 16, 16)` | `(4, 2, 16, 16)` |
| `target` | `(16, 16)` | `(4, 16, 16)` |

Use `torch.int64` class codes from `0` to `num_classes - 1`. Set unknown labels
to `ignore_index`, here `255`, which loss and metrics exclude in every split.
Keep the Dataset, DataModule, Module and criterion aligned on that value.
`SampleDataset` describes the required `__getitem__` output; your source can
remain an ordinary torch Dataset. Input availability and acceptable values
belong to the source/model contract, with no task-level input-validity field.

## Add a small model and fit

Append to `train.py`:

```python
class TinySegmentation(nn.Module):
    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(image)


if __name__ == "__main__":
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
        model_chain={"model": {"class_path": "train.TinySegmentation"}},
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
`class_path` names its module and class, `train.TinySegmentation`, so the
Module imports it to construct the model, in this run and whenever a checkpoint
loads.
A single model can serve as the whole chain; an encoder, decoder, and head
become useful when you need to reuse or swap stages. See
[Model composition](models.md).

The DataModule batches the Datasets and finishes tensors on the training
device. The Module owns loss, optimizer, and metrics; it defaults to cross
entropy and AdamW with learning rate `0.001`. Lightning's `Trainer` runs the
training and validation loops. Validation accumulates metrics over the chip
batches; it does not reconstruct a full scene before scoring it.

Lightning constructs the chain through the Module's `configure_model` hook
before fitting. If you construct a Module and call it outside a Trainer before
that hook has run, call `module.configure_model()` first. The DataModule defaults
to batch size `8` and `4` worker processes; this example chooses `4` and `0` for
a small local CPU run.

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

The segmentation Module's `predict_step` returns the logits of a prepared
`Sample`, ignoring any target. The DataModule has no prediction DataLoader, so
for `Trainer.predict` pass your own loader of prepared samples. Use the
named-input call above for quick model development.

## Save a checkpoint

With the fitted Trainer and Module from above:

```python
trainer.save_checkpoint("last.ckpt")
restored = Module.load_from_checkpoint("last.ckpt", map_location="cpu").eval()
with torch.inference_mode():
    restored_logits = restored(**batch["inputs"])
restored_logits.shape
```

```text
torch.Size([4, 2, 16, 16])
```

This explicit save works even though automatic checkpointing was disabled in
the short example. Loading restores the task's model and hyperparameters; in
a new process, the Module imports `train` to rebuild `TinySegmentation`, so
run from the folder holding `train.py`. The main guard keeps that import from
starting another fit.
Resume optimization by passing `ckpt_path="last.ckpt"` to `Trainer.fit` with
a larger `max_epochs`. Loading a Module alone does not resume the Trainer's
optimizer and scheduler state.

For reuse through Transformers, construct each stage with a registered `name`
as shown in [Model composition](models.md), then save its chain and input spec
as a [model release](release.md). This example uses `class_path`, which native
releases reject; keep its Lightning checkpoint for reuse. A release preserves
the chain's computation and weights, while a checkpoint also preserves the task
and training state.

## Connect real data and configure runs

Replace `Chips` with a Dataset for your source while retaining its sample
contract. Keep training and validation regions or scenes separate, and choose
band order, target codes, and nodata handling explicitly. The
[catalogs and cuts guide](catalogs-and-cuts.md) explains how to list spatial
chips and read their rasters; [the raster basics](../basics/rasters.md) covers
native xarray objects and conversion to tensors.

Notebook 0 supplies a real-file Dataset, notebook 2 acquires imagery and builds
a catalog, and notebook 3 prepares features and exports a prediction map. Follow the [examples guide](../examples.md) to run
them. Notebooks 0 and 1 return image/label tuples for their own LightningModules;
notebook 3 supplies `inputs` and `target` to the library task pair.

Use [a workspace](workspaces.md) when you want an editable project and YAML
run configuration. The segmentation and regression starters provide
configurations for library task pairs; you still supply the Dataset classes.
They select larger encoders that may download pretrained weights.

To replace the tiny classifier with a published encoder, decoder, and head,
follow [Build from pretrained weights](pretrained.md). That guide shows matching
band/context inputs, different learning rates for stages, and encoder freezing.

Continue with [training contracts](../concepts/training.md) for augmentation,
other tasks, semi-supervised methods, and confidence calibration. See
[the training API](../reference/training.md) for constructor arguments.
