# Train an offline starter

```bash
pip install 'geosave[ml]'
geosave create demo --task lightning --method boilerplate
cd demo
python main.py fit --config configs/lightning-boilerplate.yaml \
  --trainer.accelerator cpu --trainer.devices 1 --trainer.max_epochs 1
```

This starter trains a small classifier on deterministic synthetic data, without
pretrained weights or data downloads. The generated files are yours to edit.
The run reports training and validation loss and saves a checkpoint under
`artifacts/lightning-boilerplate/version_N/` with the default logger enabled.
A one-epoch run checks the training setup; it does not establish model quality.

## Follow one sample through training

| File under `modules/lightning/boilerplate/` | Responsibility |
| --- | --- |
| `datasets.py` | `ExampleDataset` returns `({"features": tensor}, target)`: four floating features and one integer class, with no batch dimension. |
| `data.py` | `DataModule` constructs split Datasets and ordinary DataLoaders. A batch has features shaped `(batch, 4)` and targets shaped `(batch,)`. |
| `module.py` | `Module` runs a small network, computes cross-entropy, logs loss, and constructs AdamW. |

The config selects these classes and their constructor arguments.
`main.py` invokes GeoSave's LightningCLI integration; Lightning owns training,
validation, checkpointing, and device placement. Change the Dataset and model
together when adapting the example to another sample shape.

The synthetic train and validation datasets use overlapping deterministic
examples. This is a wiring demonstration. For real experiments, supply separate
scenes or geographic regions for validation and reserve independent test data.

## Use raster data

[Images to training chips](../quickstart.md) produces a two-band image tensor
and an integer label raster. A segmentation Dataset can return those pixels as
`{"inputs": {"image": image}, "target": target}`; that is the geospatial
supervised recipe's sample contract, distinct from this boilerplate's tuple.

Your Dataset owns file reads, chip selection, channel order, missing inputs,
and label codes. Read only the requested sample. Decode stored imagery when
physical values are required, preserve integer target classes, and choose an
explicit ignore code for missing targets. Keep the loss and metrics aligned
with that code. [Training contracts](../concepts/training.md) describes the
geospatial recipes and their tensor layouts.

## Geospatial recipes

The segmentation, regression, classification, and MIM starters use ModelSpec v3.
Their DataModules take an explicit `size` for augmentation and call the spec's
shared tensor transforms in every split. Sampling and chip layout use
`catalog.gs.sample(...)` and `gs.Chips(...)` outside ModelSpec.

The project-owned Dataset implementations remain extension points: supply your
own reads, labels, and sample contract. The [processing workflow](model-spec.md)
includes an offline end-to-end test using a real generated regression DataModule
and a tiny temporal model. Existing generated workspaces need the documented
v3 migration; upgrading the library does not rewrite copied project files.

## Save and reuse the result

A Lightning checkpoint holds the Module's weights and training state. Use it
to resume training or load the same project-owned Module implementation.
The boilerplate's custom network is not a registered GeoSave model chain, so it
cannot be exported directly with `GeoSaveModel.from_chain`.

For native releases, build a supported registered chain and follow
[save and load a release](release.md). That guide includes an offline round trip
using a built-in segmentation head. [Workspaces](workspaces.md) explains adding
methods and managing generated files; [model inputs](model-spec.md) explains
preprocessing separately from training.
