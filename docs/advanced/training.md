# Train in an editable Lightning workspace

Install `geosave[ml]`, then create the offline Lightning starter and run its
synthetic Dataset through a training and validation epoch.

```bash
geosave create demo --task lightning --method boilerplate
cd demo
python main.py fit --config configs/boilerplate.yaml --trainer.max_epochs 1
```

The first run saves its resolved configuration, checkpoint, and TensorBoard
events in `artifacts/lightning-boilerplate/version_0/`. Subsequent runs use
new version folders.

## Connect real imagery

```bash
geosave create landcover --task segmentation --method supervised
cd landcover
```

Open the generated README for the expected image bands, target format and
supported stages. Set the paired image and label paths in
`configs/supervised.yaml`, then run `python main.py fit --config configs/supervised.yaml`.
This starter trains on aligned chips. Validation and test await a project
evaluation implementation.

| File | Where to make a change |
| --- | --- |
| `configs/supervised.yaml` | Model, optimizer, split Datasets, callbacks and run settings. |
| `training/datasets.py` | Read and prepare one sample from the project's data source. |
| `training/datamodule.py` | Build DataLoaders and apply training augmentation. |
| `training/module.py` | Compute the training objective and metrics. |
| `processing/` | Ingest source imagery, prepare model inputs, predict, postprocess or write the output. |
| `pipeline.py` | Connect those stages for inference with a trained model. |

Share input preparation between the Dataset and Pipeline. Keep training
augmentation in the DataModule. The Pipeline can run on an ordinary model
without constructing a Lightning trainer. Some starters provide stage stubs
for project-specific work. [Recipe status](../reference/training.md)

A starter is copied once. Its files belong to the project, including any
additional Datasets or methods you add. See [CLI inputs](../reference/cli.md)
and [training contracts](../concepts/training.md).

## Save and share your work

Keep the effective config, code and data revisions, dependency versions, and
checkpoint together so another user can continue the run. The supervised
segmentation starter includes `scripts/publish.py` for releasing the trained
chain as a tagged version with its lineage in the model card, and optionally
archiving the run in a separate training repository.
Use `python scripts/publish.py --help` inside that workspace for its arguments.
The editable Pipeline lives with the project source. See
[model releases](release.md) and [experiment provenance](../mlops/index.md).
