# Train in an editable Lightning workspace

Generate a project, run one training epoch, then adapt its Python files to
your data and model.

**You need:** `geosave[ml]`. The supplied Lightning starter runs offline.

## Generate and run the starter

```bash
geosave create demo --task lightning --method boilerplate
cd demo
python main.py fit --config configs/lightning-boilerplate.yaml --trainer.max_epochs 1
```

The starter trains its supplied example Dataset. Lightning reports:

```text
`Trainer.fit` stopped: `max_epochs=1` reached.
```

The first run writes its configuration, checkpoint and TensorBoard events to
`artifacts/lightning-boilerplate/version_0/`. Later runs use new version folders.

## Find the files you edit

| File | Purpose |
| --- | --- |
| `datasets.py` | Read one sample with inputs and targets. |
| `data.py` | Build loaders and apply shared transforms. |
| `module.py` | Define the model, loss, metrics and optimizer. |
| `processing.py`, where supplied | Share raster preparation with inference. |
| YAML config | Select components and training settings. |
| `main.py` | Run the LightningCLI entry point. |

Generated files belong to your project. Updating GeoSave does not rewrite them.
Generation does not install packages or create a dependency lockfile.

## Connect geospatial data

Choose [model components](models.md) and reuse the same
[preprocessor](model-inputs.md) for training and inference. Your Dataset returns
the named inputs that the model expects. Choose geographic or temporal splits
before reading training chips.

The Lightning starter uses synthetic samples to verify the training setup.
Geospatial recipes still need migration or project implementation; check
[recipe status](../reference/training.md). There is no verified complete
raster-training starter to copy yet.

Use `geosave add-task` inside a workspace to add another method.
See the [CLI reference](../reference/cli.md) for selectors and file behavior,
and [training concepts](../concepts/training.md) for sample and target contracts.

## Save and share your work

Keep checkpoints for resuming training. Use [native releases](release.md)
for inference with registered GeoSave stages and pipeline settings.
The generic `geosave publish` command is a placeholder; use the release API.

Archive code, effective configuration, data revision, dependency lockfile,
metrics and selected checkpoints with the experiment. Keep data, experiments
and inference releases separate; [MLOps](../mlops/index.md) shows the three
repositories and how their revisions connect.

The supervised-segmentation publication script can archive training and export
a chain to separate Hub repositories. It currently exports a chain without a
raster pipeline. Other task publication scripts still need migration; see
[release status](../contributing/open-decisions.md#template-and-notebook-migrations).
