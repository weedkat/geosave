# Create a training workspace

To learn from real satellite images through training and prediction maps,
follow the [notebooks](../examples.md). A workspace keeps your Python modules
and run configurations together once you choose a training setup.

```bash
geosave create demo
cd demo
geosave add-task lightning-boilerplate
python main.py fit --config configs/lightning-boilerplate.yaml --trainer.max_epochs 1
```

Output, after the Lightning startup messages:

```text
`Trainer.fit` stopped: `max_epochs=1` reached.
```

`create` makes an empty workspace; `add-task` copies one task template into it.
The `lightning-boilerplate` task trains a small feature classifier on generated
data. It requires no dataset download, pretrained weights, or remote service.
The command above limits it to one epoch; the config itself trains for 100,
logging to `artifacts/lightning-boilerplate/` and keeping the best checkpoint.

## Find the files you edit

```text
demo/
├── configs/lightning-boilerplate.yaml
├── modules/
│   ├── task.py
│   └── data.py
├── artifacts/
├── data/
├── notebooks/
├── scripts/
├── .env.example
├── .gitignore
└── main.py
```

`modules/task.py` contains an ordinary LightningModule with the network, loss,
and optimizer. `modules/data.py` supplies the generated examples through a
Dataset and DataLoaders. `configs/lightning-boilerplate.yaml` selects those classes and sets
Trainer options. `main.py` runs `GeosaveCLI`, a LightningCLI subclass.
The [GeosaveCLI API](../reference/training-tools.md) describes logger and
configuration behavior. `.gitignore` keeps `.env`, data, and checkpoints out
of the repository, so the workspace can be pushed as is.
Each run's effective configuration is saved as `config.yaml` in its
`artifacts/<model_name>/version_N/` folder.

Change a run without editing Python:

```bash
python main.py fit --config configs/lightning-boilerplate.yaml \
  --trainer.max_epochs 3 --data.init_args.batch_size 8
```

To change the data source, edit the Dataset in `modules/data.py`; to change
training behavior, edit `modules/task.py`.

## Choose a task for your workspace

| Task template | Provides | Before the first fit |
| --- | --- | --- |
| `lightning-boilerplate` | Editable Lightning Module/DataModule and a synthetic example. | Run the command above. |
| `segmentation-supervised` | Config selecting GeoSave's supervised segmentation pair, a model spec, a Dataset stub, plus example-preparation and publication scripts. | Implement `modules/datasets.py` and supply suitable data; review encoder weight downloads. |
| `regression-supervised` | Config selecting GeoSave's supervised pixelwise regression pair, a model spec, and a Dataset stub. | Implement `modules/datasets.py` and supply suitable data; review encoder weight downloads. |

Without an argument, `add-task` prompts for a template. A workspace has one
dataset: `modules/datasets.py` and `configs/model_spec.yaml` are shared, and
each added task contributes its own `configs/<task>.yaml`, so you can compare
training methods on the same data. Files that already exist prompt before they
are overwritten. To start a segmentation project:

```bash
geosave create landcover
cd landcover
geosave add-task segmentation-supervised
```

This creates the project; its first fit needs your dataset implementation.
The supplied config references `datasets.MyDataset`, which is a placeholder,
not a bundled class. Put your Dataset in an importable Python module and
replace that `class_path` and its `init_args` for each split.
[Train a segmentation model](training.md) shows a complete Dataset feeding the
real library Module/DataModule before you connect larger encoders or files.

Both task starters include `configs/model_spec.yaml`. It describes raster
requirements, chip size, input bindings, and tensor transforms; the training
YAML separately configures the Module, DataModule, optimizer, and Trainer.
See [Model spec](model-spec.md).

ModelSpec is still under active development in Alpha. The starter reflects the
current API; your Dataset remains responsible for raster reading, preparation,
cuts, and conversion to the task's sample contract.

The segmentation starter's default DINOv3 encoder uses pretrained weights.
For a complete recipe with explicit band/context requirements, see
[Build from pretrained weights](pretrained.md). Choose `pretrained: false` in
the encoder's `init_args` when testing an architecture without downloading
weights; that trains from random encoder parameters.

## Keep a run configuration

```yaml
# An excerpt from the lightning-boilerplate task's config
model:
  class_path: modules.task.CustomTask
  init_args:
    input_size: 4
    hidden_size: 8
    learning_rate: 0.001
data:
  class_path: modules.data.CustomDataModule
  init_args:
    train_size: 16
    val_size: 8
    batch_size: 4
    num_workers: 0
```

`class_path` identifies the Python class to construct. `init_args` supplies its
constructor arguments. This allows a project to change training classes
without regenerating the workspace. Library task Modules construct their
network from their own `model_chain` settings; the CLI constructs the Module
and DataModule selected above.

To override a configured stage from the shell, pass the whole stage mapping:

```bash
python main.py fit --config configs/segmentation-supervised.yaml \
  --model.init_args.model_chain.encoder \
  '{name: dinov3, init_args: {in_channels: 4, input_size: 224, pretrained: false}}'
```

This command belongs to a segmentation or regression task, once its Dataset
is configured. `model_chain` is a mapping; CLI overrides do not reach further
inside an encoder's `init_args`. The whole-stage override replaces the original
encoder specification, so include every argument you want to retain.

`GeosaveCLI` selects TensorBoard under `artifacts/<model_name>/`. Set
`MLFLOW_TRACKING_URI` if you also want an MLflow logger, or configure Lightning
loggers explicitly. Choose checkpoint behavior through Trainer options and
Lightning callbacks.

Resume a saved task run with the same task/data configuration and a checkpoint:

```bash
python main.py fit --config configs/lightning-boilerplate.yaml \
  --ckpt_path path/to/last.ckpt \
  --trainer.max_epochs 5 --trainer.default_root_dir artifacts/run-3
```

The checkpoint path must name a file from a run that enabled checkpoint saving.
`max_epochs` is the total epoch limit including completed epochs. Use a new
output directory for the effective config while restoring the checkpoint's
training state.

## Optional scripts and publication

The segmentation starter includes `scripts/prepare_example.py`, a small
synthetic reflectance-preprocessing example, and `scripts/publish.py`, which
uploads a trained model release and selected workspace files to Hugging Face. Review
these scripts and their inputs before running them. Publication is a separate
step from fitting; see [Release a model](release.md).

For catalog preparation, call `ModelSpec.prepare` from your own script; see
[Model spec](model-spec.md). To run ingestion as a Prefect job, install the
`flows` extra (`pip install "geosave[flows]"`); see [Optional ingestion flows](workflows.md).
The `geosave` CLI provides workspace creation.

To improve what future workspaces receive, edit
`src/geosave/templates/`. Your generated workspace remains an editable
consumer project. The [architecture guide](../contributing/architecture.md) explains that
source/consumer boundary.
