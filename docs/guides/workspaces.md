# Create a training workspace

```bash
geosave create demo --workspace custom_lightning --description "Training demo"
cd demo
python main.py fit --config configs/train.yaml
```

Output, after the Lightning startup messages:

```text
`Trainer.fit` stopped: `max_epochs=1` reached.
```

The `custom_lightning` starter trains a small feature classifier on generated
data. It requires no dataset download, pretrained weights, or remote service.
The default config runs one CPU epoch on the bundled training and validation
data. Logging and checkpoints are disabled for this first run.

## Find the files you edit

```text
demo/
├── configs/train.yaml
├── config.yaml
├── modules/
│   ├── task.py
│   └── data.py
├── artifacts/
├── data/
├── logs/
├── notebooks/
├── predictions/
├── scripts/
├── .env
├── geosave.toml
└── main.py
```

`modules/task.py` contains an ordinary LightningModule with the network, loss,
and optimizer. `modules/data.py` supplies the generated examples through a
Dataset and DataLoaders. `configs/train.yaml` selects those classes and sets
Trainer options. `main.py` runs `GeosaveCLI`, a LightningCLI subclass.
The [GeosaveCLI API](../reference/training-tools.md) describes logger and
configuration behavior. `geosave.toml` records the project name, description,
and initial starter.
After fitting, `config.yaml` records the effective run configuration.

For example, extend the run without editing Python:

```bash
python main.py fit --config configs/train.yaml \
  --trainer.max_epochs 3 --trainer.default_root_dir artifacts/run-2
```

This uses all four training batches per epoch and both validation batches.
The current CLI parses the starter's `limit_train_batches: 1` and
`limit_val_batches: 1` as fractions (`1.0`, meaning all batches). Choose a new
`default_root_dir` for each run: LightningCLI saves its effective config there
and refuses to overwrite an earlier run's config. To change the data source,
edit the Dataset in `modules/data.py`; to change training behavior, edit
`modules/task.py`.

## Choose a starter for your task

| Starter | Provides | Before the first fit |
| --- | --- | --- |
| `custom_lightning` | Editable Lightning Module/DataModule and a synthetic example. | Run the command above. |
| `segmentation` | Configs selecting GeoSave's supervised segmentation pair, plus example-preparation and publication scripts. | Implement the configured Dataset classes and supply suitable data; review encoder weight downloads. |
| `regression` | Configs selecting GeoSave's supervised pixelwise regression pair. | Implement the configured Dataset classes and supply suitable data; review encoder weight downloads. |
| `blank` | Shared startup and empty project directories. | Write or select a Lightning Module and DataModule, and create a run config. |

Without `--workspace`, the CLI prompts for a starter. Use `-w` for the short
option and `-d` for the description. To create a segmentation project:

```bash
geosave create landcover -w segmentation -d "Land cover training"
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

## Keep a run configuration

```yaml
# An excerpt from the custom_lightning starter's configs/train.yaml
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

For logged runs, remove `logger: false` from the starter config. `GeosaveCLI`
then selects TensorBoard under `artifacts/<model_name>/`. Set
`MLFLOW_TRACKING_URI` if you also want an MLflow logger, or configure Lightning
loggers explicitly. Choose checkpoint behavior through Trainer options and
Lightning callbacks.

## Optional scripts and publication

The segmentation starter includes `scripts/prepare_example.py`, a small
synthetic reflectance-preprocessing example, and `scripts/publish.py`, which
uploads a trained model release and selected workspace files to Hugging Face. Review
these scripts and their inputs before running them. Publication is a separate
step from fitting; see [Release a model](release.md).

For catalog preparation, call `ModelSpec.prepare` from your own script; see
[Model spec](model-spec.md). To run ingestion as a Prefect job, install the
`flows` extra (`pip install "geosave[flows]"`); see [Optional ingestion flows](workflows.md).
The `geosave` CLI provides workspace creation and `make` for copying available
scaffolds. This checkout bundles no scaffold files.

To improve what future workspaces receive, edit
`src/geosave/templates/`. Your generated workspace remains an editable
consumer project. The [architecture guide](../contributing/architecture.md) explains that
source/consumer boundary.
