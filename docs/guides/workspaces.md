# Create a Lightning training workspace

Generate editable Python code for a small offline training run, then replace
its Dataset and Module with your project logic.

**You need:** `geosave[ml]`. The Lightning boilerplate needs no downloads.
ModelSpec-based geospatial starters are paused; use this boilerplate or the
[complete raster training tutorial](training.md) for a working starting point.

## Generate and run the starter

```bash
geosave create demo --task lightning --method boilerplate
cd demo
python main.py fit --config configs/lightning-boilerplate.yaml --trainer.max_epochs 1
```

The command generates code without creating an environment or installing
packages. Run it in the environment where you installed GeoSave.

```text
demo/
├── main.py
├── modules/lightning/boilerplate/
│   ├── module.py
│   ├── data.py
│   ├── datasets.py
│   └── README.md
├── configs/lightning-boilerplate.yaml
├── .env.example
└── .gitignore
```

The first `fit` creates `artifacts/lightning-boilerplate/version_0/` with the
effective `config.yaml`, checkpoints and TensorBoard events.

The starter creates deterministic synthetic samples and trains a small network.
Its result checks the training setup; it is not a geospatial model benchmark.

## Find the files you edit

| File | Responsibility |
| --- | --- |
| `datasets.py` | Read and return one sample, including inputs and targets. |
| `data.py` | Construct loaders, batching and project transforms. |
| `module.py` | Define the model, loss, metrics and optimizer. |
| YAML config | Select class arguments and Trainer settings. |
| `main.py` | Launch the LightningCLI-based training entry point. |

Each run adds a new `version_N/` beside the first. Record your GeoSave version and project
dependencies separately; generation does not create a manifest or lockfile.
`main.py` does not automatically load `.env`; export settings before launch.

For raster inputs, adapt the [Dataset in the training tutorial](training.md#return-one-sample-from-a-dataset)
and [input preparation](model-inputs.md). Inspect one sample and one batch,
then run a short CPU fit before a long experiment.

## Other task selectors

Without task and method options, `geosave create demo` prompts interactively.
Selectors also include segmentation, regression, classification, pretraining,
and blank projects. Their geospatial workflows depend on the disabled ModelSpec
or incomplete data preparation and are not current ready-to-run starters.
See [workspace method status](../reference/training.md).

Use `--task blank` for shared startup files. Non-empty destinations require
`--force`; existing startup files are preserved. Within a workspace,
`geosave add-task` adds a selected method without replacing existing methods
or `main.py`. Generation does not establish that the selected recipe can run.

## Save and share your work

Use Lightning checkpoints for custom training Modules. Use
[native model releases](release.md) when the trained architecture was built from
registered GeoSave stages. The generic `geosave publish` command is a placeholder;
there is no automatic working publication route through the paused geospatial
starter scripts.

Generated files belong to your project. Updating GeoSave does not rewrite them.
Changes to future starters belong in `src/templates/`, not a generated workspace.
