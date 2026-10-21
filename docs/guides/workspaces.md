# Create a training workspace

GeoSave generates editable Python training code. Choose a task, then a method:

```bash
geosave create demo
```

The terminal selector offers segmentation (supervised or FixMatch), regression
(supervised or Mean Teacher), classification (supervised), pretraining
(experimental MIM), Lightning (boilerplate), and a blank workspace.
Omitted options prompt for the missing selection. For scripts, provide both:

```bash
geosave create landcover --task segmentation --method supervised
```

Use `--task blank` to create shared startup files without choosing a method.
Invalid combinations are rejected before writing files. Non-empty directories
require `--force`; existing startup files are preserved.

## Find the files you edit

```text
landcover/
├── main.py
├── modules/segmentation/supervised/
│   ├── module.py
│   ├── data.py
│   ├── datasets.py
│   └── README.md
├── configs/
│   ├── segmentation-supervised.yaml
│   └── segmentation-supervised/model_spec.yaml
├── scripts/segmentation/supervised/
│   ├── prepare_example.py
│   └── publish.py
├── artifacts/
├── data/
├── notebooks/
├── .env.example
└── .gitignore
```

`module.py` owns training, loss, metrics, and optimizer behavior. `data.py` owns
batching and device-side augmentation; `datasets.py` owns sample reads. These
classes are local implementations you can edit, using reusable GeoSave helpers.
FixMatch and Mean Teacher include a local supervised base in their own package.
Changing that base does not change a separate supervised recipe.

Implement the geospatial Dataset stubs and supply data before fitting. The default
encoder uses pretrained weights and can download them. See each method's README
for target shapes and method requirements.

`main.py` uses `GeosaveCLI`, a LightningCLI subclass. Each run saves its effective
configuration under `artifacts/<model_name>/version_N/`. `.gitignore` excludes
local environment files, data, and training artifacts.

```bash
cd landcover
python main.py fit --config configs/segmentation-supervised.yaml
```

## Add another method

Inside the workspace, run `geosave add-task` for the same task/method selectors,
or provide explicit options:

```bash
geosave add-task --task segmentation --method fixmatch
python main.py fit --config configs/segmentation-fixmatch.yaml
```

This adds `modules/segmentation/fixmatch/`, its config and processing document,
and `scripts/segmentation/fixmatch/publish.py`. It leaves existing methods and
`main.py` untouched. Existing methods are disabled in the selector; selecting one
explicitly reports that it already exists. To share a Dataset, point compatible
method configurations at the same project Dataset class.

## Run the offline starter

The Lightning boilerplate needs no downloads or external dataset:

```bash
geosave create demo --task lightning --method boilerplate
cd demo
python main.py fit --config configs/lightning-boilerplate.yaml --trainer.max_epochs 1
```

It supplies separate editable Module, DataModule, and deterministic Dataset
files under `modules/lightning/boilerplate/`.

## Publish a native model

For a supported registered model architecture:

```bash
python scripts/segmentation/supervised/publish.py your-account/your-model \
  artifacts/segmentation-supervised/version_0/checkpoints/best.ckpt
```

The script loads the local Module checkpoint and exports its native trained
model. The repository root holds inference weights/config and the current
processing document; `workspace/` holds the selected training workspace and
run checkpoints. Deployment does not import uploaded training code.
Mean Teacher's starter exports the student; see its README before choosing EMA
teacher deployment. Training-only calibration state is not automatically part
of a chain-only release.

ModelSpec behavior is preserved for now. Its redesign and publication of custom
preprocessing/postprocessing code are deferred; see
[open decisions](../contributing/open-decisions.md). The existing release
format still requires registered model stages.

## Template ownership

Edit generated project files freely. Future starter changes belong in
`src/templates/tasks/<task>/<method>/`; reusable library operations
remain in GeoSave. Updating GeoSave does not overwrite an existing project.
