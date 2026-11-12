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

The geospatial recipes use ModelSpec version 1 and an explicit DataModule `size`.
See [training setup](#geospatial-recipes). Supply data before fitting; the supervised segmentation recipe includes a
paired-raster Dataset, while other recipes expose Dataset extension points. The default
encoder uses pretrained weights and can download them. See each method's README
for target shapes and method requirements.

The generated files use your active Python environment; generation does not
create a dependency manifest or lockfile. Record the GeoSave version and your
project dependencies before sharing the workspace. `main.py` does not load
`.env` automatically; supply needed environment variables when launching it.
See [CLI environment behavior](../reference/cli.md#shared-options).

`main.py` uses `GeosaveCLI`, a LightningCLI subclass. Each run saves its effective
configuration under `artifacts/<model_name>/version_N/`. `.gitignore` excludes
local environment files, data, and checkpoint files. Add `artifacts/` to your
project ignore rules if you also want to exclude logs and saved configurations.

```bash
cd landcover
python main.py fit --config configs/segmentation-supervised.yaml
```

## Geospatial recipes

The segmentation, regression, classification, and MIM starters use ModelSpec version 1.
Their DataModules take an explicit `size` for augmentation and call the recipe's
`normalized` call in every split. Optional catalog sampling and chip
layout use `catalog.gs.sample(...)` and `gs.Chips(...)` outside ModelSpec.

The supervised segmentation starter includes `RasterDataset`: configure paired
`images` and `targets` files, the same `spec` as the DataModule, `num_classes`,
and `ignore_index`. Each image is one selected observation or chip with a
single-band label raster on the same grid. Label nodata is ignored; nonfinite
image pixels, fractional/out-of-range classes, and mismatched grids are
rejected. It neither resamples nor fills pixels. Its default recipe produces
`(band, y, x)` inputs; select dates explicitly before using it.

Other project-owned Dataset implementations remain extension points: supply
your own reads, labels, and sample contract. The [processing workflow](model-spec.md)
includes an offline end-to-end check using a generated regression DataModule
and a tiny temporal model. Existing generated workspaces need the documented
[v3 migration](model-spec.md#migrating-v2-recipes); upgrading the library does
not rewrite copied project files.

## Check a starter before a long run

The starters are editable recipes, not all ready-to-train configurations. Five
geospatial recipes (FixMatch, both regression methods, classification, and MIM)
still encode a time axis while selecting DINOv3, which expects a single image
per sample. Select one observation and use `(band, y, x)` encoding, or choose
and configure a temporal model before fitting. Supervised segmentation already
uses the single-image layout.

All six geospatial specs currently select `[B02, B03, B04, B08]` (blue, green,
red, NIR), while the default pretrained DINOv3 adapter retains RGB channel
semantics. Match band order and normalization to the chosen weights together;
changing only the channel count does not establish that match.

Run one batch through your Dataset, DataModule, and model with
`pretrained=False` before downloading weights or starting a long job. The
[offline tutorial](training.md) is the verified starting path.

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
run checkpoints. The script reads the **current** workspace source and processing
document, so a checkpoint alone does not bind them to the training run. Preserve
the matching recipe and source before editing a trained experiment. Review files
under `configs/`, `modules/`, and `scripts/` before publication: broad upload
patterns can include private files there even when Git ignores them.
Deployment does not import uploaded training code.
Mean Teacher's starter exports the student; see its README before choosing EMA
teacher deployment. Training-only calibration state is not automatically part
of a chain-only release.

ModelSpec version 1 executes named Python calls. Releases save recipes but do
not package or install custom preprocessing code; install required project
modules in the execution environment. The release format still requires
registered model stages.

## Template ownership

Edit generated project files freely. Future starter changes belong in
`src/templates/tasks/<task>/<method>/`; reusable library operations
remain in GeoSave. Updating GeoSave does not overwrite an existing project.
