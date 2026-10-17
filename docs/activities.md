# Choose an activity

Start with the data or model you already have. Each guide shows the code,
expected result, and the contracts needed for that task.

## Ingest and organize data

| I have… | I want… | Start here | Output |
| --- | --- | --- | --- |
| An image or vector file | Open it, prepare values, and save a usable copy. | [Ingest files](guides/ingest-files.md) | Dataset or GeoDataFrame, then saved files. |
| An area and time range | Find and load remote imagery. | [STAC search and loading](guides/stac.md) | Item table, then raster. |
| A model spec with acquisition recipes | Acquire all declared inputs on one grid. | [Ingestion flows](guides/workflows.md) | Sample directory with one store per raster. |
| Saved imagery and labels | Index observations and select training windows. | [Catalogs and cuts](guides/catalogs-and-cuts.md) | Item and window GeoDataFrames. |
| Polygons | Build raster labels on the image grid. | [Vector data](basics/vectors.md) | Label raster. |
| Benchmark labels | Download the publisher's original files. | [Benchmarks](guides/benchmarks.md) | Local label directory. |

Opening an existing file and acquiring imagery from a service are separate
activities. A catalog describes assets; it does not contain their pixels.
Reading a window opens the assets it references.

## Prepare model inputs

| Problem | Guide | Decision you make |
| --- | --- | --- |
| Imagery and labels have different grids. | [Raster grids](concepts/raster-grids.md) | Target CRS, resolution, and resampling per variable. |
| Stored integers have scaling or missing values. | [Describe values](basics/attrs.md) | Decode before physical calculations; preserve class codes for targets. |
| Data is larger than memory. | [Large data](guides/large-data.md) | Select the region, variables, and useful chunk sizes. |
| Preparation must match across training and inference. | [Declare model inputs](guides/model-spec.md) | Required bands, preprocessing, named inputs, context, and normalization. |
| A scene is too large for one model call. | [Catalogs and cuts](guides/catalogs-and-cuts.md) | Temporal frames, chip shape, overlap, and prediction merge. |

## Build, train, and reuse models

| Activity | Guide | Output |
| --- | --- | --- |
| Learn through one land-cover segmentation project. | [Notebook learning path](examples.md) | Trained models, input features, prediction maps, and a dataset catalog. |
| Make a model from ordinary PyTorch or named stages. | [Build a model](guides/models.md) | Callable `nn.Module` or `ModelChain`. |
| Attach a task head to a pretrained encoder. | [Build from pretrained](guides/pretrained.md) | Initialized chain ready for your task's training. |
| Run an offline segmentation training example. | [Train a model](guides/training.md) | Fitted Lightning Module and validation metrics. |
| Choose classification, regression, or an unlabeled-data method. | [Training contracts](concepts/training.md) | Sample contracts and method behavior, including experimental MIM scaffolding. |
| Create an editable training project. | [Workspaces](guides/workspaces.md) | Python files and YAML configurations. |
| Save, load, or publish a trained chain. | [Model releases](guides/release.md) | Native release with weights and optional model spec. |

Your Dataset reads the data and returns a sample. The DataModule batches and
prepares tensors, the Module defines loss and metrics, and Lightning's Trainer
runs the loop. See [the complete data flow](concepts/data-flow.md) for how
these steps connect. Use [API reference](reference/index.md) for individual
inputs, outputs, defaults, and errors.
