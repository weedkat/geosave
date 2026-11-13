# Find a task

Start with the data or model you already have. The guides below use current
APIs; paused functionality is listed separately at the end.

## Process imagery and vectors

| I have… | I want… | Start here | Output |
| --- | --- | --- | --- |
| A small satellite scene | Calculate and save NDVI. | [First vegetation map](first-raster.md) | GeoTIFF and plot. |
| An area and date range | Find remote imagery. | [Download imagery](guides/download-imagery.md) | Local GeoTIFF on a chosen grid. |
| Existing raster files | Select pixels, calculate a feature and save. | [Prepare files](guides/ingest-files.md) | Georeferenced raster. |
| Raster or vector data | Inspect locations and values. | [Explore on a map](guides/explore.md) | Interactive viewer. |
| A vector file | Select features and save a table. | [Vector data](basics/vectors.md) | GeoDataFrame and GeoParquet. |
| Image and labels on different grids | Match labels to the image. | [Align rasters](guides/align-rasters.md) | Labels on the exact reference grid. |
| More pixels than memory | Select before computing. | [Large rasters](guides/large-data.md) | Lazy subset and saved result. |

## Prepare training data and predictions

| I have… | I want… | Start here | Output |
| --- | --- | --- | --- |
| One image and label raster | Read training samples. | [Training chips](quickstart.md) | Pixel windows, input tensors and integer targets. |
| A large image | Read overlapping windows and blend predictions. | [Chips and merging](guides/chips.md) | NumPy outputs placed on the source grid. |
| Dated observations | Write assets and keep an index of them. | [Create a catalog](guides/catalogs-and-cuts.md) | TIFFs, Item table and reopened time series. |
| Packed imagery | Decode physical values. | [Describe values](basics/attrs.md) | Floating values with missing pixels masked. |
| A model needing consistent inputs | Reuse preparation at training and inference. | [Prepare model inputs](guides/model-inputs.md) | Named input tensors. |

## Train and reuse models

| Activity | Guide | Output |
| --- | --- | --- |
| Run a complete offline training example. | [Train and map predictions](guides/training.md) | Trained model, checkpoint and prediction TIFF. |
| Compose a PyTorch model or registered stages. | [Build a model](guides/models.md) | Callable module or ModelChain. |
| Attach a head to a pretrained encoder. | [Pretrained models](guides/pretrained.md) | Chain ready for task-specific training. |
| Create an editable starting project. | [Lightning workspace](guides/workspaces.md) | Local Module, DataModule, Dataset and configuration. |
| Save, load or publish a registered trained chain. | [Model releases](guides/release.md) | Construction data and trained weights. |
| Explore the real land-cover teaching project. | [Notebook series and migration status](examples.md#real-land-cover-project) | Project lessons; some notebooks need migration. |

Your Dataset owns reading and preparing one sample. A DataLoader batches it;
a LightningModule defines learning, and Trainer runs the loop.
See [data flow](concepts/data-flow.md) for how the stages fit together.

## Current limitations

ModelSpec, dependent geospatial workspace recipes, catalog sampling, catalog
row pixel viewing, and vector rasterization are paused. They are not required
for the file, chip, training and model workflows above. See
[ModelSpec status](guides/model-spec.md) and [catalog availability](reference/catalog.md#unavailable-operations).

For individual signatures, defaults and return values, use the
[task-based API reference](reference/index.md).
