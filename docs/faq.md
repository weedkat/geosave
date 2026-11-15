# Common questions

## Where do I start with geospatial AI research?

Start with the [offline training tutorial](guides/training.md), then adapt it
to your own image/label pairs. The [notebook learning path](examples.md) records
a real land-cover project but still needs migration to current APIs. The
[training guide](guides/training.md) continues the chip tutorial through a small segmentation model, checkpoint,
and prediction map. For editable project recipes, see [workspaces](guides/workspaces.md).

## What can I use without training a model?

Read and save rasters or vectors, calculate indices, preserve value metadata,
search STAC, build catalogs, and read raster chips. Begin with
[activities](activities.md).

## Which installation do I need?

The base install covers geodata. Add `ml` for tensors,
model stages, and training; `hub` for native releases; `flows` for Prefect jobs;
and `cloud` for S3 or Google Cloud filesystems. `hub` includes `ml`.
See [Installation](install.md) for the extras and commands.

## Do I have to use a GeoSave-specific raster container?

Use native xarray Dataset, DataArray, and DataTree objects and GeoDataFrames.
`import geosave as gs` registers `.gs` accessors on them. Normal native methods
remain available. See [basic objects](basics/index.md).

## Is opening a raster the same as loading all its pixels?

Readers can defer pixel reads through a backend or Dask arrays. Selection
usually stays lazy. `.compute()`, `.load()`, `.gs.to_numpy()`, and
`.gs.to_tensor()` read the selected values. Plotting and writing also need
pixels. `chunks=None` is reader-specific: STAC source loading uses it to request
eager execution. See [lazy execution](concepts/lazy-data.md).

## What band order and tensor shape does GeoSave use?

Select variables explicitly before conversion: `scene[["red", "nir"]]` fixes
that channel order. Dataset variables are identified by name; their iteration
order after persistence is not a model-input contract. A DataArray can carry an
ordered band coordinate. Dataset conversion inserts the band axis before `y, x`:
`(band, y, x)` for a single image and `(time, band, y, x)` for a time series.
The DataLoader adds a batch axis. Some encoders accept only one frame;
follow the selected encoder's contract. See [rasters](basics/rasters.md) and
[pretrained models](guides/pretrained.md).

## Which pretrained encoder should I choose?

Use the [encoder comparison](guides/pretrained.md#choose-an-encoder) to match
bands, temporal frames, context inputs, and available weights. `list_models()`
lists registered adapter names; `model_name` chooses a variant. Model accuracy
and resource comparisons are planned for a separate benchmark repository.

## Is self-supervised pretraining ready for research comparisons?

Masked image modeling is experimental scaffolding. The current tests exercise
masking, reconstruction loss, and training wiring; a validated encoder-transfer
recipe is still being developed. See [training contracts](concepts/training.md#experimental-masked-image-modeling)
for the current method's inputs and behavior.

## How do I move a custom architecture into deployment?

Prototype it in ordinary PyTorch, a notebook, or a workspace. For supported
deployment, integrate the model implementation and registered factory into
GeoSave first. A native release contains construction data, weights and
processor settings, without executable custom Python. See [model releases](guides/release.md).

## Why does converting to a tensor leave my stored integers unchanged?

Conversion does not decode nodata or scale and offset. Use
`scene.gs.mask_and_scale()` explicitly when the model expects physical values.
Keep integer class targets separate; decoded missing labels need an explicit
conversion back to the task's ignore convention.
See [value metadata](basics/attrs.md) and [training targets](concepts/training.md).

## Can I combine imagery and labels just because their shapes match?

Match CRS, transform, and pixel shape. Equal array sizes can describe different
locations. A tree retains separate native grids until explicitly aligned.
Choose nearest resampling for categorical labels and a suitable method for
continuous imagery. See [the alignment recipe](guides/align-rasters.md).

## Does a catalog contain the imagery? Can I move it?

An Item table stores metadata and asset addresses; pixels stay in their
source files. Item catalogs can use relative hrefs that resolve against the
catalog's location. Move the catalog and its asset folders together. Catalog sampling is currently unavailable; select rows with GeoPandas and
load their assets explicitly. Moving files does not provide credentials for remote assets.

## Will GeoSave make my Dataset and train/validation split?

Supply your own torch Dataset for each split. It returns named model-ready inputs
and targets where needed. The DataModule batches and prepares
tensors; Lightning's Trainer runs learning. Choose separate scenes or geographic
regions for validation to avoid overlap with training. See [Training](guides/training.md).

## Are pretrained weights, a Lightning checkpoint, and a release interchangeable?

Encoder weights initialize feature extraction; your new task head still needs
training. Lightning checkpoints include training state for the task. A native
release saves a reconstructable chain and its weights for reuse.
Use [pretrained models](guides/pretrained.md) to choose the loading path and
[releases](guides/release.md) for save/load examples.

## Does a saved model run all preprocessing automatically?

Only when the release carries a processor. `load_processor(repo)` rebuilds the
model with its registered ingestor, preprocessor and postprocessor, and
`predict_path` runs preparation, chipping, merging and decoding. Import the
module that registers those classes first. A release without a processor
forwards named tensors to its chain and prepares nothing. See
[run the model with a processor](guides/release.md#run-the-model-with-a-processor).

## How do I find exact parameters, defaults, and return values?

Use the [task-based API reference](reference/index.md), or the
[symbol inventory](reference/public-api.md) when you know a name. Check the format's
options class when a reader or writer forwards `**options`. Constructors for
validated specs use their documented fields as keyword inputs. The reference
is generated from source; [documentation maintenance](contributing/documentation.md)
explains how to refresh it.
