# Common questions

## Where do I start with geospatial AI research?

Follow the [notebook learning path](examples.md). Start with real image/label
pairs and a small model, then learn stage composition, feature preparation,
prediction maps, and acquisition for your own labels. The
[training guide](guides/training.md) separately shows the library's supervised
task pair and its sample dictionary contract.

## What can I use without training a model?

Read and save rasters or vectors, calculate indices, preserve value metadata,
search STAC, build catalogs, and select windows. Begin with
[activities](activities.md).

## Which installation do I need?

The base install covers geodata and model specifications. Add `ml` for tensors,
model stages, and training; `hub` for native releases; `flows` for Prefect jobs;
and `cloud` for S3 or Google Cloud filesystems. `hub` includes `ml`.
See [Installation](install.md) for the extras and commands.

## Do I have to use a GeoSave-specific raster container?

Use native xarray Dataset, DataArray, and DataTree objects and GeoDataFrames.
`import geosave as gs` registers `.gs` accessors on them. Normal native methods
remain available. See [basic objects](basics/index.md#know-the-objects).

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
GeoSave first. A native release contains construction data and weights, without
executable custom Python. See [model releases](guides/release.md).

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
continuous imagery. See [raster grids](concepts/raster-grids.md).

## Does a catalog contain the imagery? Can I move it?

An Item table stores metadata and asset addresses. Stack rows store source
selection and target grids. Pixels stay in their source files. Item catalogs
can use relative hrefs that resolve against the catalog's location; window
sources retain absolute addresses. Moving a table alone does not move those
files or provide credentials. See [catalog rules](concepts/catalog-windows.md).

## Will GeoSave make my Dataset and train/validation split?

Supply your own torch Dataset for each split. It returns named model-ready inputs
and targets where needed. The DataModule batches and prepares
tensors; Lightning's Trainer runs learning. Choose separate scenes or geographic
regions for validation to avoid overlap with training. See [Training](guides/training.md).

## Are pretrained weights, a Lightning checkpoint, and a release interchangeable?

Encoder weights initialize feature extraction; your new task head still needs
training. Lightning checkpoints include training state for the task. A native
release saves a reconstructable chain and optional model spec for reuse.
Use [pretrained models](guides/pretrained.md) to choose the loading path and
[releases](guides/release.md) for save/load examples.

## Does a saved model run all preprocessing automatically?

A ModelSpec declares separate raster preprocessing, named inputs, context,
and tensor transforms. Use its processing methods explicitly. A release's
forward call does not automatically normalize tensors using the spec. Match
the training input contract before calling the model. See
[model inputs](concepts/model-inputs.md) and [data flow](concepts/data-flow.md).

## How do I find exact parameters, defaults, and return values?

Use [API reference](reference/index.md) by topic, or search the
[public API inventory](reference/public-api.md) by symbol. Check the format's
options class when a reader or writer forwards `**options`. Constructors for
validated specs use their documented fields as keyword inputs. The reference
is generated from source; [documentation maintenance](contributing/documentation.md)
explains how to refresh it.
