# Task-specific contributor guidance

Read the sections linked from [AGENTS.md](https://github.com/weedkat/geosave/blob/main/AGENTS.md) when working on
those areas. These project constraints supplement the root workflow. Check
paths and implementation details against the current source.

## Structure

```text
src/geosave/
├── cli/          # CLI and AI workspace generation
├── geodata/      # raster/vector data, I/O, STAC, transforms, metadata
└── ml/           # framework-free processing stages; one subpackage per ML framework

src/geosave/ml/
├── pipeline/     # Pipeline and its framework-free stages, one package per stage:
│   ├── ingestion/        # one ingestor per source collection
│   ├── preprocessing/    # one preprocessor per model input
│   ├── prediction/       # the predictor contract; predictors live with their framework
│   ├── postprocessing/   # one postprocessor per task
│   └── publishing/       # one publisher per product format
└── torch/        # everything that imports torch
    ├── chain/        # the module contract (Published, forward type hints), its registry,
    │                 # ModelChain, configured construction and list_models
    ├── modules/      # the library's modules: encoder/decoder/head/monolith
    ├── prediction/   # TorchPredictor: runs a model over chips in the pipeline
    ├── training/     # optim, criterion, metrics, transforms, callbacks, cli
    └── typing.py     # ChainModule (the module contract) and the types modules pass (Image, FeatureMap, ...)

src/geosave/geodata/
├── core/         # concrete raster/vector accessors, conventions and GeoRegion
├── attrs/        # attr groups: keys that mean one value together (nodata, packing, legend, ...); ACDD
├── io/           # readers and writers; storage.py turns every path or URL into a UPath
├── sampling/     # temporal samples, chip locations, RasterChipper and RasterMerger
├── stac/         # search, ODC loading and stamps, catalog conversion, explicit Item/asset publication
└── features/     # spectral indices and masks

src/geosave_flows/  # Prefect flows that run a released pipeline; installed with the `flows` extra
src/templates/      # source for generated workspaces; bundled CLI resources

tests/            # mirrors source; flows/ and templates/ mirror the sibling source directories
workspace/        # generated consumer workspace, not library source
docs/contributing/  # architecture, open decisions, maintenance ledger and design specs (published)
```

`src/templates/` is the source for generated workspace files.
The CLI copies `common/` for every workspace, `tasks/<task>/common/` when
present, and `tasks/<task>/methods/<method>/` for the selected method.
Author files at their final workspace paths.
Public geospatial I/O lives in `geodata/io/`. Internal helpers stay with their
owning package.

`core/raster/transform` owns substantial pixel-changing implementations;
`core/raster/utils` owns structural support such as grid placement.
`core/raster/rules/{common,dataarray,dataset,datatree}.py` owns validation-only
callables consumed by `.gs.validate`; parameterized rules such as
`compatible_concat(dim)` use the existing `Rule[T]` callable contract.
Accessors own variant-specific operations, argument checks and traversal.
Rules do not mutate inputs, execute pixel tasks or return transformed data.
Internal functions used across
modules have plain names; underscore names are module-local. Supported public
API is declared explicitly, not inferred from implementation filenames.

Dependencies point one way: `geodata <- ml`, and `geosave <- geosave_flows`.
`geosave` never imports `geosave_flows` or Prefect. Geodata imports no torch at
import time (`.gs.to_tensor` imports it when called). `ml.pipeline`
imports no ML framework; framework code lives in its own
subpackage (`ml.torch`). The torch inference modules (chain, modules,
prediction) do not import `ml.torch.training`, so loading a release does not load
Lightning training code.

Every location is a `Location` (path, URL or UPath) with typed `StorageOptions`, so local and remote storage take one code path; `io/storage.py` owns that conversion.

STAC table conversion lives in `stac/catalog/` (layout, hrefs, collections, rules, items);
publication in `stac/publish.py`; loading metadata in `stac/stamps/`.
`stac/extensions/` has one module per schema and is the only production importer
of `pystac.extensions`. `StacClient` and `StacTable` discover native Items;
`stac.load_dataset` and `stac.load_native_datasets` load selected Items through ODC. Format
readers/writers live in `io/raster/` and `io/vector/`. `stac/publish.py` owns `StacItem` (`from_region`, `add_raster_asset`).
`stac.publish_item` writes a native Item plus explicit asset Datasets; loading returns Datasets without reconstructing a tree;
general DataTree accessors impose no STAC layout.
`GeoRegion` describes geometry/time; native `GeoBox` defines exact pixel placement.

## Machine Learning

Use Lightning as the training framework rather than building a parallel training loop.

```text
LightningModule -> training behavior
Trainer         -> orchestration
Callback        -> reusable lifecycle behavior
LightningCLI    -> workspace entry point and configuration
Logger          -> experiment logging
```

A kit supplies a project-owned LightningModule, DataModule, Datasets and any
custom callbacks under `training/`. Its configuration lives in `configs/`.
Task and method select a starter at creation. The user owns the generated
project and its subsequent organization.

Geospatial starters ship a root `pipeline.py` and a `processing/` package.
Keep ingestion, preprocessing, prediction, postprocessing and persistence
in their respective modules. The Dataset can call the same input preparation
as the Pipeline. Keep training-only augmentation in the DataModule.
Store shared task code in `templates/tasks/<task>/common/`. Method-specific
files go in `methods/<method>/`. Each copied file has one owner.

Within `ml`, `pipeline` owns the framework-free stages. `ml.torch` holds `chain`, `modules` and `prediction` for
inference, and `training` (`metrics`, `optim`, `criterion`, `transforms`,
`callbacks`, `cli`) for Lightning. A new framework gets a sibling of `torch/`;
it never adds framework branches to `pipeline`.

Naming: a **stage** is one step of the processing pipeline (ingest,
preprocess, predict, postprocess, publish); a **module** is an `nn.Module`
a `ModelChain` composes (encoder, decoder, head, monolith); a module's **forward** is what the chain calls on it.

`ml.pipeline.Pipeline` connects five stages, each identified by one thing:
the ingestor by its source collection, the preprocessor by the model input, the
predictor by its ML framework, the postprocessor by the task and the publisher
by the product format. Stage contracts are designed one at a time; undesigned
methods raise `NotImplementedError`. Collection readers implement the
`ml.pipeline.ingestion.Ingestor` protocol.

Check the [recipe status](../reference/training.md) for supported stages and
project Dataset requirements.

## Public API and Docstrings

Public APIs use concise Google-style docstrings.

Show the public object users work with rather than inspecting its internal attributes.

```python
def rename_vars(self, mapping: dict[str, str]) -> xr.Dataset:
    """Rename data variables without changing their order or pixels.

    Args:
        mapping: Existing variable names mapped to replacement names.

    Returns:
        Dataset with renamed data variables.

    Raises:
        KeyError: If a source variable is absent.
        ValueError: If a replacement is empty or creates a duplicate.

    Examples:
        >>> renamed = ds.gs.rename_vars({"B04": "red", "B08": "nir"})
        >>> renamed
        <xarray.Dataset> ...
    """
```

Keep obvious accessors short:

```python
@property
def variable_count(self) -> int:
    """Return the number of data variables."""
```

Comments explain non-obvious domain constraints, not syntax or implementation history.

## Release

Packaging uses Hatchling.

PyPI releases are handled by the existing GitHub Actions workflow.

Documentation uses Zensical.
