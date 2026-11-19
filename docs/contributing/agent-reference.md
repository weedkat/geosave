# Task-specific contributor guidance

Read the sections linked from [AGENTS.md](https://github.com/weedkat/geosave/blob/main/AGENTS.md) when working on
those areas. These project constraints supplement the root workflow. Check
paths and implementation details against the current source.

## Structure

```text
src/geosave/
├── cli/          # CLI and AI workspace generation
├── geodata/      # raster/vector data, I/O, STAC, transforms, metadata
└── ml/           # chain, components, raster processing, releases and reusable training support

src/geosave/geodata/
├── core/         # concrete raster/vector accessors, conventions and GeoRegion
├── attrs/        # attr groups: keys that mean one value together (nodata, packing, legend, ...); ACDD
├── io/           # readers and writers; storage.py turns every path or URL into a UPath
├── sampling/     # ChipLayout; RasterChipper/RasterMerger and TensorChipper/TensorMerger
├── stac/         # search, ODC loading and stamps, catalog conversion, explicit Item/asset publication
└── features/     # spectral indices and masks

src/geosave_flows/  # Prefect flows that run a released pipeline; installed with the `flows` extra
src/templates/      # source for generated workspaces; bundled CLI resources

tests/            # mirrors source; flows/ and templates/ mirror the sibling source directories
workspace/        # generated consumer workspace, not library source
docs/contributing/  # architecture, open decisions, maintenance ledger and design specs (published)
```

`src/templates/` is the source for generated workspace files.
It contains `common/` startup and `tasks/` templates copied by `geosave add-task`.
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
import time (`.gs.to_tensor` imports it when called). The inference modules
inside `ml` (chain, components, processing, release) do not import its training
support, so loading a release does not load Lightning training code.

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

A training setup is one project-owned LightningModule, DataModule, and Dataset,
kept together in `modules/<method>/` in the generated workspace. Each project
owns one data preparation workflow and one AI task, recorded in `project.yaml`.
Additional methods must belong to that same task. The
source recipes live in `templates/tasks/<task>/<method>/`; `geosave create`
and `geosave add-task` select the task and then the method. Recipes call reusable
library operations directly; there is no library-owned `ml/tasks` package.

Within `ml`, `factory.py` owns BuildSpec and Registry; `chain`, `components`,
`processing` and `release` own inference. Training support lives in `metrics`,
`builders`, `criterion`, `transforms`, `callbacks` and `cli`. Each method's `data.py`
declares the samples its Datasets return. Semi-supervised methods carry a local
supervised base; editing one method must not change another method's files.
Generated configs and scripts use separate method paths.

Preparation shared by training and inference implements
`ml/processing/raster/preprocessing/base.py`'s `RasterPreprocessor`.
A sub-processor's `__call__` carries `@chain_step()` and returns a named
variable (`sample`, `output`, `product`). A release stores each
sub-processor's spec, a library one by name and a project's own by class path,
and `load_pipeline` rebuilds the raster pipeline around its model.

ModelSpec is unavailable. Template DataModules now use explicit preprocessors;
most Datasets are placeholders and supervised segmentation's sample integration
is unfinished. Check [recipe status](../reference/training.md). Workspaces
compose GeoSave and Lightning APIs and own their training implementations.

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
