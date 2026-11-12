# Task-specific contributor guidance

Read the sections linked from [AGENTS.md](../../AGENTS.md) when working on
those areas. These project constraints supplement the root workflow. Check
paths and implementation details against the current source.

## Structure

```text
src/geosave/
├── cli/          # CLI and AI workspace generation
├── geodata/      # raster/vector data, I/O, STAC, transforms, metadata
├── model/        # what a release contains: chain, registry, spec, release; stages/ holds encoder/decoder/head/monolith
└── ml/           # reusable training builders, criterion, metrics, transforms, callbacks, and CLI

src/geosave/geodata/
├── core/         # `gs` accessors, GeoRegion and Chips; raster/ (dataarray, dataset, datatree, chips, grid), vector/ (vector base, frame, catalog, sampling)
├── attrs/        # attr groups: keys that mean one value together (nodata, packing, legend, ...); ACDD
├── io/           # readers and writers; storage.py turns every path or URL into a UPath
├── stac/         # extensions/ per schema, catalog/item/asset/band metadata, search/
└── features/     # spectral indices and masks

src/geosave_flows/  # Prefect flows using geosave; installed with the `flows` extra
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
`core/raster/utils` owns structural support such as grid placement. Accessors
own variant-specific checks and traversal. Internal functions used across
modules have plain names; underscore names are module-local. Supported public
API is declared explicitly, not inferred from implementation filenames.

Dependencies point one way: `geodata <- model <- ml`, and `geosave <- geosave_flows`; `geosave` never imports `geosave_flows` or Prefect. `geodata` imports no torch at import time (`.gs.to_tensor()` imports it when called), and `model` does not import `ml`, so a release loads and preprocesses without the training stack.

Every location is a `Location` (path, URL or UPath) with typed `StorageOptions`, so local and remote storage take one code path; `io/storage.py` owns that conversion.

STAC metadata lives in `stac/catalog.py`, `item.py`, `asset.py`, and `band.py`.
`stac/extensions/` has one module per schema and is the only production importer
of `pystac.extensions`. `StacClient` and `StacTable` discover native Items;
`stac.load_dataset` and `stac.load_datatree` load selected Items through ODC. Format
readers/writers live in `io/formats/`. The legacy asset reader currently lives
in `stac/publish.py`; consult the active plan before changing that ownership.
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
kept together in `modules/<task>/<method>/` in the generated workspace. The
source recipes live in `templates/tasks/<task>/<method>/`; `geosave create`
and `geosave add-task` select the task and then the method. Recipes call reusable
library operations directly; there is no library-owned `ml/tasks` package.

Each package at the root of `ml/` owns one reusable concern: metrics, builders,
criterion, transforms, callbacks, or CLI. Each generated method's `data.py`
declares the samples its Datasets return. Semi-supervised methods carry a local
supervised base; editing one method must not change another method's files.
Generated configs and scripts use separate method paths.

ModelSpec redesign is on hold until preprocessing/postprocessing code publication
and execution trust are settled. Preserve its current behavior during the
training-template migration. Generated workspaces compose GeoSave and Lightning
APIs and own their experiment-specific training implementations.

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
