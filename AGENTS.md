# GeoSave

GeoSave makes geospatial ML development feel like normal Python: ingest geospatial data into standard ecosystem objects, preserve their metadata and lazy behavior, and move from data preparation to Lightning-based model development with minimal glue code.

The project is in active Alpha development. Prefer simple, replaceable designs. Change an obsolete design instead of building compatibility layers around it unless compatibility is explicitly required.

Lead with code. Explain after showing the implementation. When comparing design options, show each option as the code a caller and a maintainer would see, then explain the difference. Always do smoke test before the design is settled. Ignore ipynb file unless asked to. Choose variable and parameter name carefully, don't make naming too verbose. Name a function with a verb for what it does, and a variable for the value it holds, never for its role. Write loops out in full; no comprehension with more than one `for`, no clever one-liners. Put behavior on the class that owns it, and return a plain value rather than a throwaway container class.

## Commands

```bash
uv sync                      # dev group pulls geosave[all]
uv run pytest
uv run pytest tests/path/to/test_file.py
uv run pytest -m integration
uv run pytest -m slow

uv run ruff check .
uv run pre-commit run --all-files
```

`pyproject.toml` is the source of truth for dependencies and tool configuration.

## Structure

```text
src/geosave/
├── cli/          # CLI and AI workspace generation
├── geodata/      # raster/vector data, I/O, STAC, transforms, metadata
├── model/        # what a release contains: chain, registry, spec, release; stages/ holds encoder/decoder/head/monolith
└── ml/           # reusable training builders, criterion, metrics, transforms, callbacks, and CLI

src/geosave/geodata/
├── core/         # `gs` accessors, GeoAnchor and Chips; raster/ (dataarray, dataset, datatree, chips, grid), vector/ (vector base, frame, catalog, sampling)
├── attrs/        # attr groups: keys that mean one value together (nodata, packing, legend, ...); ACDD
├── io/           # readers and writers; storage.py turns every path or URL into a UPath
├── stac/         # extensions/ per schema, catalog/item/asset/band metadata, search/
├── transform/    # explicit operations, one module per thing changed: warp, extent, combine, nodata, temporal, color, vector
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

Dependencies point one way: `geodata <- model <- ml`, and `geosave <- geosave_flows`; `geosave` never imports `geosave_flows` or Prefect. `geodata` imports no torch at import time (`.gs.to_tensor()` imports it when called), and `model` does not import `ml`, so a release loads and preprocesses without the training stack.

Every location is a `Location` (path, URL or UPath) with typed `StorageOptions`, so local and remote storage take one code path; `io/storage.py` owns that conversion.

STAC metadata lives in `stac/catalog.py`, `item.py`, `asset.py`, and `band.py`.
`stac/extensions/` has one module per schema and is the only production importer
of `pystac.extensions`. `StacClient` and `StacTable` discover native Items;
`io.load_dataset` and `io.load_datatree` load selected Items through ODC. Format
readers/writers live in `io/formats/`, and `io/stac.py` owns raster asset I/O.
`GeoAnchor` describes geometry/time; native `GeoBox` defines exact pixel placement.

## Design

GeoSave composes the existing geospatial and ML ecosystem instead of replacing it with parallel abstractions.

```text
Raster representation   -> xarray Dataset / DataArray
Vector representation   -> GeoDataFrame
Training                -> PyTorch Lightning
Workflow orchestration  -> Prefect
Model publishing        -> Hugging Face Hub
Remote training data    -> LitData
Visualization           -> HoloViz ecosystem
```

Prefer native ecosystem objects in public APIs. Introduce a GeoSave-specific wrapper only when it owns an invariant that the underlying object cannot represent cleanly.

Use capabilities provided by dependencies before implementing equivalent behavior.

## Geospatial Contracts

Raster operations should remain lazy when possible.

Do not eagerly materialize an entire raster unless the operation requires it.

Preserve metadata required for CF/GDAL-compatible round trips, including applicable CRS, coordinates, spatial dimensions, transform, nodata, dtype, and band or variable identity.

Reprojection, resampling, dtype conversion, coordinate changes, and eager computation are explicit operations.

Built-in persistence:

```text
Raster -> Zarr / NetCDF / GeoTIFF
Vector -> Parquet / GeoParquet
```

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

## Tests

Tests mirror the source tree:

```text
src/geosave/geodata/core/raster/dataset.py
tests/geodata/core/raster/test_dataset.py
```

Persistence changes require round-trip tests.

Operations that promise lazy execution must test that laziness is preserved.

Choose tests from core workflows and meaningful failure modes: wrong values or
grids, lost metadata, eager reads, broken training, and unusable saved artifacts.
Each test must add distinct protection beyond the surviving tests. Prefer small
real examples and representative cases over exhaustive combinations.

Do not add a test just because a method exists. Avoid re-testing dependency
validation, private helper structure, incidental defaults or layout, exact export
lists, removed names, and internal call paths. Preserve checks that prevent
silent data corruption or unintended execution. An intentional contract change
should update its tests; existing tests are evidence to assess, not a reason to
preserve an obsolete design.

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

## Changes

* Inspect the relevant implementation and tests first.
* Replace a wrong abstraction instead of adding adapters around it.
* Do not add compatibility aliases or duplicate execution paths unless required.
* Keep unrelated code unchanged.
* Use existing dependencies before adding new ones.
* Do not discard unrelated working-tree changes.

## Release

Packaging uses Hatchling.

PyPI releases are handled by the existing GitHub Actions workflow.

Documentation uses Zensical.

## Handoff

Report only:

1. What changed.
2. Checks run.
3. Breaking changes or remaining risks.
