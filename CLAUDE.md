# GeoSave

GeoSave makes geospatial ML development feel like normal Python: ingest geospatial data into standard ecosystem objects, preserve their metadata and lazy behavior, and move from data preparation to Lightning-based model development with minimal glue code.

The project is in active Alpha development. Prefer simple, replaceable designs. Change an obsolete design instead of building compatibility layers around it unless compatibility is explicitly required.

Lead with code. Explain after showing the implementation. When comparing design options, show each option as the code a caller and a maintainer would see, then explain the difference. Always do smoke test before the design is settled. Ignore ipynb file unless asked to. Choose variable and parameter name carefully, don't make naming too verbose. Name a function with a verb for what it does, and a variable for the value it holds, never for its role. Write loops out in full; no comprehension with more than one `for`, no clever one-liners. Put behavior on the class that owns it, and return a plain value rather than a throwaway container class.

## Function Design

Slimmer code comes from removing cases, stale paths and duplicate mechanisms,
not from compact expressions, clever constructs or premature DRY. Readable code
may take more lines.

* Each function has one responsibility. A long function is fine when it follows
  one straightforward path; complexity is the number of cases it serves, not its
  line count.
* When a function becomes complex, split it into steps that each own one
  operation a reader can name, follow and test on its own. Do not create private
  helpers that only relocate lines without adding reuse or readability.
* A union type hint beyond optional `| None`, such as `str | Sequence[str]` or
  `xr.Dataset | xr.DataArray`, is a warning that one function serves several
  cases. Prefer one function per case unless the behavior is genuinely identical.
* Do not pass or return a `dict` or `Mapping` whose keys carry the structure.
  Use the native object (Dataset, GeoDataFrame, pystac Item), a typed value or
  separate parameters. A mapping is acceptable only for genuinely open-ended
  key-value content, such as attrs or STAC properties, or where a dependency's
  API requires one.
* Prefer showing over explaining: code, printed xarray structures, tables and
  diagrams instead of prose, in docs, plans and reviews.

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
└── ml/           # framework-free pipeline stages; one subpackage per ML framework

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
├── io/           # raster/ (Dataset <-> file, codecs/), vector/ (GeoDataFrame <-> file); storage.py turns every path or URL into a UPath
├── sampling/     # temporal samples, chip locations, RasterChipper and RasterMerger
├── stac/         # search, ODC loading and stamps, catalog conversion, Item publication
└── features/     # spectral indices and masks

src/geosave_flows/  # Prefect flows that run a released pipeline; installed with the `flows` extra
src/templates/      # source for generated workspaces; bundled CLI resources

tests/            # mirrors src/geosave; tests/flows mirrors src/geosave_flows
workspace/        # generated consumer workspace, not library source
docs/contributing/  # architecture, open decisions, maintenance ledger and design specs (published)
```

`src/templates/` is the source for generated workspace files.
Keep template files at their final destination-relative paths for `geosave create`.
Public geospatial I/O lives in `geodata/io/`. Internal helpers stay with their
owning package.

`core/raster/transform` owns substantial pixel-changing implementations;
`core/raster/utils` owns structural support such as grid placement. Accessors
own variant-specific checks and traversal. Internal functions used across
modules have plain names; underscore names are module-local. Supported public
API is declared explicitly, not inferred from implementation filenames.

Dependencies point one way: `geodata <- ml`, and `geosave <- geosave_flows`; `geosave` never imports `geosave_flows` or Prefect. `geodata` imports no torch at import time (`.gs.to_tensor()` imports it when called). `ml.pipeline` imports no ML framework; each framework is one subpackage of `ml` (today `ml.torch`), and a new framework is a sibling, never a branch inside `pipeline`. Inside `ml.torch`, `chain <- modules <- prediction`; this inference path never imports `ml.torch.training`.

Naming: a **stage** is one step of the processing pipeline (ingest, preprocess, predict, postprocess, publish); a **module** is an `nn.Module` that a `ModelChain` composes (encoder, decoder, head, monolith); a module's **forward** is what the chain calls on it.

Every location is a `Location` (path, URL or UPath) with typed `StorageOptions`, so local and remote storage take one code path; `io/storage.py` owns that conversion.

STAC table conversion lives in `stac/catalog/` (layout, hrefs, collections, rules, items); publication in
`publish.py`; loading metadata in `stac/stamps/`.
`stac/extensions/` has one module per schema and is the only production importer
of `pystac.extensions`. `StacClient` and `StacTable` discover native Items;
`stac.load_dataset` and `stac.load_native_datasets` load selected Items through ODC. Format
readers/writers live in `io/raster/` and `io/vector/`; `io/raster/codecs/` translates
GeoSave's conventions to each format's and back.
`GeoRegion` describes geometry/time; native `GeoBox` defines exact pixel placement.

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

The generated workspace owns its `training/` kit, `processing/` stages,
and `pipeline.py`. For template layout, sample contracts and shared training
support, see the [ML contributor guide](docs/contributing/agent-reference.md#machine-learning).

## Tests

Tests mirror the source tree:

```text
src/geosave/geodata/core/raster/dataset.py
tests/geodata/core/raster/test_dataset.py
```

Persistence changes require round-trip tests.

Operations that promise lazy execution must test that laziness is preserved.

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
