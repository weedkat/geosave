# Test coverage map

Use this map to find the workflow a suite protects and its source owner. It tracks
suite responsibilities, not individual test names. Update it when a responsibility
moves or a supported workflow changes.

## Hierarchy

```text
tests/                         Source owner
├── geodata/                   src/geosave/geodata/
│   ├── attrs/                 Metadata interpretation and updates
│   ├── core/
│   │   ├── raster/            Native array, dataset and tree accessors
│   │   │   ├── transform/     Spatial, temporal and pixel operations
│   │   │   └── utils/         CF storage, statistics and tensor conversion
│   │   └── vector/            Features, Item rows, catalogs and samples
│   ├── features/              Derived imagery and masks
│   ├── io/                    Opening, storage, paths and resource ownership
│   │   └── formats/           File-format round trips
│   ├── stac/                  Item metadata, search, loading and publication
│   ├── utils/                 Shared geographic and display utilities
│   └── viz/                   Static plots and interactive exploration
├── model/                     src/geosave/model/
├── ml/                        src/geosave/ml/
├── templates/tasks/           src/templates/tasks/ (recipe hierarchy)
├── cli/                       src/geosave/cli/
├── flows/                     src/geosave_flows/
├── examples/                  Consumer workflows in examples/
└── test_*.py                  Package imports, boundaries and installation
```

Directories follow source ownership. Workflow descriptions below connect the
boundaries a caller crosses; they do not introduce a second directory hierarchy.

## Geospatial workflows

| Caller workflow | Protection that warrants coverage | Suites |
| --- | --- | --- |
| Describe and decode imagery | Packing, missing values, units, flags, display and geographic metadata retain their meaning without unintended reads. | [attrs](geodata/attrs/), [raster utilities](geodata/core/raster/utils/) |
| Work with native rasters | Arrays retain band identity; datasets combine compatible variables; trees preserve independent grids and observation timelines. Conversion retains values and explicit dtype choices. | [raster accessors](geodata/core/raster/) |
| Select and transform pixels | Cropping/chipping selects the right pixels and grid; reprojection matches the target; composites preserve coverage intervals; masks preserve missing-data semantics. Lazy operations defer reads. | [transforms](geodata/core/raster/transform/), chips and sampling in [raster](geodata/core/raster/) |
| Select features and training samples | Geographic/time selection, positional sample membership and label rasterization yield the intended features and pixels on the requested grid. | [vector](geodata/core/vector/), [regions](geodata/core/) |
| Derive imagery | Indices and cloud/shadow masks respect units, missing pixels and chunk boundaries. | [features](geodata/features/) |
| Find and load observations | Search selects observations; native Items load their assets with the intended bands, grids, dates and acquisition properties. | [STAC](geodata/stac/), [catalog accessors](geodata/core/vector/) |
| Save, reopen and move data | Writers preserve pixels and representable metadata; readers recover native objects; relative assets remain usable after relocation; invalid batches do not leave partial output. Credentials and resources reach their actual consumers. | [I/O](geodata/io/), [formats](geodata/io/formats/) |
| Describe saved assets as Items | Published geometry, time and band metadata describe actual saved files. Raster accessor publication is currently paused; fixture-created Items do not establish a publication contract. | [STAC publication](geodata/stac/test_publish.py), [Item metadata](geodata/stac/test_item.py) |
| Inspect and share data | Plots render correct classes/RGB and missing pixels; Explorer reads the selected viewport, handles interactions, releases subscriptions and exports usable snapshots. | [visualization](geodata/viz/) |

Accessor persistence tests check the public entry point, return value and deferred
execution. Format tests check encoding and metadata round trips. STAC tests check
observation/asset translation. A second test is justified only when it protects a
distinct failure at one of these boundaries.

## Modeling and delivery workflows

| Caller workflow | Protection that warrants coverage | Suites |
| --- | --- | --- |
| Assemble and execute a model | Native values, shapes and gradients flow through stages and chains; a recipe executes its selected dependency graph once. | [model](model/), [spec](model/spec/), [chain](model/chain/) |
| Train and evaluate | Builders create usable components; metrics measure the intended task; augmentation transforms images and targets together; callbacks expose predictions. | [ML](ml/) |
| Generate and run a task recipe | Classification, segmentation, regression and pretraining retain their own target, loss, checkpoint and prediction semantics. | [templates](templates/tasks/) |
| Release and reload a model | Saved configuration and weights reconstruct a usable model and its raster processing workflow. | [release](model/release/) |
| Create a workspace | CLI output is usable and preserves user files; packaged templates can be discovered and instantiated. | [CLI](cli/), root packaging tests |
| Orchestrate geospatial work | Flow inputs define the intended regions and label tables. | [flows](flows/) |
| Follow a documented example | Consumer code still completes its real workflow through public APIs. | [examples](examples/) |
| Import only the needed layer | Geodata does not pull in training dependencies; model and orchestration boundaries remain usable independently. | Root import and layering tests |

## Retention and execution

For each group, trace the caller input, operation and consumed result before
changing tests. Keep a case when a supported use case or demonstrated correctness
failure needs its protection. Remove cases that only assert fixture behavior,
private call routing, retired APIs or native-library behavior. Merge duplicate
protection while retaining distinct metadata, numerical and execution failures.
An unexplained failure needs investigation, not deletion.

Helpers can prepare inputs, but must not manufacture the behavior under test.
Check the production operation actually runs. Deferred execution needs evidence
that work waits for compute, followed by a usable result; checking for a chunked
object alone is insufficient.

`pyproject.toml` owns execution categories:

- `uv run pytest tests/path/`: focused checks; the default excludes `slow` and
  `integration`.
- `uv run pytest`: the default suite across these groups.
- `uv run pytest -m slow`: fresh processes, local servers or large-model checks.
- `uv run pytest -m integration`: external-service checks requiring their runtime
  and credentials.

Markers describe execution needs, not architectural layers. Report failures,
xfails, skipped checks and incomplete runs separately. A passing suite does not
establish that its asserted use cases belong in the architecture.

The [dated audit](../docs/contributing/test-audit.md) records cleanup decisions and
verification results. [Architecture](../docs/contributing/architecture.md) records
current contracts; [open decisions](../docs/contributing/open-decisions.md) records
unresolved designs.
