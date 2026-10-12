> Status 2026-10-11: ModelSpec acquisition, selection, preprocessing, input
> binding and COG preparation are implemented. Normalization statistics,
> catalog folders and Prefect preparation orchestration remain future work.
> Unused Clay sensor metadata and pure-raise dense tasks were removed.
> The details below retain the original proposal; current APIs are documented
> in `docs/guides/model-spec-and-release.md`.

# Prepare: raw catalog to model-ready catalog

Status: Implemented. The catalog layout (one `catalog.parquet`) is superseded by
[catalog per collection](2026-10-09-catalog-per-collection-design.md).

First of three specs for the end-to-end slice (ingest → prepare → train →
predict → release → serve). This one covers ingest, prepare, the dataset that
reads a prepared catalog, and the first two example notebooks. Training with a
small `ModelChain` model, then predict/release/serve, follow as their own specs.

## Problem

- Nothing in the library turns raw data into what a model trains on.
  `prepare_dense_data` and `workflow/tasks/dense.py` raise `NotImplementedError`.
- The segmentation `Dataset` preprocesses every frame in its constructor and
  holds the results in memory, again in every DataLoader worker. That repeats
  the work each run and does not scale past memory.
- Opening a stack from its catalog rows is written by hand wherever it is needed.
- All five example notebooks are stale; nothing runs them.

## Data flow

```text
raw files ──► raw catalog ──► prepared catalog ──► chips ──► model inputs
  COG           item table       item table         window     tensors
                (as acquired)    (one model spec)   table
```

A raw catalog holds data as acquired and serves any model. A prepared catalog
holds what one model spec's preprocessing made, plus the groups kept beside it
(labels). Both are ordinary item tables; a prepared one is read exactly like a
raw one.

## Library changes

### `stac.table.load_stack`

```python
sample = table.load_stack(rows)        # rows of one stack -> DataTree, one group per collection
```

- `rows`: item-table rows sharing one `geosave:stack`.
- Returns the lazy stack `gd.stack` builds from each collection's `table.load`.
- Raises `ValueError` when the rows name more than one stack, or none.
- Takes `storage_options` like `table.load`.

### `ModelSpec.prepare`

```python
catalog = spec.prepare(raw, "data/prepared", keep=["label"])
```

- `raw`: raw item table (GeoDataFrame). A path is read by the caller with
  `read_vector`; cuts and prepare take tables.
- `output`: new directory. Each stack is written to `output/<stack>` as Zarr,
  one store per group; the table goes to `output/catalog.parquet` with one
  Collection per group; the spec is saved to `output/model_spec.yaml`.
- `keep`: raw groups copied through unchanged, such as labels.
- Per stack: `load_stack` → `spec.preprocess` → stack of `spec.input_rasters`
  plus `keep` → `gs.to_zarr` → `create_stack_items`. Nothing else.
- Returns the catalog path.
- Raises `FileExistsError` when `output` exists, and `ValueError` when a stack
  lacks a raster the spec declares or a `keep` group.

Preprocessing runs on the whole stack; frames are then cut from the prepared
catalog. Today it runs per frame. Per-pixel steps give the same result; a
temporal step now runs once over the full series, which is what it means.

Item properties beyond those GeoSave writes are not carried from raw to
prepared Items in this version.

### `supervised.Dataset`

- Reads a prepared catalog: its groups must cover `spec.input_rasters` and
  `target`. A raw catalog raises `ValueError` naming `spec.prepare`.
- `_prepare`, the in-memory prepared parents, the reopen-and-compare check and
  the grid-change check are deleted. Each process opens stacks lazily with
  `load_stack`; a chip is `cuts.select(stack, row)`.
- `reference` is the chips table plus the annotation columns of each stack's
  target row; `times` comes from the cut, not from reopened rasters.
- `DataModule` arguments are unchanged.

### Small fixes

- `read_vector` reads an item table with openable hrefs, as its docstring
  said; it is the one reader for catalogs. `table.read` stays as what it calls.
- `table.write` stores the collection link as `./<file name>`, so a stored
  table names no absolute location; reading resolves it.
- `geosave_engine.geodata` exposes `cuts` and `stac`, loaded on first use so
  importing `geodata` does not import the STAC dependencies.
- `cuts.select_times` reads a single-date sample, whose time is a scalar
  coordinate.
- The GeoTIFF writer marks a band holding a colour table as a palette. A band
  read as gray and then given a legend used to lose its colours on write.
- `unpack` keeps float32: numbers stored as float32 (16-bit integers with
  nodata already blanked) unpack to float32, not float64.

Asset keys are unchanged. An asset is a file and its key names what the file
holds: the variable for a one-variable file, `image` otherwise. Name the
variable (`label`, not rioxarray's default `band_1`) rather than rekey.

### After review

An independent review led to these, each with a test:

- `prepare` writes into a sibling draft folder and renames it when complete,
  so a failed run leaves nothing behind. It refuses an empty table, rows
  naming no stack, a stack named outside `output`, and `keep` given as text.
- `DataModule` compares the `model_spec.yaml` beside a local catalog with its
  own spec and refuses a catalog whose `rasters`, `preprocessing` or `inputs`
  differ, or a folder holding no spec. Group names alone cannot tell a raw
  catalog from a prepared one when an input reads a raster directly.
- `Dataset.parents` opens a sample when a chip first asks for it and keeps at
  most 64 open, instead of opening the whole catalog in every worker.
- `read_vector` treats a table as an item table only when it has both
  `stac_version` and `assets`, and imports the STAC package only then.
- `table.write` relativises the collection link whether or not Collections
  are passed, and no longer stores a filtered table's index as a property.

## Examples

Notebooks are the single source and the suite executes them.

| Notebook | Shows | Runs |
| --- | --- | --- |
| `01_imagery.ipynb` | For readers from GIS or computer vision: what a raster is here, searching a STAC API, loading onto a grid, labels and legends, stacking, writing COGs, the raw catalog | `-m integration` (network) |
| `02_prepare.ipynb` | The model spec block by block next to the plain calls it becomes, each pair asserted equal; then `spec.prepare`, cuts, one chip, the dataset | `-m slow`, offline |
| `00_quickstart.ipynb` | The whole path in Python only: prepare, a small model, `Trainer.fit` | `-m slow`, offline |
| `03_workspace.ipynb` | `geosave create`, the config files, an overlay config, `main.py fit` | `-m slow`, offline |
| `04_model.ipynb` | A hand-written chain, the registry, a pretrained DINOv3 chain, save and load a release | `-m integration` (weights download) |
| `05_train.ipynb` | Module, DataModule, Trainer and callbacks in Python and as YAML; evaluate and resume | `-m slow`, offline |

- `01_imagery.ipynb` is the existing introduction, kept and extended; running
  it regenerates `examples/data/dw_raw/`.
- `02_prepare.ipynb` reads `examples/data/dw_raw/`, committed: three Dynamic
  World samples of 510×510, four Sentinel-2 bands stored as delivered with
  their reflectance scale recorded, 6.3 MB.
- `examples/configs/model_spec.yaml` is the spec the notebooks use.
- `tests/examples/test_notebooks.py` executes each notebook in a temporary
  copy of the data and configs with `nbclient`, so a run writes nothing into
  the repository.
- Notebooks are committed with outputs cleared.
- `02_pipeline.ipynb` is deleted.
- New dev dependencies: `nbclient`, `ipykernel`.

## Tests

- `load_stack`: groups by collection; lazy; rejects several stacks.
- `spec.prepare`: round trip (prepared pixels equal `spec.preprocess` of the
  raw stack; kept group equals the raw one); stays lazy until the write;
  existing output and missing groups raise; the catalog moves with its folder.
- `Dataset`: reads a prepared catalog without computing at construction;
  rejects a raw catalog; existing dataset and module tests prepare their
  fixtures first.
- The fixes above, each with a test that failed first.
- Both notebooks execute.

## Out of scope

- Normalization statistics from the prepared catalog and removing
  `geodata/sensors` (next spec, with training).
- Prefect flows. `prepare_dense_data` is rewritten as a wrapper over
  `spec.prepare` once the notebooks have settled the calls.
- The COG media type. `create_items` types a `.tif` as GeoTIFF because a path
  does not say whether the file is cloud-optimized; left as is.
- `zarr:zarr_format` reading as `3.0` in a pandas table; the stored value is `3`.
