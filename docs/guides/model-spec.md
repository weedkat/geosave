# ModelSpec recipes

```python
from geosave.model.spec import ModelSpec

spec = ModelSpec.load("model_spec.yaml")
prepared = spec.run(entry="checked", output="reflectance", raster=raster)
image = spec.run(raster=raster)  # Saved entry/output defaults
```

ModelSpec version 1 connects installed Python functions using their arguments
and results. Download the [complete recipe](model_spec.yaml). Supply a native
Dataset whose variables are already selected in model order and have `(y, x)`
dimensions. The recipe validates its grid, decodes stored pixels, and explicitly
materializes a tensor only when `encoded` is requested.

```yaml
schema_version: 1
entry: checked
output: encoded
inputs:
  raster: {}
  destination: {}

steps:
  checked:
    call: geosave.model.processing.validate_raster
    kwargs: {raster: !ref raster}

  reflectance:
    call: !ref entry.gs.mask_and_scale

  encoded:
    call: geosave.model.processing.to_tensor
    kwargs: {raster: !ref reflectance, dims: [band, y, x]}

  saved:
    call: geosave.geodata.io.formats.geotiff.write_cog
    kwargs: {ds: !ref reflectance, path: !ref destination}
```

The root fields are `schema_version`, `inputs`, `steps`, `entry`, and `output`.
Named operations belong inside `steps`, each with `call` and optional `kwargs`.
There are no lifecycle stages or named flows. Python functions implement loops, branches, model inference,
acquisition, and resource cleanup.

## Inputs and references

An empty input declaration requires a caller-supplied native value. `default`
supplies a value copied for every invocation; `default: null` explicitly supplies
None. Consuming operations validate input values; ModelSpec does not declare or
convert their runtime types.
Input and step names must be distinct Python identifiers. `entry` and `output`
are reserved selectors and cannot be input names. Use `destination` for a file
location. Only `entry` is reserved as a step name, because `!ref entry` selects
the active entry. Other document field names can also be step names inside `steps`.

`!ref raster` passes the external input. `!ref reflectance` passes a call result.
`!ref entry` passes the selected entry step's result. Dotted references use
native attribute lookup: `!ref raster.gs.geobox` or `call: !ref model.predict`.
Mapping keys, indexing, expressions, and calls are not reference syntax. Pass
whole mappings to Python functions when processing requires their contents.

References inside nested argument lists and dictionaries work the same way.
YAML mapping keys must be unique strings; aliases are not supported.

## Planning and execution

```python
spec.plan()  # entry, output, ordered steps, and required input names
outputs = spec.run(
    entry="checked", output=["reflectance", "encoded"], raster=raster
)
spec.save("memory://recipes/model_spec.yaml")
```

Planning follows dependencies from the requested results and detects cycles.
It does not import configured code or read object properties. Declaration
order does not establish side-effect order. Every required call runs once per
invocation; multiple requested results share their dependencies. Separate runs
recompute. The selected entry must participate in the output's dependencies;
an unrelated entry is rejected before operations are imported. The entry may
have prerequisites, which also execute. It need not be the first step.

Every run resolves an entry and output from its arguments or saved defaults.
Defaults are optional in the document, but missing selections must then be
provided by the caller. An output can be a step name or a nonempty list.
Overrides apply only to that run and do not change the saved defaults.

Only inputs needed on that path are required. `destination` is needed only when
requesting `saved`. Unknown and unused arguments fail before execution.
Intermediate results are selectable but cannot be injected as overrides.
Loading and saving use GeoSave locations and optional `storage_options`.

Each step loads or reads its operation, binds its arguments, and calls it in one
execution pass. Python reports incompatible call arguments. ModelSpec does not
inspect operation signatures in advance: earlier steps may already have executed
when a later operation fails to load or rejects its arguments. Missing external
inputs still fail before any operation is imported or called.

Alternative entry operations must satisfy their shared consumer's contract:
bands, units, grid, time, and missing-value policy. `validate_raster` checks Dataset geometry and optional dimensions, sizes, and
units without reading pixels. The recipe opts into those checks; the executor
does not automatically validate convergence.

## Bulk processing operations

```python
from geosave.model.ingestion import regions_from_features
from geosave.model.spec import ModelSpec

regions = regions_from_features(features, timespan="2025-06")
spec = ModelSpec.load("model_spec_bulk.yaml")
batch = spec.run(
    regions=regions,
    catalog=catalog,
    collection="sentinel-2-l2a",
    crs="EPSG:32748",
    resolution=10,
    bands=["B02", "B03", "B04", "B08"],
    mean=[0.1105, 0.1355, 0.1552, 0.2743],
    std=[0.1809, 0.1757, 0.1888, 0.1742],
)
```

Download the [bulk recipe](model_spec_bulk.yaml). It ingests regions, encodes
rasters, stacks compatible tensors, and normalizes the batch. Its output axes
are `[sample, band, time, y, x]`. Regions must produce compatible spatial and
temporal shapes to stack; use the `images` output to obtain separate tensors
when shapes differ. It does not select a single observation or compose dates
into fixed temporal slots.

A reference passes its **whole value to one call**. A list does not trigger
mapping. In this recipe, `regions` is a collection of requests, `bands` is a
shared variable order, and `mean` is a vector of statistics. Their meanings
belong to the consuming functions.

| Operation | Callable | Contract |
| --- | --- | --- |
| Construct regions | `geosave.model.ingestion.regions_from_features` | One GeoRegion per feature, preserving row order and CRS, with a common optional timespan. |
| Ingest imagery | `geosave.model.ingestion.ingest_stac` | Regions → tuple of Datasets; owns per-region search and grid construction and fetches Collection metadata once. |
| Encode rasters | `geosave.model.processing.encode_rasters` | Selects ordered bands, decodes nodata/packing, transposes, and jointly materializes lazy arrays → tuple of CPU tensors. |
| Form a batch | `torch.stack` | Compatible tensors → one tensor; no xarray coordinate alignment across locations. |
| Normalize | `geosave.model.processing.normalize` | Applies fixed statistics along the declared axis, preserving gradients and missing values. |
| Restore predictions | `geosave.model.processing.restore_rasters` | Predictions + ordered reference rasters → tuple of DataArrays on their respective grids. Rejects unequal counts and grid-shape mismatches. |

`encode_rasters` checks all selected variables' dimensions before pixel reads.
It never broadcasts static variables over time. Lazy pixel arrays participate
in one Dask computation, allowing shared tasks to run once. The entire supplied
collection is materialized: choose a batch size that fits memory. This does not
imply that STAC requests execute concurrently or that separately constructed
read graphs always deduplicate their I/O.

Raster collections preserve input order. Empty collections produce empty tuples;
stacking requires at least one tensor. Ingestion raises when a requested region
has no imagery rather than silently removing it. Restore predictions in that
same order; equal counts cannot detect an accidental permutation by a model.

### Regions and grids

`regions_from_features` keeps features separate, including rows with duplicate
index labels. `GeoRegion.from_geometry(frame)` remains the explicit union route.
For different dates per feature, construct a sequence of GeoRegions directly.
GeoRegion factories for bounding boxes, geometries, and coordinates remain usable
from Python; ingestion accepts any sequence of these regions.

Polygon ingestion covers the geometry's bounding grid at the declared CRS and
resolution. It does not mask pixels outside the polygon. Point ingestion needs
`shape=(height, width)` in addition to resolution and CRS. The pixel window is
centered exactly on each point; resolution uses CRS units, so use a projected
CRS when specifying metres. Do not supply shape for polygon regions.

Ingestion searches the full output window, including its date interval, so tiles
covering its edges can contribute. It loads one collection using the existing
ODC loader, retaining native metadata and lazy pixels by default. `groupby`
defaults to `solar_day`; this mosaics observations within a day, not across the
whole requested timespan. Provider URL signing comes from the catalog unless
explicitly overridden. Native load options such as `bands`, `chunks`, `resampling`,
and `stac_cfg` pass through to the loader.

Custom Item selection and exact reference-grid loading remain available through
`StacQuery`, catalog search, and `geosave.geodata.stac.load_dataset`. The bulk
operation does not add implicit cloud ranking, temporal compositing, or a file
opening mode.

### Individual operations and resource ownership

For unusual preparation, recipes can call native `sel`, `isel`, `fillna`,
`gs.mask_and_scale`, and `gs.reproject` methods directly. Scalar operations
`select_variables`, `validate_raster`, `to_tensor`, and `to_raster` also remain
available. `validate_raster` checks Dataset grid structure plus optional exact
`dims`, dimension `sizes`, and literal per-variable `units`; it returns the input
unchanged. These checks are optional operations, not executor-level validation.

Normalization statistics must describe decoded physical values in the selected
band order. Missing-value replacement and model-specific temporal/location
encoding are explicit decisions. The bulk encoder does not infer either.

For file-backed inputs, the caller keeps resources open through consumption:

```python
from geosave.geodata.io import open_raster
from geosave.model.processing import encode_rasters

with open_raster("scene.tif") as raster:
    images = encode_rasters([raster], bands=["red", "nir"], dims=["band", "y", "x"])
```

A grouped store returns a DataTree: select its intended Dataset first. ModelSpec
does not close arbitrary operation results on success or failure. If an output
is lazy, consume or persist it while its input resources remain open.

## Training and deployment

Generated geospatial recipes name `prepared`, `encoded`, and `normalized` steps.
Training normalizes tensors after augmentation using entry `from_tensor`.
Inference requests `normalized` with entry `encoded`. These names belong to
the project recipe, not the ModelSpec class. DataModules declare their own
`pixels` tuple for synchronized augmentation.

A release can save the recipe beside weights. It does not package custom Python
functions: install the project code on the machine executing the recipe.
Use an explicit, typed project function to call `run` inside Prefect; see
[workflows](workflows.md). Nodes are not automatically Prefect tasks.

## Execution boundaries

Executing a recipe is trusted Python execution. Imports and attribute properties
can run code; safe YAML loading is not a sandbox. No remote code is installed
implicitly. References preserve native values and lazy arrays, but called
functions may compute or mutate them. Results remain alive until the run ends.
Functions own open resources; runtime failures do not roll back side effects.

## Construction and errors

`ModelSpec.from_yaml(text)` parses a YAML string. `ModelSpec.from_dict(document)`
reads the saved document structure. For Python-owned declarations, construct
`ModelSpec(steps=..., inputs=..., entry=..., output=...)`; references are
`Reference("name")` values imported from `geosave.model.spec`.

Input and Step are Pydantic models. Their fields describe individual declarations;
misspelled fields and invalid declaration types fail during loading. ModelSpec is
a plain class that owns connections, planning, execution, and YAML persistence.
Its constructor accepts validated Input and Step objects; use from_dict() or
from_yaml() for raw saved declarations. Python callers can construct typed
declarations and inspect their public fields:

```python
from geosave.model.spec import Input, ModelSpec, Reference, Step

spec = ModelSpec(
    inputs={"value": Input()},
    steps={
        "returned": Step(call="builtins.dict", kwargs={"value": Reference("value")}),
    },
    entry="returned",
    output="returned",
)
result = spec.run(value=raster)
```

Pydantic validates the declarations, not the raster or tensor data flowing through
them. `Reference.read_value()` returns the native object named by a reference.
`find_references()` discovers dependencies; `bind_arguments()` returns the keyword
argument dictionary after replacing references with their values. A single
workflow output has the concrete type returned by its operation, so its static
annotation is `object`. An explicit output list returns `dict[str, object]`.
`object` requires callers to establish the concrete type before using its methods;
it does not disable type checking as `Any` does.

Loading and saving use the same workflow declarations; there is no separate
retained source document. Error messages identify the requested output, failing
step, and relevant operation or argument. The original Python exception is
preserved, with additional context in its notes. Missing inputs explain which
step uses them and how to supply them.

This unreleased version 1 replaces `source`/`result` with `entry`/`output`,
`!ref source` with `!ref entry`, and `ModelSpec(text)` with
`ModelSpec.from_yaml(text)`. Steps are now nested under `steps`; input `type`
fields are removed. No compatibility methods or automatic migration are provided.
