# Ingest model rasters

```bash
geosave workflow ingest \
  --anchor '{"kind":"coordinates","latitude":45,"longitude":12,"shape":1024,"resolution":10,"crs":"EPSG:32633","timespan":"2025-01"}' \
  --output data/scenes/s1 \
  --spec model_spec.yaml
```

This loads the rasters declared in your model spec onto one spatial and
temporal target, then writes them into a new sample directory. It requires a
`stac` acquisition recipe for **every** declared raster and access to the
endpoints and assets in those recipes. Create that spec first using
[Model spec](model-spec.md); the small tensor-only
training example has no acquisition recipes and cannot be ingested this way.

The command runs a Prefect flow locally and waits for completion. Prefect may
start a temporary server; a Prefect account is not required. On success, the
final command output is the directory path:

```text
data/scenes/s1
```

This is the expected result for a successful run with the command above; the
external data download is separate from the offline training examples.

## Open the saved sample

```python
from geosave_engine import geodata as gs

scene = gs.open_stack("data/scenes/s1")
scene
```

The result is a native xarray DataTree, with one raster group per name in
`ModelSpec.rasters`. For example, a spec declaring `sentinel_2_l2a` produces:

```text
data/scenes/s1/
└── sentinel_2_l2a.zarr/
```

The default Zarr format retains a time series. Use `--format geotiff` for
rasters with no time dimension or exactly one instant; the writer refuses a
multi-instant raster in that format. Reading the directory opens its rasters;
it does not create training splits or a torch Dataset.

Ingestion writes only the declared rasters. It does not run the spec's
preprocessing graph, tensor transforms, or training. Add saved rasters to a
catalog when you need to organize many samples; see [STAC](stac.md) and
[catalogs and cuts](catalogs-and-cuts.md).

## Choose the target grid

An **anchor** describes where and when data is needed. Three JSON kinds are
supported:

| Kind | Grid source | Time source |
| --- | --- | --- |
| `coordinates` | A WGS84 point, pixel shape, resolution, and optional target CRS. | Optional `timespan`. |
| `geojson` | The extent of a local GeoJSON file, with either shape or resolution. | Optional `timespan`. |
| `raster` | An existing raster's exact grid. | The raster's own temporal metadata. |

For raster labels, use their grid as the anchor:

```json
{"kind": "raster", "path": "data/reference.tif"}
```

For a GeoJSON area, provide exactly one of `shape` or `resolution`. Optional
`pad` expands the area before grid construction:

```json
{"kind": "geojson", "path": "data/area.geojson", "resolution": 10,
 "crs": "EPSG:32633", "timespan": "2025-01", "pad": 100}
```

A coordinate anchor requires positive `shape` and `resolution`. The shape can
be one integer for a square or `[height, width]`. Select an appropriate CRS
and temporal range explicitly for your task.

## Call ingestion from Python

```python
from pydantic import TypeAdapter
from geosave_engine.workflow.configs import AnchorConfig
from geosave_engine.workflow.flows import ingest

anchor = TypeAdapter(AnchorConfig).validate_python(
    {"kind": "raster", "path": "data/reference.tif"}
)
scene_path = ingest(
    anchor=anchor,
    output="data/scenes/s1",
    spec="model_spec.yaml",
)
scene_path
```

The Python flow accepts a validated anchor configuration, rather than an
unvalidated dictionary. It opens that anchor, loads the spec's rasters, and
returns the completed sample directory as a string. Validating the
configuration alone does not open files or contact STAC services.

| CLI option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--anchor JSON` | Yes | — | Spatial and temporal target. |
| `--output PATH` | Yes | — | New local sample directory. |
| `--spec PATH` | Yes | — | Model spec YAML file. |
| `--format [geotiff\|zarr]` | No | `zarr` | Storage for each raster. |
| `--write-options JSON` | No | `{}` | Encoding options passed to the native writer. |

[`ModelSpec.load`](../reference/model-spec.md#geosave_engine.model.spec.ModelSpec.load)
reads the supplied YAML file directly. The output directory must not already exist. Files are written into a staging
directory and the completed sample is then published as the destination.
The writer also requires co-registered rasters. It owns publication settings,
so `compute`, `layout`, and `overwrite` are not accepted in `write-options`.

## Prepare an existing catalog for training

```python
from geosave_engine import geodata as gs
from geosave_engine.model.spec import ModelSpec

raw = gs.open_vector("data/raw/catalog.parquet")
spec = ModelSpec.load("model_spec.yaml")
prepared = spec.prepare(raw, "data/prepared", on="label")
prepared
```

For a compatible raw Item table and spec, this returns
`data/prepared/catalog.parquet`. `on="label"` selects the collection whose
items define the preparation windows. The method runs model-owned raster
selection and preprocessing, writes the input rasters and retained labels as
COGs, and saves the new catalog and model spec. Tensor transforms still run
later in the DataModule. This operation reads and writes pixels; the output
folder must be new. See [Model spec](model-spec.md)
for a complete catalog example.

The Prefect `prepare_dense_data` flow and its optional workspace scaffold are
unfinished: calling the flow raises `NotImplementedError`. It is not a
registered CLI command. The supported workflow CLI currently exposes
`ingest`; use `ModelSpec.prepare` directly for the catalog operation above.
