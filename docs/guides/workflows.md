# Ingest data with optional flows

Ingestion is a Prefect flow in `geosave_flows`, installed with the `flows` extra:

```bash
pip install "geosave[flows]"
```

```bash
geosave-flows ingest \
  --anchor '{"kind":"coordinates","latitude":45,"longitude":12,"shape":1024,"resolution":10,"crs":"EPSG:32633","timespan":"2025-01"}' \
  --output data/raw/s1 \
  --spec model_spec.yaml
```

The optional `geosave_flows` package runs Prefect jobs that compose the GeoSave
library. Its `ingest` command loads the rasters declared in your model spec onto
one spatial and temporal target, saves them as published, and lists them in one
catalog.
It requires a `stac` acquisition recipe for **every** declared raster and access to the
endpoints and assets in those recipes. Create that spec first using
[Model spec](model-spec.md); a spec declaring only input variables has no acquisition recipes and cannot
be ingested this way.

The CLI currently converts `--output` and `--spec` through `pathlib.Path`,
which corrupts remote URL schemes. Use local paths with this command until
that parsing is fixed. The Python `geosave_flows.ingest` function accepts
unmodified path/URL strings; storage access still requires the corresponding
backend and credentials. See [the tracked issue](../contributing/review-2026-11-04.md#26-preserve-url-protocols-through-the-flows-cli).

If you already have raster or vector files, start with
[file ingestion](ingest-files.md). Opening those files does not require a
model spec or Prefect. This flow uses STAC recipes to acquire rasters that a
model declares.

| Input | Used for | Output |
| --- | --- | --- |
| Anchor configuration | Location, time and grid settings | Geographic GeoAnchor and native GeoBox opened by the flow. |
| Model spec YAML with STAC recipes | Collection search and required raster validation | One Dataset per declared raster. |
| New output folder | Computing and writing the loaded rasters | Path of the written `catalog.parquet`, as a string. |

The command runs a Prefect flow locally and waits for completion. Prefect may
start a temporary server; a Prefect account is not required. On success, the
final command output is the catalog path:

```text
data/raw/s1/catalog.parquet
```

This is the expected result for a successful run with the command above; the
external data download is separate from the offline training examples.

## Open the catalog

```python
import geosave as gs

catalog = gs.open_vector("data/raw/s1/catalog.parquet")
catalog.iloc[0].gs.to_datatree()
```

The result is an item table: one STAC Item per declared raster and instant,
each a COG under the output folder named after its raster. For example, a spec
declaring `sentinel_2_l2a` produces:

```text
data/raw/s1/
├── catalog.parquet
└── sentinel_2_l2a/
    └── 20250115T000000.tif
```

Each run writes one catalog. Join several runs with `gs.GeoCatalog.concat`.
Ingestion writes only the declared rasters. It does not run the spec's
preprocessing graph, tensor transforms, or training; `ModelSpec.prepare` takes
this catalog next. See [STAC](stac.md) and
[catalogs and cuts](catalogs-and-cuts.md).

## Choose the target grid

An **anchor** describes where and when data is needed. Three JSON kinds are
supported:

| Kind | Grid source | Time source |
| --- | --- | --- |
| `coordinates` | A WGS84 point, pixel shape, resolution, and optional target CRS. | Optional `timespan`. |
| `geojson` | The extent of a local GeoJSON file, with either shape or resolution. | Optional `timespan`. |
| `raster` | An existing raster's exact grid. | The raster's own temporal metadata. |

For raster labels, derive geographic coverage and the exact grid from the file:

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
from geosave_flows import ingest
from geosave_flows.configs import AnchorConfig

anchor = TypeAdapter(AnchorConfig).validate_python(
    {"kind": "raster", "path": "data/reference.tif"}
)
scene_path = ingest(
    anchor=anchor,
    output="data/raw/s1",
    spec="model_spec.yaml",
)
scene_path
```

The Python flow accepts a validated anchor configuration, rather than an
unvalidated dictionary. It opens the geographic anchor and pixel grid, loads the spec's rasters, and
returns the written catalog's path as a string. Validating the
configuration alone does not open files or contact STAC services.

| CLI option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--anchor JSON` | Yes | — | Spatial and temporal target. |
| `--output PATH` | Yes | — | New folder for the rasters and their catalog. |
| `--spec PATH` | Yes | — | Model spec YAML file. |

[`ModelSpec.load`](../reference/model-spec.md#geosave.model.spec.ModelSpec.load)
reads the supplied YAML file directly. The output folder must not already
exist.

See the [optional flows API](../reference/workflow.md) for flow and anchor
arguments.

## Load rasters directly in Python

For a Python application, the library exposes acquisition without a Prefect
flow. This uses only the `geosave` package and the same model spec with STAC
recipes:

```python
import geosave as gs
from geosave.model.spec import ModelSpec

from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 510240, 5010240),
    crs='EPSG:32633', resolution=10,
)
anchor = gs.GeoAnchor.from_geometry(grid.extent, timespan='2025-01')
spec = ModelSpec.load("model_spec.yaml")
rasters = spec.load_rasters(anchor, geobox=grid)
rasters
```

The result maps each declared raster name to a selected xarray Dataset. This
performs STAC search and opens the selected assets; pixel loading stays lazy
where the reader supports it. It needs access to those endpoints and assets.
Run `spec.preprocess(rasters)` separately when you want the declared raster
preprocessing; saving and training are separate calls too.

## Prepare an existing catalog for training

The Prefect package provides ingestion. Catalog preparation through
`ModelSpec.prepare` currently has a nested-tree compatibility gap; see
[the limitation and in-memory alternative](model-spec.md#catalog-preparation-limitation).
Sample an Item table with `items.gs.sample(...)` and cut opened raster trees
with `gs.Chips(...)`. ModelSpec has no `sample` method.
