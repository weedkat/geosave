# Search and load imagery with STAC

Create a searchable table from a saved image:

```python
import tempfile
from pathlib import Path

import numpy as np

import geosave as gs

root = Path(tempfile.mkdtemp())
anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=64, resolution=10,
    crs="EPSG:32633", timespan="2025-01-01",
)
pixels = np.random.default_rng(0).integers(1, 3000, (64, 64), dtype="uint16")
scene = gs.raster({"red": (("y", "x"), pixels)}, anchor.geobox, nodata=0)
scene = scene.expand_dims(time=[np.datetime64("2025-01-01")])

collection = gs.stac.create_collection("forest", description="Forest samples")
scenes = scene.gs.to_cog(root / "samples/forest_{time:%Y%m%d}.tif")
items = gs.stac.create_catalog(
    gs.stac.create_items(scenes, id_format="forest_{start:%Y%m%d}", collection=collection),
    collections=[collection],
)
path = items.gs.to_geoparquet(root / "catalog.parquet")
items[["collection", "datetime"]]
```

Output:

```text
  collection                  datetime
0     forest 2025-01-01 00:00:00+00:00
```

STAC is metadata for discovering geospatial data. An **Item** describes one
observation: its footprint, date, properties, and files. Each file is an
**Asset**, addressed by an `href`. A **Collection** describes a named dataset
whose Items belong together.

Here `items` is a GeoPandas `GeoDataFrame`, with one Item per row. Properties
are table columns; `assets` contains the nested file descriptions. The
GeoParquet file stores this metadata, while the pixels remain in the TIFF.

`to_cog` returns one `Scene` per date: the files of one collection at one
instant, keyed by asset. A scene is one STAC Item, so one row of the table.
`create_items` reads each scene's file headers to state their grid, bands,
dtype and nodata, and yields one `pystac.Item` per scene; `create_catalog`
lays them out as the table, refusing two Items of one collection that share an
id. A tree's scenes name their layer, so they fall into one collection per layer
without `collection=`.

| Task | Input | Public call | Output |
| --- | --- | --- | --- |
| Describe saved rasters | Scenes from `to_cog`, optional id format, collection, date | `gs.stac.create_catalog(gs.stac.create_items(scenes, ...))` | Item GeoDataFrame; reads headers, not pixels. |
| Search a table | GeoParquet path and query | `gs.StacClient.open(path).search(query)` | Matching Item GeoDataFrame. |
| Search a service | API endpoint and query | `gs.StacClient.open(endpoint).search(query)` | Matching Item GeoDataFrame. |
| Load a collection | Configured source and GeoAnchor | `source.load(anchor)` | Dataset on the target grid; Dask-backed by default. |

A scene's headers are not read again. Files you saved yourself become a scene
with your own asset keys: `Scene.open({"red": "red.tif", "nir": "nir.tif"})`, importing `Scene` from
`geosave.geodata.io.raster.file`.
`to_cog` keys a file by the band it holds, or `"image"` where it holds several.
Each Item's id is filled from `id_format` with its own time and grid, by default
`"{start:%Y%m%dT%H%M%S}_{lat:.4f}_{lon:.4f}"`. Undated files take `datetime=` as
a Python datetime or a `(start, end)` pair; without one the Item spans all of
time and `create_items` warns.
A file with several dates is refused; split the time series into TIFFs first.

## Search the local table

```python
client = gs.StacClient.open(path)
found = client.search(gs.StacQuery(collections=["forest"]))
found[["collection", "datetime"]]
```

Output:

```text
  collection                  datetime
0     forest 2025-01-01 00:00:00+00:00
```

Search returns another item table, suitable for normal GeoPandas filtering or
[window selection](catalogs-and-cuts.md). Search carries the returned Items' Collection definitions, so
`found.gs.to_geoparquet("subset.parquet")` preserves them. This may fetch Collection
metadata from the service, but does not load raster pixels. Empty results fetch
no Collection definitions. `client.collection("forest")` returns the native PySTAC `Collection`.

## Share as STAC JSON

Tools that read static STAC, such as STAC Browser or QGIS, expect a folder of
JSON files rather than one table. `to_stac_json` writes one:

```python
catalog = items.gs.to_stac_json(root / "stac", collections=[collection])
```

```text
stac/catalog.json
stac/forest/collection.json
stac/forest/forest_20250101/forest_20250101.json
```

Asset hrefs are relative to each Item JSON file. This writes metadata and
keeps the asset files at their existing locations; move both together while
preserving those relative paths. A URL writes
the folder to that filesystem, as
`items.gs.to_stac_json("s3://bucket/stac", storage_options={"anon": False})`. PySTAC reads it
back, and `create_catalog` turns its Items into an item table again:

```python
import pystac

root_catalog = pystac.Catalog.from_file(catalog)
items = gs.stac.create_catalog(root_catalog.get_items(recursive=True))
```

`items.gs.to_items()` goes the other way, one PySTAC `Item` per row.
`gs.open_vector` reads vector files and Item GeoParquet tables; use PySTAC for
a static STAC JSON catalog.

## Load onto a chosen grid

```python
source = client.source("forest").set_config(bands=["red"])
cube = source.load(anchor)
cube
```

Output:

```text
<xarray.Dataset> Size: 9kB
Dimensions:      (time: 1, y: 64, x: 64)
Coordinates:
  * time         (time) datetime64[ns] 8B 2025-01-01
  * y            (y) float64 512B 4.988e+06 4.988e+06 ... 4.987e+06 4.987e+06
  * x            (x) float64 512B 2.632e+05 2.632e+05 ... 2.639e+05 2.639e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (time, y, x) uint16 8kB dask.array<chunksize=(1, 64, 64), meta=np.ndarray>
Attributes:
    id:       forest
    summary:  Forest samples
    license:  other
```

The source searches within the anchor's footprint and time range, then loads
the selected bands onto its exact grid. The Collection lands on the root as
ACDD attrs, read back with `cube.gs.acdd`; each variable carries the packing,
units, spectral facts and classes its STAC band states. `cube` is a lazy xarray `Dataset`.
It keeps the stored `uint16` measurements; decoding is explicit with
`cube.gs.mask_and_scale()`.

An anchor provides missing spatial and temporal search selectors. Explicit
`bbox` or `intersects` and `datetime` in `set_query` take precedence; explicit
Item `ids` bypass anchor narrowing. An empty search returns an empty table;
loading it raises `AnchorFetchError`. A one-Item query retains ODC's `time`
dimension; call `.squeeze("time")` explicitly for a two-dimensional scene.

Variables use STAC asset keys, such as `B04`, or loader-configured aliases. The
bands of a multiband asset take the names its `eo:bands` state, as odc-stac
names them, unless another asset already holds that name; a band without a
free name is `image.1`, `image.2`. Inspect the Item's assets or configure
aliases before requesting bands from an unfamiliar collection.

Native row reading with `row.gs.to_xarray()` uses its own
stable naming rules, independent of ODC configuration: a single-band asset takes
its key, while multiple bands use their published names or common names when
unclaimed, falling back to `key.1`, `key.2`. Only eligible data assets participate
in naming and sample-grid selection. Assets with no stated roles remain eligible;
explicit thumbnail or metadata roles without `data` are excluded. The first
eligible asset stating a grid supplies the sample grid; multiple data resolutions
remain separate spatial samples unless their footprints are equal.

When contributing Items disagree on band metadata, source loading warns and
omits the conflicting source attributes. Loader metadata remains authoritative;
published digital numbers are not decoded automatically.

Choose resampling explicitly when changing grid resolution, especially for
categorical labels. `set_config(chunks=None)` requests eager loading;
`stac_cfg` supplies asset aliases or loader overrides when a provider's metadata
needs correction. See [raster grids](../concepts/raster-grids.md) and the
[STAC reference](../reference/stac.md).

`set_config` updates only the settings passed to it. `set_query` replaces the
source's query. Both mutate and return the same source, so create a fresh
`client.source(...)` for independently configured loads. The default
`groupby="solar_day"` groups Items by solar day; choose `"time"` for timestamps
or `"id"` for one group per Item when those are the model's intended units.

## Search a remote API

This example requires network access. Available provider names are
`planetary_computer`, `cdse`, and `element84`. Access to an API's catalog and
access to its image assets can require different credentials.
The result depends on the live catalog, so no output is shown here.

<!-- NETWORK: public STAC API -->
```python
client = gs.StacClient.open("planetary_computer")
source = client.source("sentinel-2-l2a").set_config(bands=["B04", "B08"])
source = source.set_query(filter="eo:cloud_cover <= 10")
found = source.search(anchor)
cube = source.load(anchor)
```

`search(anchor)` lets you review the matching rows before loading. The source's
query can also select dates, Item IDs, ordering, or a maximum number of Items.
`load(anchor)` runs the search again; pass the reviewed IDs to `set_query(ids=...)`
when you need to load that exact selection.
Pixel access may need separate GDAL credentials or URL signing from the
catalog search itself; see [remote I/O](../concepts/storage.md#use-remote-storage).

## Search a remote item table

For an S3-compatible table, configure a native rustac DuckDB session.
This example requires your endpoint and keys:

<!-- CREDENTIALS: S3-compatible endpoint and access keys -->
```python
import rustac

session = rustac.DuckdbClient()
session.execute("""
    CREATE SECRET (
        TYPE s3, KEY_ID 'your-access-key', SECRET 'your-secret-key',
        ENDPOINT 'localhost:9000', URL_STYLE 'path', USE_SSL false
    )
""")
client = gs.StacClient.open("s3://samples/catalog.parquet", duckdb=session)
```

The same session searches rows and reads stored Collection metadata directly
from the table. This configures table access; remote TIFF reads still need
the [GDAL setup](../concepts/storage.md#use-remote-storage).

GeoSave uses PySTAC for STAC objects, stac-geoparquet for the table format,
rustac for search, and odc-stac for pixel loading. For a local training-data
catalog spanning several collections, continue with
[catalogs and windows](catalogs-and-cuts.md).

## Preserve properties and Collection definitions

```python
source = client.source("forest").set_config(with_properties=["eo:cloud_cover"])
catalog = source.search(anchor)  # includes source.collection as a native definition
catalog.gs.to_geoparquet(root / "source.parquet")

raster = source.load(anchor)
raster["eo:cloud_cover"]  # ODC property variable, indexed by time
raster = raster.set_coords("eo:cloud_cover")  # explicitly publish it as metadata
scenes = raster.gs.to_cog(root / "prepared/{time:%Y%m%d}.tif")
prepared = gs.stac.create_collection("prepared", description="Prepared observations")
catalog = gs.stac.create_catalog(
    gs.stac.create_items(scenes, collection=prepared), collections=[prepared],
)
catalog.gs.to_geoparquet(root / "prepared.parquet")
```

Recognized scalar coordinates export automatically: `eo:cloud_cover`,
`eo:snow_cover`, `platform`, `constellation` and `gsd`. Missing values are omitted;
invalid or unresolved multivalue properties are refused. Other coordinates stay
available for processing. Names are never renamed. Cropping does not recalculate
source cloud cover: recompute it or drop that coordinate when it is inappropriate
for the output. ODC's configured property fuser determines values when Items group.

A `Scene` preserves these coordinates separately from its files' actual headers,
including deferred writes. Reopening a TIFF cannot recover coordinates the format
did not store. Composite nominal time and explicit resampling bounds are both
published when those bounds are explicitly present on the raster. `StacSource`
keeps ODC's time representation and does not reconstruct composite bounds.
Retain the source Items if those intervals are needed for later publication;
requesting properties alone does not establish their aggregation semantics.
A multidate store must be split into observations before publication.

`catalog.gs.collections` returns native Collection copies. Definitions live once
in the table's attrs and are restored when reading STAC GeoParquet. Filtering
retains them; writers emit only referenced definitions and update extents on
copies. Missing referenced definitions are errors. One Collection per published
Parquet is a useful convention, while multiple Collections remain supported.

Combine catalogs before joining them into samples:

```python
combined = gs.GeoCatalog.concat([optical_catalog, label_catalog])
samples = combined.gs.sample()
rasters = samples.iloc[0].gs.to_xarray()
```

`GeoCatalog.concat` merges definitions by id and keeps the first definition. Plain
`pandas.concat` can discard differing attrs, so use the catalog operation when
Collection metadata must survive. A stack row may reference many dates; it is a
sampling record, not an Item. `source.collection` remains a native PySTAC Collection.

An Item without Projection metadata can still be read with
`rows.iloc[0].gs.to_xarray()`: the file reader uses georeferencing stored in the
file. An unreferenced image remains a lazy Dataset with pixel coordinates and no
CRS, accompanied by the reader's warning. No geographic location is invented.
Metadata-only `row.gs.anchor` returns `None` with `MissingGridWarning` when the
Item states no grid. Spatial sampling needs only its footprint; chipping requires an explicit grid.
ODC loading is a geospatial operation and is not an ordinary-image fallback.
