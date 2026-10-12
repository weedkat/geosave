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
files = scene.gs.to_cog(root / "samples/forest_{time:%Y%m%d}.tif")
items = gs.stac.create_items(files, collection=collection)
path = items.gs.to_geoparquet(root / "catalog.parquet", collections=[collection])
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

`create_items` reads saved file headers to describe their actual grid, bands,
dtype, and nodata. It groups files with the same collection, instant, and grid
into one Item. The returned COG records describe files written by a successful
call; they are not an inventory of a destination folder.

## Search the local table

```python
client = gs.stac.StacClient.open(path)
found = client.search(gs.stac.StacQuery(collections=["forest"]))
found[["collection", "datetime"]]
```

Output:

```text
  collection                  datetime
0     forest 2025-01-01 00:00:00+00:00
```

Search returns another item table, suitable for normal GeoPandas filtering or
[window selection](catalogs-and-cuts.md). Search does not load raster pixels.
`client.collection("forest")` returns the native PySTAC `Collection`.

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
the selected bands onto its exact grid. `cube` is a lazy xarray `Dataset`.
It keeps the stored `uint16` measurements; decoding is explicit with
`cube.gs.mask_and_scale()`.

Choose resampling explicitly when changing grid resolution, especially for
categorical labels. `set_config(chunks=None)` requests eager loading;
`stac_cfg` supplies asset aliases or loader overrides when a provider's metadata
needs correction. See [raster grids](../concepts/raster-grids.md) and the
[STAC reference](../reference/stac.md).

## Search a remote API

This example requires network access. Available provider names are
`planetary_computer`, `cdse`, and `element84`; CDSE needs credentials.
The result depends on the live catalog, so no output is shown here.

<!-- NETWORK: public STAC API -->
```python
client = gs.stac.StacClient.open("planetary_computer")
source = client.source("sentinel-2-l2a").set_config(bands=["B04", "B08"])
source = source.set_query(filter="eo:cloud_cover <= 10")
found = source.search(anchor)
cube = source.load(anchor)
```

`search(anchor)` lets you review the matching rows before loading. The source's
query can also select dates, Item IDs, ordering, or a maximum number of Items.
Pixel access may need separate GDAL credentials or URL signing from the
catalog search itself; see [remote I/O](../basics/io.md#use-remote-storage).

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
client = gs.stac.StacClient.open("s3://samples/catalog.parquet", duckdb=session)
```

The same session searches rows and reads stored Collection metadata directly
from the table. This configures table access; remote TIFF reads still need
the [GDAL setup](../basics/io.md#use-remote-storage).

GeoSave uses PySTAC for STAC objects, stac-geoparquet for the table format,
rustac for search, and odc-stac for pixel loading. For a local training-data
catalog spanning several collections, continue with
[catalogs and windows](catalogs-and-cuts.md).
