# Search and load imagery with STAC

Create a searchable table from a saved image:

```python
import tempfile
from pathlib import Path

import numpy as np
import pystac

import xarray as xr

import geosave as gs

root = Path(tempfile.mkdtemp())
from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 500640, 5000640),
    crs='EPSG:32633', resolution=10,
)
anchor = gs.GeoAnchor.from_geometry(grid.extent, timespan='2025-01-01')
pixels = np.random.default_rng(0).integers(1, 3000, (64, 64), dtype="uint16")
scene = xr.Dataset({"red": (("y", "x"), pixels)}).gs.write_grid(grid).gs.write_nodata(0)
scene = scene.expand_dims(time=[np.datetime64("2025-01-01")])

# Any extent will do: to_catalog sets it to cover the Items it writes.
everywhere = pystac.Extent(
    pystac.SpatialExtent([[-180, -90, 180, 90]]), pystac.TemporalExtent([[None, None]])
)
collection = pystac.Collection("forest", "Forest samples", everywhere)
items = scene.gs.to_catalog(
    root / "samples", path_format="forest_{time:%Y%m%d}.tif",
    id_format="forest_{start:%Y%m%d}", collection=collection,
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

`to_catalog` writes the files and builds one Item per date from the raster
itself: its grid, dates, band attrs, and the STAC metadata it carries (see
[properties and Collections](#preserve-properties-and-collection-definitions)).
A tree uses its group names as collection ids. The result is a native
GeoDataFrame.

| Task | Input | Public call | Output |
| --- | --- | --- | --- |
| Catalog a raster | Dataset, DataArray or DataTree, folder and path format | `raster.gs.to_catalog(path, path_format=...)` | Item GeoDataFrame of the files written. |
| Search a table | GeoParquet path and query | `gs.StacClient.open(path).search(query)` | Matching Item GeoDataFrame. |
| Search a service | API endpoint and query | `gs.StacClient.open(endpoint).search(query)` | Matching Item GeoDataFrame. |
| Load a collection | Selected Items and output-grid settings | `io.load_dataset(items, geobox=grid)` | Dataset on the target grid; Dask-backed by default. |

Each date becomes one Item. With `{band}` in the template, the files of one
date are that Item's assets, each keyed by its variable, or `"image"` for a
file of several. Each Item's id is filled from `id_format` with its own time
and grid, by default `"{start:%Y%m%dT%H%M%S}_{lat:.4f}_{lon:.4f}"`. Undated
rasters take `datetime=` as a Python datetime or a `(start, end)` pair; without
one the Item spans all of time, with a warning. Several dates sharing a file
are refused before anything is written; add `{time}` to the template. Catalog
creation rejects duplicate Item ids within a collection.

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

## Items and PySTAC

`stac.from_items` turns PySTAC Items, such as a static catalog's, into
an item table; `items.gs.to_items()` goes the other way, one PySTAC `Item` per row:

```python
import pystac
from geosave.geodata import stac

root_catalog = pystac.Catalog.from_file("stac/catalog.json")
items = stac.from_items(root_catalog.get_items(recursive=True))
```

`gs.open_vector` reads vector files and Item GeoParquet tables; use PySTAC for
a static STAC JSON catalog.

## Search first, then load onto a chosen grid

```python
from geosave.geodata import io

query = gs.StacQuery(collections=["forest"]).set_anchor(anchor)
selected = client.search_items(query)
cube = io.load_dataset(
    selected, geobox=grid, bands=["red"],
    collection=client.collection("forest"), patch_url=client.patch_url,
)
```

GeoAnchor describes geometry and time only. `set_anchor` returns a new query,
replacing its spatial and temporal selectors while retaining other criteria.
A timeless anchor clears datetime; an existing bbox is cleared in favor of
WGS84 intersects. Select Items explicitly by ID by building a query with ids
and omitting set_anchor. No search is repeated by the pixel loader.

The grid is a separate choice. Match an existing raster with
`geobox=reference.gs.geobox`, or ask ODC to construct one:

```python
cube = io.load_dataset(
    selected, geopolygon=anchor.geometry, crs="EPSG:32633", resolution=10,
    bands=["red"], collection=client.collection("forest"),
)
```

Polygon bounds are snapped to a rectangular pixel grid; pixels outside the
polygon are not masked. A point alone does not describe a raster window:
choose an area or construct an explicit GeoBox first. The `anchor` load option
is ODC's pixel-alignment setting, distinct from GeoSave's GeoAnchor.

Load options use `StacLoadOptions` and keyword completion. The defaults are
`groupby="solar_day"` and spatial Dask chunks of 1024. Pass `chunks=None` for
eager loading. Native ODC settings such as `stac_cfg`, per-band dtype and
resampling remain available. Radiometric decoding stays explicit through
`cube.gs.mask_and_scale()`.

GeoSave retains CF axes, loader-authoritative nodata, agreed band metadata,
Collection/common attrs and requested per-time properties. An empty selection
raises AnchorFetchError. Conflicting source metadata is omitted with a warning.
Collection metadata is supplied explicitly and never fetched by a loader.

```python
tree = io.load_datatree(selected, geobox=grid, collection=client.collection("forest"))
```

A tree has collection parents and grid leaves containing variables. The finest
selected band family uses the requested grid; other families use scaled grids
according to source resolution ratios. This preserves relative resolution,
not necessarily absolute native resolution or source pixel alignment. Families
are currently inferred from the first Item. Native `row.gs.to_datatree()` reads
saved assets on their original grids without choosing a new output grid.

## Search a remote API

This example requires network access. Provider names are `planetary_computer`,
`cdse`, and `element84`. Catalog and asset credentials are separate.

<!-- NETWORK: public STAC API -->
```python
client = gs.StacClient.open("planetary_computer")
query = gs.StacQuery(collections=["sentinel-2-l2a"])
query = query.set_anchor(anchor).set_filter("eo:cloud_cover <= 10")
selected = client.search_items(query)
cube = io.load_dataset(
    selected, geobox=grid, bands=["B04", "B08"],
    collection=client.collection("sentinel-2-l2a"), patch_url=client.patch_url,
)
```

`search_items` returns native PySTAC Items; `search` returns a GeoDataFrame.
Pass `client.patch_url` to sign assets at loading time without changing durable
hrefs on the selected Items. Native PySTAC Client also works:

```python
from pystac_client import Client

api = Client.open("https://earth-search.aws.element84.com/v1")
selected = list(api.search(**query.to_search_params()).items())
cube = io.load_dataset(selected, geobox=grid, bands=["red", "nir"])
```

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
planetary = gs.StacClient.open("planetary_computer")
query = gs.StacQuery(collections=["sentinel-2-l2a"]).set_anchor(anchor)
selected = planetary.search_items(query)
raster = io.load_dataset(
    selected, geobox=grid, bands=["B04", "B08"],
    with_properties=["eo:cloud_cover"],
    collection=planetary.collection("sentinel-2-l2a"), patch_url=planetary.patch_url,
)
raster.gs.stac_properties              # one row per date: the requested properties
clear = raster.sel(time=raster.gs.stac_properties["eo:cloud_cover"] < 30)
catalog = clear.gs.to_catalog(root / "prepared", path_format="{time:%Y%m%d}.tif")
catalog.gs.to_geoparquet(root / "prepared.parquet")
```

A loaded raster carries its STAC metadata, so `to_catalog` needs nothing more:

| Metadata | Lives on | Read with |
| --- | --- | --- |
| Collection | root `StacMetadata` attrs | `raster.gs.stac_collection` |
| Properties every Item shares | root `StacMetadata` attrs | `raster.gs.stac_common_properties` |
| Properties requested with `with_properties` | `stac_properties` coordinate, one JSON mapping per date | `raster.gs.stac_properties` |

Each Item's properties are the shared ones overlaid with its date's mapping.
No other coordinate becomes a property, and dates, `proj:*` and band fields
always come from the raster's own time, grid and attrs. A date odc-stac builds
from several Items takes its fused value of each requested property; an Item
lacking a key leaves it out. Cropping does not recalculate source cloud cover:
recompute it or drop it from `stac_properties` when it no longer describes the
output. A derived product, such as a composite, names its own collection with
`to_catalog(..., collection=...)`.

Reopening a catalog row with `row.gs.to_datatree()` puts its properties back on
`stac_properties` and its Collection on `StacMetadata`, so catalog, xarray and
catalog again agree. Composite nominal time and explicit resampling bounds are
both published when those bounds are present on the raster. `io.load_dataset`
keeps ODC's time representation and does not reconstruct composite bounds.
Catalog assets are COGs; a Zarr or netCDF store is written with `to_zarr` or `to_netcdf`, not catalogued.

`catalog.gs.collections` returns native Collection copies. Definitions live once
in the table's attrs and are restored when reading STAC GeoParquet. Filtering
retains them; writers emit only referenced definitions and update extents on
copies. Missing referenced definitions are errors. One Collection per published
Parquet is a useful convention, while multiple Collections remain supported.

Combine catalogs before joining them into samples:

```python
combined = gs.GeoCatalog.concat([optical_catalog, label_catalog])
samples = combined.gs.sample()
rasters = samples[0].gs.to_datatree()
```

`GeoCatalog.concat` merges definitions by id and keeps the first definition. Plain
`pandas.concat` can discard differing attrs, so use the catalog operation when
Collection metadata must survive. A stack row may reference many dates; it is a
sampling record, not an Item. `client.collection(collection_id)` returns a native PySTAC Collection.

An Item without Projection metadata can still be read with
`rows.iloc[0].gs.to_datatree()`: the file reader uses georeferencing stored in the
file. An image without one takes the grid its asset states in `proj:*` when the
stated shape matches; otherwise it has no grid node and is refused, naming the
asset. `open_raster(path)` still opens it with pixel coordinates and no CRS.
No geographic location is invented.
Metadata-only `row.gs.anchor` reads Item geometry and time even without projection
metadata. `rows.gs.anchor` unions geometry in the frame CRS and envelopes dated
members; timeless members do not widen the interval. Standalone STAC rows must
remain in WGS84. Select projected rows as a one-row GeoDataFrame to retain CRS.
Chipping requires an explicit grid.
ODC loading is a geospatial operation and is not an ordinary-image fallback.
