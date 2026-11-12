# STAC catalogs, search, and loading

For a first download, follow [Download satellite imagery](download-imagery.md).
This guide covers local catalog creation, advanced search, loading grids, and
metadata preservation.

## Create a local catalog

Create a searchable table from a saved image:

```python
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pystac

import xarray as xr
from odc.geo.xr import xr_coords

import geosave as gs

root = Path(tempfile.mkdtemp())
from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 500640, 5000640),
    crs='EPSG:32633', resolution=10,
)
region = gs.GeoRegion.from_geometry(grid.extent, timespan='2025-01-01')
pixels = np.random.default_rng(0).integers(1, 3000, (64, 64), dtype="uint16")
scene = xr.Dataset({"red": (("y", "x"), pixels)}).assign_coords(xr_coords(grid)).gs.write_nodata(0)
scene = scene.expand_dims(time=[np.datetime64("2025-01-01")])

# Choose the Collection extent deliberately; publication preserves it.
everywhere = pystac.Extent(
    pystac.SpatialExtent([[-180, -90, 180, 90]]), pystac.TemporalExtent([[None, None]])
)
collection = pystac.Collection("forest", "Forest samples", everywhere)
# Raster publication accessors are paused. Write a file and describe it explicitly.
from geosave.geodata.stac.publish import create_stac_item

asset = scene.red.isel(time=0).gs.to_cog(root / "samples" / "red.tif")
item = create_stac_item(
    {"red": asset}, id="forest_20250101",
    input_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
    collection=collection.id, with_proj=True, with_raster=True,
)
items = gs.create_catalog([item], collections=[collection])
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

Raster `to_item` and `to_catalog` currently raise `NotImplementedError` while
publication moves to the Item/asset tree. Array `to_cog` and `to_gtiff` write
one explicit file; select observations before writing. The saved-file Item
constructor above remains available, with observation metadata supplied explicitly.

`gs.create_catalog(items)` converts existing native Items without raster writes.
Collections can remain in memory without a self href; STAC GeoParquet embeds
their definitions in its footer.

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

`stac.create_catalog` turns PySTAC Items, such as a static catalog's, into
an item table; `items.gs.to_items()` goes the other way, one PySTAC `Item` per row:

```python
import pystac
from geosave.geodata import stac

root_catalog = pystac.Catalog.from_file("stac/catalog.json")
items = stac.create_catalog(root_catalog.get_items(recursive=True))
```

`gs.open_vector` reads vector files and Item GeoParquet tables; use PySTAC for
a static STAC JSON catalog.

## Search first, then load onto a chosen grid

```python
from geosave.geodata import stac

query = gs.StacQuery(collections=["forest"]).set_region(region)
selected = client.search_items(query)
cube = stac.load_dataset(
    selected, geobox=grid, bands=["red"],
    collection=client.collection("forest"), patch_url=client.patch_url,
)
```

GeoRegion describes geometry and time only. `set_region` returns a new query,
replacing its spatial and temporal selectors while retaining other criteria.
A timeless region clears datetime; an existing bbox is cleared in favor of
WGS84 intersects. Select Items explicitly by ID by building a query with ids
and omitting set_region. No search is repeated by the pixel loader.

The grid is always an explicit choice. Match an existing raster with
`reference.gs.geobox`, or build one around a region with odc.geo:

```python
from odc.geo.geobox import GeoBox

grid = GeoBox.from_geopolygon(region.extent, resolution=10, crs="EPSG:32633")
cube = stac.load_dataset(selected, grid, bands=["red"])
```

Polygon bounds are snapped to a rectangular pixel grid; pixels outside the
polygon are not masked. Choosing Items, including by time, is the search's
job; the loader reads every Item it is given.

Load options use `StacLoadOptions` and keyword completion. The defaults are
`groupby="solar_day"` (Items of one local day mosaic into one label, whose
`start_datetime`/`end_datetime` state that day) and spatial Dask chunks of
1024. Pass `chunks=None` for eager loading, or `groupby="id"` for one plane per
Item. Native ODC settings such as `stac_cfg`, per-band dtype and resampling
remain available. Radiometric decoding stays explicit through
`cube.gs.mask_and_scale()`.

GeoSave retains CF axes, loader-authoritative nodata, agreed band metadata,
Collection/common attrs and requested per-time properties. An empty selection
raises RegionFetchError. Conflicting source metadata is omitted with a warning.
Collection metadata is supplied explicitly and never fetched by a loader.

```python
tree = stac.load_datatree(selected, collection=client.collection("forest"))
scene = tree.gs.reproject(grid)["forest"].gs.combine_stac()   # Items along time on one grid
```

A tree has a collection node and one node per STAC asset, each on the exact
grid its `proj:*` states; no grid is chosen and no band is resampled. Items
that state different grids for one asset are refused: load those onto one
chosen grid with `stac.load_dataset`. `row.gs.to_datatree()` and
`catalog.gs.to_datatree()` load the same way; `catalog.gs.to_dataset(grid)`
loads one collection onto one grid.

## Search a remote API

This example requires network access. Provider names are `planetary_computer`,
`cdse`, and `element84`. Catalog and asset credentials are separate.

<!-- NETWORK: public STAC API -->
```python
client = gs.StacClient.open("planetary_computer")
query = gs.StacQuery(collections=["sentinel-2-l2a"])
query = query.set_region(region).set_filter("eo:cloud_cover <= 10")
selected = client.search_items(query)
cube = stac.load_dataset(
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
cube = stac.load_dataset(selected, geobox=grid, bands=["red", "nir"])
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
query = gs.StacQuery(collections=["sentinel-2-l2a"]).set_region(region)
selected = planetary.search_items(query)
raster = stac.load_dataset(
    selected, geobox=grid, bands=["B04", "B08"],
    with_properties=["eo:cloud_cover"],
    collection=planetary.collection("sentinel-2-l2a"), patch_url=planetary.patch_url,
)
raster.gs.stac_properties              # one row per date: the requested properties
clear = raster.sel(time=raster.gs.stac_properties["eo:cloud_cover"] < 30)
clear.red.isel(time=0).gs.to_cog(root / "prepared" / "red.tif")
```

A loaded raster carries metadata for analysis and future tree publication:

| Metadata | Lives on | Read with |
| --- | --- | --- |
| Collection | root `StacCollection` attrs | `raster.gs.stac_collection` |
| Properties shared by selected observations | derived from `stac_properties` | `raster.gs.stac_common_properties` |
| Properties requested with `with_properties` | `stac_properties` coordinate, one JSON mapping per date | `raster.gs.stac_properties` |

The planned publisher will read each observation's property mapping.
No other coordinate becomes a property, and dates, `proj:*` and band fields
always come from the raster's own time, grid and attrs. A date odc-stac builds
from several Items takes its fused value of each requested property; an Item
lacking a key leaves it out. Cropping does not recalculate source cloud cover:
recompute it or drop it from `stac_properties` when it no longer describes the
output. Publication of derived products remains pending.

Reopening a catalog row with `row.gs.to_datatree()` puts its properties back on
`stac_properties` and its Collection on `StacCollection`, so catalog, xarray and
catalog again agree. Composite nominal time and explicit resampling bounds are
both published when those bounds are present on the raster. `stac.load_dataset`
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
Metadata-only `row.gs.region` reads Item geometry and time even without projection
metadata. `rows.gs.region` unions geometry in the frame CRS and envelopes dated
members; timeless members do not widen the interval. Standalone STAC rows must
remain in WGS84. Select projected rows as a one-row GeoDataFrame to retain CRS.
Chipping requires an explicit grid.
ODC loading is a geospatial operation and is not an ordinary-image fallback.

Shared observation properties are derived from `stac_properties`; they are not
stored separately in root attrs. Selecting time changes the observations used by
`raster.gs.stac_common_properties`. Loading retains properties shared by all
source Items in each observation record, alongside requested ODC property summaries.
Collection metadata remains separate in `stac_collection`.
