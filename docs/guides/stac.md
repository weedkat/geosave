# Search and load STAC data

Find observations by area and date, then choose how to read their pixels.
For a complete first download, follow [Download satellite imagery](download-imagery.md).
For local publication, follow [Create and reopen a raster catalog](catalogs-and-cuts.md).

A STAC **Item** describes an observation, an **Asset** points to a file, and
a **Collection** describes the dataset its Items belong to. Search retrieves
metadata. Loading or computing a raster reads the assets.

## Search a remote API

This example needs internet access. It searches public Sentinel-2 metadata;
Planetary Computer asset reads use the URL signer supplied by the client.

```python
import geosave as gs
from geosave.geodata import stac

client = gs.StacClient.open("planetary_computer")
query = gs.StacQuery(
    collections=["sentinel-2-l2a"],
    bbox=(-121.82, 38.52, -121.79, 38.55),
    datetime="2025-06-01/2025-06-30",
    max_items=10,
).set_filter("eo:cloud_cover <= 10")
selected = client.search_items(query)
```

`search_items` returns native PySTAC Items; `search` returns a GeoDataFrame.
Provider names include `planetary_computer`, `element84`, and `cdse`.
Collection IDs and band names depend on the provider. A scene-wide cloud score
does not guarantee clear pixels in your area.

## Search first, then load onto a chosen grid

```python
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox

grid = GeoBox.from_bbox(
    BoundingBox(-121.82, 38.52, -121.79, 38.55, crs="EPSG:4326").to_crs("EPSG:32610"),
    resolution=10,
)
cube = stac.load_dataset(
    selected, grid, bands=["B04", "B08"],
    collection=client.collection("sentinel-2-l2a"),
    patch_url=client.patch_url,
)
```

The GeoBox chooses the exact output CRS, resolution, shape and placement.
Use `reference.gs.geobox` to match an existing raster. Bounds snap to a
rectangular grid; the loader does not mask pixels outside a search polygon.
It reads the Items supplied without repeating the search. An empty selection
raises `RegionFetchError`.

| Loading choice | Behavior |
| --- | --- |
| Default `groupby="solar_day"` | Mosaic Items from one local solar day into one time plane. |
| `groupby="id"` | Keep one plane per Item. |
| Default spatial chunks of 1024 | Build Dask-backed pixel reads. |
| `chunks=None` | Load eagerly; unlike backend-lazy file reading. |
| `mask_and_scale=True` | Decode stored values into physical values. |
| `resampling="nearest"` or a band mapping | Choose ODC resampling explicitly. |

Opening normally retains stored digital numbers (DN). Decode with
`cube.gs.mask_and_scale()` before physical calculations. The loader rejects
selected sources with conflicting scale/offset declarations.
Collection metadata is supplied explicitly; the pixel loader never fetches it.

## Keep each asset's native grid

```python
with stac.load_datatree(
    selected,
    collections=[client.collection("sentinel-2-l2a")],
    bands=["B04", "B08"],
    patch_url=client.patch_url,
) as tree:
    scene = tree["sentinel-2-l2a"].gs.combine(["B04", "B08"])
    result = scene.gs.mask_and_scale().mean("time").compute()
```

A tree keeps `collection/Item/asset` nodes. Each asset has the exact grid its
Projection metadata states; assets and Items can differ in grid. `combine`
requires the selected assets to share CRS, resolution and pixel alignment.
Extents may differ. For mixed grids, use
`tree.gs.reproject(grid)["sentinel-2-l2a"].gs.combine(...)` or load directly
onto a grid with `load_dataset`.

`load_datatree` and catalog `.gs.to_datatree()` require asset grids in
`proj:*`. When that metadata is absent, supply a GeoBox to `load_dataset`, or
open a known asset file directly with `gs.open_raster`. No geographic location
is invented. Consume lazy pixels before closing the source tree.

## Search a saved Item table

Use the file produced by the [catalog tutorial](catalogs-and-cuts.md):

```python title="Adaptation example — use your catalog path"
client = gs.StacClient.open("catalog.parquet")
found = client.search(gs.StacQuery(collections=["default"]))
```

Use the collection IDs actually stored in your table. `found.gs.collections`
retains their definitions. Search does not load raster pixels.

For a static STAC JSON catalog, use native PySTAC:

```python title="Adaptation example — use your STAC JSON catalog"
import pystac

catalog = pystac.Catalog.from_file("stac/catalog.json")
rows = stac.create_catalog(catalog.get_items(recursive=True))
```

`rows.gs.to_items()` converts a table back to native Items. `gs.open_vector`
reads vector files and Item GeoParquet tables, rather than STAC JSON catalogs.

## Preserve properties and Collection definitions

Pass `with_properties=["eo:cloud_cover"]` to `load_dataset` to retain a
requested property per observation, alongside the properties shared by its
source Items. Read them from `cube.stac_properties`; Collection metadata lives
separately in `cube.gs.stac_collection`.

A time plane built from several Items uses their shared properties and ODC's
requested property summaries. Cropping does not recalculate scene cloud cover;
remove or recompute facts that no longer describe a derived output.

To publish a Dataset with a time dimension, use
`cube.gs.expand().gs.to_catalog("assets")`. The tree publisher reads Item
properties from `stac_properties`, and time and grid facts from the raster.
For a derived dataset, replace the source Collection with one describing the
new product. See [metadata behavior](../concepts/metadata.md).

## Regions and credentials

`gs.GeoRegion` describes geometry and optional time independently of a pixel
grid. `query.set_region(region)` returns a query with replacement spatial and
time selectors; it retains other criteria. A timeless region clears datetime.

Search credentials and raster credentials are separate. Pass
`patch_url=client.patch_url` for signed assets; configure other remote storage
through [storage options and GDAL](../concepts/storage.md#use-remote-storage).
The same client can search remote Item tables with a configured rustac
DuckDB session; see [STAC reference](../reference/stac.md) for its arguments.

Catalog sampling and direct `rows.gs.to_dataset()` are currently unavailable.
Select rows with GeoPandas, load a tree, and combine a collection explicitly.
