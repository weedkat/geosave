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
    selected, geobox=grid, bands=["B04", "B08"], chunks={},
    patch_url=client.patch_url,
)
```

The GeoBox chooses the exact output CRS, resolution, shape and placement.
Use `reference.gs.geobox` to match an existing raster. Bounds snap to a
rectangular grid; the loader does not mask pixels outside a search polygon.
It reads the Items supplied without repeating the search. Options are
`odc.stac.load`'s, with its defaults; an empty selection raises ODC's
`ValueError`.

| Loading choice | Behavior |
| --- | --- |
| Default `groupby="time"` | Mosaic Items with one timestamp into one time plane. |
| `groupby="solar_day"` | Mosaic Items from one local solar day into one time plane. |
| `groupby="id"` | Keep one plane per Item. |
| Default `chunks=None` | Load eagerly, as ODC does. |
| `chunks={}` or sizes | Build Dask-backed pixel reads. |
| `resampling="nearest"` or a band mapping | Choose ODC resampling explicitly. |

Loading retains stored digital numbers (DN). Decode with
`cube.gs.mask_and_scale()` before physical calculations. The loader rejects
selected sources with conflicting scale/offset declarations.
The Collection is the one the Items link to, as `item.get_collection()`
resolves it: searched Items reach it through their catalog, catalog rows carry
it, and a hand-built Item names it with `item.set_collection(collection)`.

## Keep each asset's native grid

```python
for crs, resolution, native in stac.load_native_datasets(
    selected, bands=["B04", "B08", "B11"], chunks={}, patch_url=client.patch_url
):
    native.gs.to_zarr(f"s2_{abs(resolution.x):g}m.zarr")
```

`load_native_datasets` loads once per group of `stac.group_bands_by_grid`, the
bands sharing the CRS and resolution their `proj:*` states, finest first. With
no grid option, ODC loads each group onto its native grid, the finest among the
group's bands, which is every band's own: no band is resampled. A region option (`bbox`, `geopolygon`) crops each group at its own
resolution. Bands without `proj:*` are left out; load them onto a chosen grid
with `geobox=`.

## Search a saved Item table

Use the file produced by the [catalog tutorial](catalogs-and-cuts.md):

```python title="Adaptation example — use your catalog path"
table = gs.StacClient.open("catalog.parquet")
query = gs.StacQuery(
    collections=["my-collection"], bbox=(-121.82, 38.52, -121.79, 38.55),
)
found = table.search(query)
```

Opening a `.parquet` path returns a `StacTable`, which searches the file with
the same queries as an API. `table.collections()` lists its Collection ids;
`found.gs.collections` retains their definitions. To read the whole table
without filtering, use `gs.open_vector("catalog.parquet")`. Search does not
load raster pixels.

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
property per time plane. ODC names it `eo_cloud_cover`; the coordinate's
`stac_property` attr keeps the STAC key, and `cube.gs.stac_properties` maps each
key to its coordinate. Text properties need `{"key": ..., "dtype": "str"}`.
Collection metadata lives separately in `cube.gs.stac_collection`.

A time plane built from several Items holds ODC's fused value: numbers
averaged, text joined with ",". Cropping does not recalculate scene cloud cover;
remove or recompute facts that no longer describe a derived output.

To publish a Dataset with a time dimension, use
`cube.gs.to_catalog("assets")`. Publication reads marked Item property
coordinates, and time and grid facts from the raster.
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

Select rows with GeoPandas, then call `rows.gs.to_dataset(geobox=grid, chunks={})`
to load one collection. Catalog sampling remains unavailable.
