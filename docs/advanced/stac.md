# Search and load STAC data

Find observations by area and date, then choose how to read their pixels.
For a complete first download, follow [Download satellite imagery](download-imagery.md).
For local publication, follow [Store imagery as a catalog](../basics/catalog.md).

A STAC **Item** describes an observation, an **Asset** points to a file, and
a **Collection** describes the dataset its Items belong to. Search retrieves
metadata. Loading or computing a raster reads the assets.

## Search a remote API

This example needs internet access. It searches public Sentinel-2 metadata;
Planetary Computer asset reads use the URL signer supplied by the client.

```python
import geosave as gs
from geosave.geodata import stac

client = gs.StacClient.planetary_computer()
query = gs.StacQuery(
    collections=["sentinel-2-l2a"],
    bbox=(-121.82, 38.52, -121.79, 38.55),
    datetime="2025-06-01/2025-06-30",
    max_items=10,
).set_filter("eo:cloud_cover <= 10")
selected = client.search_items(query)
```

`search_items` returns native PySTAC Items; `search` returns a GeoDataFrame.
Use `StacClient.planetary_computer()`, `.element84()` or `.cdse()` for
built-in providers. Use `StacClient("https://example.org/stac/v1")`
for another API; this does not configure provider-specific asset signing.
A client connects on its first request. A filter selects the same Items on
every API: where a server ignores it, such as Earth Search, it is applied to
the returned Items.
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
    patch_url=client.patch_url, stamps=["raster"],
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

Loading retains stored digital numbers (DN). Here `stamp_raster` translates
the scale, offset and units the assets declare, checking that contributing
Items agree on each band's calibration. Decode those declarations with
`cube.gs.mask_and_scale()` before physical calculations. When a product needs
calibration from other fields, its Ingestor supplies the corresponding stamp.

Every load applies `stamp_nodata` to declare the output fill. Omit `stamps`
or pass `None` for the ODC grid, time axis and bands with that fill. For
additional metadata, pass one built-in name, one function, or a sequence of
both in execution order. The same argument works with `load_native_datasets`
and `rows.gs.to_dataset`.

```python
def my_stamp(ds, items, stac_cfg):
    return ds.assign_attrs(selected_item_count=len(items))

described = stac.load_dataset(
    selected, geobox=grid, bands=["B04"], chunks={}, patch_url=client.patch_url,
    stamps=["eo", my_stamp],
)
```

`"eo"` reads the band's spectral metadata. Then `my_stamp` receives that
Dataset, the original PySTAC Items and the load's `stac_cfg`. The built-in names are the keys of
`stac.stamps.STAMPS`.

Band stamps find a band alias defined in an ODC `stac_cfg` through the same
configuration the load used, so `stamps=["raster"]` works with aliases too.

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

Use the file produced by the [catalog tutorial](../basics/catalog.md):

```python title="Adaptation example — use your catalog path"
table = gs.StacTable("catalog.parquet")
query = gs.StacQuery(
    collections=["my-collection"], bbox=(-121.82, 38.52, -121.79, 38.55),
)
found = table.search(query)
```

`StacTable(path)` searches a local or remote GeoParquet file with
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

Choose the metadata the resulting Dataset needs.

```python
observations = stac.load_dataset(
    selected, geobox=grid, bands=["B04"], chunks={}, patch_url=client.patch_url,
    with_properties=["eo:cloud_cover", {"key": "platform", "dtype": "str"}],
    stamps=["properties", "observation_period", "collection"],
)
```

`observations.gs.stac_properties` maps the original STAC keys to their
coordinate names. The observation period stamp writes `start_datetime` and
`end_datetime`. The Collection stamp resolves the Items' linked Collection
and makes it available as `observations.gs.stac_collection`. ODC's native
`with_properties` option alone gives you properties as data variables.
For Items with start and end dates, include `PERIOD_REQUESTS` from
`geosave.geodata.stac.stamps.period` in `with_properties` so ODC fuses the
endpoints for each time plane.

A time plane built from several Items holds ODC's fused value: numbers
averaged, text joined with ",". Cropping does not recalculate scene cloud cover;
remove or recompute facts that no longer describe a derived output.

To publish, first slice each observation position and prepare its asset Datasets.
Call `stac.publish_item(item, assets=assets, root="assets")` with an explicit native
Item. Choose its properties from marked coordinates when they describe the output.
For a derived dataset, replace the source Collection with one describing the
new product. See [metadata behavior](../concepts/metadata.md).

## Regions and credentials

`gs.GeoRegion` describes geometry and optional time independently of a pixel
grid. `region.set_time("2025-06")` returns the same place covering the whole
month; a `(start_datetime, end_datetime)` tuple preserves exact endpoints.
`region.set_time(None)` clears time. `region.format("scene_{start:%Y%m%d}")`
builds a string without changing the region.

`query.set_region(region)` returns a query with replacement spatial and
time selectors; it retains other criteria. A timeless region clears datetime.

Search credentials and raster credentials are separate. Pass
`patch_url=client.patch_url` for signed assets; configure other remote storage
through [storage options and GDAL](../concepts/storage.md#use-remote-storage).
`StacTable(path, duckdb=session)` can search remote Item tables with a configured
rustac DuckDB session; see [STAC reference](../reference/stac.md) for its arguments.

Select rows with GeoPandas, then call `rows.gs.to_dataset(geobox=grid, chunks={})`
to load one collection. Catalog sampling remains unavailable.
