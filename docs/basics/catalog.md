# Store imagery as a catalog

Save an observation as Cloud Optimized GeoTIFFs with a searchable index of
where and when it was taken. Any GIS or STAC tool reads the files and the
index; GeoSave reopens them on a grid you choose.

**You need:** the base package and the [sample scene](first-raster.md#open-the-sample).
A single file is enough for one map; see [Read and save data](io.md). Use a
catalog once you keep several observations, dates or derived products.

## Publish one observation

```python
import geosave as gs
from geosave.geodata import stac

scene = gs.open_raster("sample-scene.tif")
item = scene.gs.publish_item("catalog/assets", id="20250627")
list(item.assets)
```

```text
['red', 'green', 'blue', 'nir']
```

`publish_item` writes one COG per band to `catalog/assets/20250627/` and
returns a STAC Item describing them: footprint, acquisition time, grid, data
type, nodata and scaling. The scene's scalar `start_datetime` and
`end_datetime` coordinates give the Item's time; no date is invented, so a
raster without a time is refused. The Item id names the observation and its
folder.

## Choose which bands share a file

```python
grouped = scene.gs.publish_item(
    "catalog/grouped",
    id="20250627",
    assets={"visual": ["red", "green", "blue"], "nir": ["nir"]},
)
list(grouped.assets)
```

```text
['visual', 'nir']
```

Each asset key names one file holding the listed bands. Bands in one file must
share a data type and value meaning; class maps and continuous values go to
separate files.

## Add a product on another grid

A derived product often has its own resolution. Publish it into the same Item
with `stac.publish_item`:

```python
ndvi = gs.features.ndvi(scene, nir="nir", red="red").rename("ndvi").to_dataset()
ndvi_20m = ndvi.gs.reproject(scene.gs.geobox.zoom_out(2), resampling="average")
item = stac.publish_item(item, assets={"ndvi_20m": ndvi_20m}, root="catalog/assets")
item.assets["red"].extra_fields["proj:shape"], item.assets["ndvi_20m"].extra_fields["proj:shape"]
```

```text
([337, 267], [169, 134])
```

Every asset records its own grid. Its time must agree with the Item's: an
asset belongs to this observation, not just to the same place.

## Save the index

```python
items = stac.create_catalog([item])
path = items.gs.to_geoparquet("catalog/observations.parquet")
print(path)
```

```text
catalog/observations.parquet
```

The table holds one row per Item with metadata and asset addresses; pixels stay
in the COGs. Asset addresses are absolute, so any STAC GeoParquet reader finds
the files; publish to their final location.

## Load imagery from the index

```python
rows = gs.open_vector(path)
loaded = rows.gs.to_dataset(geobox=scene.gs.geobox, chunks={})
sorted(loaded.data_vars), loaded.sizes["time"]
```

```text
(['blue', 'green', 'ndvi_20m', 'nir', 'red'], 1)
```

The GeoBox chooses the output grid, so the 20 m product is read onto the
10 m grid. Loading prepares lazy reads; it does not copy pixels into memory.

## Extend to more observations

Publish each observation as its own Item with a unique id, then list them in
one catalog:

```python title="One Item per date"
items = []
for position in range(series.sizes["time"]):
    observation = series.isel(time=position)
    day = observation.gs.timespan[0].strftime("%Y%m%d")
    items.append(observation.gs.publish_item("catalog/assets", id=day))
stac.create_catalog(items).gs.to_geoparquet("catalog/observations.parquet")
```

A model's predictions are published the same way: see
[Release and run a model](../advanced/release.md#publish-predictions).
Select Item rows with ordinary GeoPandas operations before loading. For
Collections, signed URLs and searching public catalogs, see the
[STAC guide](../advanced/stac.md) and the [STAC reference](../reference/stac.md#publication).
