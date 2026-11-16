# Create a reusable imagery catalog

Save a scene as named raster assets with a searchable index of their location
and acquisition date. Reopen that index to load the imagery on a chosen grid.

**You need:** the base package and the [sample scene](../basics/first-raster.md#open-the-sample).
This example publishes one real observation locally.

## Record the observation

```python
from pathlib import Path

import geosave as gs

scene = gs.open_raster("sample-scene.tif")
observations = scene.expand_dims(time=[scene.start_datetime.values])
Path("catalog").mkdir(exist_ok=True)
items = observations.gs.to_catalog("catalog/assets", id_format="{start:%Y%m%d}")
items["id"].tolist()
```

```text
['20250627']
```

The TIFF carries its acquisition start and end. The time dimension identifies
this as one observation; no date is invented. Publication writes one COG per
named band and returns one Item row describing the files. Pass
`collection=` to place the Items in a STAC Collection.

## Save the index

```python
path = items.gs.to_geoparquet("catalog/observations.parquet")
print(path)
```

```text
catalog/observations.parquet
```

The table contains metadata and asset addresses. Pixels stay in the TIFFs.
Asset addresses are absolute, so any STAC GeoParquet reader locates the
files; publish to their final location.

## Load imagery from the index

```python
rows = gs.open_vector(path)
loaded = rows.gs.to_dataset(geobox=scene.gs.geobox, chunks={})
sorted(loaded.data_vars), loaded.sizes["time"]
```

```text
(['blue', 'green', 'nir', 'red'], 1)
```

The GeoBox chooses the exact output grid. Loading prepares lazy reads; it
does not copy all pixels into memory.

## Extend to a collection

For multiple observations, provide a Dataset with a time dimension and unique
Item IDs. Select Item rows with ordinary GeoPandas operations before loading.

Use a catalog when location, dates and file discovery matter. Use
[training chips](overlapping-chips.md) for windows of already prepared imagery. For
multi-grid publication, Collection metadata and supported catalog operations,
see the [STAC reference](../reference/stac.md#publication).
