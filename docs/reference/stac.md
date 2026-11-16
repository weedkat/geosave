# STAC discovery, loading, and publication

Use `StacClient` and `StacQuery` to choose observations, then load their pixels
on a supplied grid, or use `load_native_datasets` for native grids.
See [Search and load STAC data](../advanced/stac.md) for examples.

## Search and loading

::: geosave.geodata.stac
    options:
      members: [StacClient, StacQuery, StacTable, StacLoadOptions, load_dataset, load_native_datasets, group_bands_by_grid, list_properties, create_catalog]

## Publication

`RasterStac` holds a prescribed Item/asset tree and native Collection metadata.
`RasterStac.from_dataset(scene)` splits each time position into an Item and
its variables into asset Datasets, without computing pixels. Collection
metadata comes from the source attrs. Grouping and variable selection use native xarray. Static variables remain
in each group. Time and marked STAC property coordinates live on Item nodes and inherit
into assets. Item attrs retain `stac_extensions`; Collection metadata lives
on the wrapper. Spatial coordinates, band attrs, and encoding stay on assets.
Assets wrap individual DataArrays as Datasets. Variable attrs and encoding
are preserved; source Dataset attrs are not copied onto assets.

```python
opened = stac.RasterStac.from_dataset(scene, id_format="{start:%Y%m%d}")
red = opened.tree["20250601/red"].to_dataset(inherit="all_coords")
```

`item_dim` defaults to `"time"` and must exist on the Dataset.

The tree stages a publication; it is not a loading format. `merge` joins a
raster on another grid into the same Items, `to_cog` writes one COG per asset
and returns the Items, and `create_catalog` lays them out as a table.
`Dataset.gs.to_catalog(path)` does all three for one grid. Published Items
load back as Datasets.

```python
staged = stac.RasterStac.from_dataset(bands_10m).merge(
    stac.RasterStac.from_dataset(bands_20m)
)
items = staged.to_cog("s3://bucket/s2")      # <root>/<item id>/<asset>.tif
catalog = stac.create_catalog(items)
catalog.gs.to_geoparquet("s3://bucket/s2/catalog.parquet")

for crs, resolution, native in stac.load_native_datasets(
    catalog.gs.to_items(), chunks={}, with_properties=stac.list_properties(items)
):
    ...
```

::: geosave.geodata.stac.raster.RasterStac

`StacItem.from_region` builds one Item from a `GeoRegion`; `add_raster_asset`
states one file's grid and bands.

::: geosave.geodata.stac.publish.StacItem

## Collection and observation metadata

Collection metadata belongs to `RasterStac.collection` or Dataset root attrs.
Observation properties belong to coordinates marked with `stac_property`;
`gs.stac_properties` maps STAC keys to their coordinate names. Band declarations
stay on asset variables. See [metadata rules](../concepts/metadata.md).

::: geosave.geodata.attrs.StacCollection

Loading stamps live in `stac/stamps/`: Collection, band, source and property
stamps translate Item metadata onto the loaded Dataset.

## Band and extension translation

These lower-level APIs translate native PySTAC values and xarray attrs.
Use accessor methods for everyday metadata edits.

::: geosave.geodata.stac.extensions.projection
    options:
      members: [build_fields, read_geobox]

::: geosave.geodata.stac.extensions.raster
    options:
      members: [build_band, read_attrs]

::: geosave.geodata.stac.extensions.eo
    options:
      members: [build_band, read_attrs]

::: geosave.geodata.stac.extensions.classification
    options:
      members: [build_classes, read_attrs]

::: geosave.geodata.stac.catalog
