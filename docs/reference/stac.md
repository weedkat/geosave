# STAC discovery, loading, and publication

Use `StacClient` and `StacQuery` to choose observations, then load their pixels
on a supplied grid or retain each asset's native grid in a DataTree.
See [Search and load STAC data](../guides/stac.md) for examples.

## Search and loading

::: geosave.geodata.stac
    options:
      members: [StacClient, StacQuery, StacTable, StacLoadOptions, load_dataset, load_datatree, create_catalog]

## Publication

`Dataset.gs.expand()` builds a collection/Item/asset tree.
`tree.gs.to_catalog(folder)` writes assets and returns an Item table.
[Catalog operations](catalog.md) explains publication, persistence, and availability.

::: geosave.geodata.stac.publish.StacItem

::: geosave.geodata.stac.publish.create_stac_collection

## Collection and observation metadata

Collection metadata belongs to collection nodes or Dataset root attrs.
Observation properties belong to Item coordinates or a Dataset's
`stac_properties` coordinate. Band declarations stay on the asset's variables.
See [metadata rules](../concepts/metadata.md).

::: geosave.geodata.attrs.StacCollection

::: geosave.geodata.stac.item
    options:
      members: [ItemTime, normalize_time, create_item, select_item_properties, read_timespan, read_instant, TIMELESS]

::: geosave.geodata.stac.raster
    options:
      members: [encode_properties, decode_properties, stamp_item_time, stamp_item_properties, build_stac_metadata]

## Band and extension translation

These lower-level APIs translate native PySTAC values and xarray attrs.
Use accessor methods for everyday metadata edits.

::: geosave.geodata.stac.band

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
