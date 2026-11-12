# STAC Items, discovery and metadata

These APIs create item tables from saved files, search catalogs, and load imagery.
See [Search and load STAC data](../guides/stac.md).

::: geosave.geodata.stac

## Item table accessor

An item table is a GeoDataFrame with the same `GeoCatalog` accessor as ordinary
features. `to_geoparquet` selects STAC encoding from the catalog's metadata,
retaining Collection definitions and portable asset references. The same method
writes ordinary feature tables with GeoPandas encoding. See the
[catalog API](catalog.md) for publication, grouping and metadata requirements.

::: geosave.geodata.core.vector.catalog.GeoCatalog
    options:
      heading: Item table accessor
      members:
        - collections
        - concat
        - sample
        - to_geoparquet
        - to_items

## Extensions

Each extension module builds a native PySTAC value from an xarray variable
and translates it back into named attrs. The band translator composes these
attrs for an Asset; the loader applies them to the raster.

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

## STAC levels and their GeoSave counterparts

A Collection, and the properties all of a raster's Items share, live in the
root `StacCollection` attrs; each date's Item properties live on the
`stac_properties` coordinate. An Item describes one observation, an Asset a
saved raster, and a band one variable. Translators work with native Datasets
and asset paths.

::: geosave.geodata.attrs.StacCollection

::: geosave.geodata.stac.item
    options:
      members: [ItemTime, normalize_time, create_item, select_item_properties, read_timespan, read_instant, TIMELESS]

::: geosave.geodata.stac.raster
    options:
      members: [stamp_properties_coordinate, read_properties_coordinate, stamp_item_time, stamp_item_properties, build_stac_metadata]

::: geosave.geodata.stac.publish.create_stac_item

::: geosave.geodata.stac.band

::: geosave.geodata.stac.catalog
