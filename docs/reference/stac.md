# STAC items and sources

These APIs create item tables from saved files, search catalogs, and load imagery.
See [Search and load STAC data](../guides/stac.md).

::: geosave.geodata.stac

## Item table accessor

An item table is still a GeoDataFrame. Its `rows.gs` accessor selects the STAC
writer automatically and includes Collection metadata. Spatial methods follow
the [vector accessor](vectors.md); GeoJSON and GeoPackage writes are refused
because those formats cannot preserve the catalog encoding.

::: geosave.geodata.core.vector.catalog.GeoCatalog
    options:
      heading: Item table accessor
      members:
        - collections
        - concat
        - to_item
        - to_geoparquet
        - to_stac_json
        - to_geojson
        - to_geopackage

## Extensions

Each extension module translates one PySTAC value to the xarray value it
describes and back; the band translator composes them for an attached Asset.

::: geosave.geodata.stac.extensions.projection
    options:
      members: [from_geobox, to_geobox]

::: geosave.geodata.stac.extensions.raster
    options:
      members: [from_dataarray, stamp]

::: geosave.geodata.stac.extensions.eo
    options:
      members: [from_dataarray, stamp]

::: geosave.geodata.stac.extensions.classification
    options:
      members: [from_dataarray, stamp]

## STAC levels and their GeoSave counterparts

A Collection is a Dataset's root ACDD attrs, an Item a `Scene`, an Asset a
`RasterFile`, and a band one variable's attrs.

::: geosave.geodata.stac.translate.collection
    options:
      members: [read_attrs]

::: geosave.geodata.stac.translate.item
    options:
      members: [from_scene, read_timespan, read_instant, read_grid, read_variable_names, assign_time, TIMELESS]

::: geosave.geodata.stac.translate.asset
    options:
      members: [from_file]

::: geosave.geodata.attrs.ACDD

::: geosave.geodata.stac.translate.band

::: geosave.geodata.stac.translate.table
