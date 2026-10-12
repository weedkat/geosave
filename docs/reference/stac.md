# STAC items and sources

These APIs create item tables from saved files, search catalogs, and load imagery.
See [Search and load STAC data](../guides/stac.md).

::: geosave.geodata.stac

## Item table accessor

An item table is still a GeoDataFrame. Its `rows.gs` accessor selects the STAC
writer automatically and includes Collection metadata. Spatial methods follow
the [vector accessor](vectors.md); GeoJSON and GeoPackage writes are refused
because those formats cannot preserve the catalog encoding.

::: geosave.geodata.core.catalog.GeoCatalog
    options:
      heading: Item table accessor
      members:
        - to_geoparquet
        - to_geojson
        - to_geopackage

## Extension fields

Item creation calls these adapters to write file metadata through PySTAC's
extension classes. Use them when constructing custom PySTAC Assets.

::: geosave.geodata.stac.extensions.projection
    options:
      members: [write]

::: geosave.geodata.stac.extensions.raster
    options:
      members: [read, write]

::: geosave.geodata.stac.extensions.eo
    options:
      members: [read, write]

::: geosave.geodata.stac.extensions.classification
    options:
      members: [read, write]
