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
        - to_item
        - to_geoparquet
        - to_stac
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

## STAC levels as xarray

Each STAC level translates to its xarray counterpart, composing the extension
modules above: a Collection reads as a Dataset's root ACDD attrs, an Item the dated
files of one scene, an Asset one file and the variables it loads into, a band
one variable's attrs.

::: geosave.geodata.stac.translate.collection
    options:
      members: [read, describe]

::: geosave.geodata.stac.translate.item
    options:
      members: [instant, files, attrs, write]

::: geosave.geodata.stac.translate.asset
    options:
      members: [bands, names, attrs, fields, write]

::: geosave.geodata.stac.translate.band
    options:
      members: [read]

::: geosave.geodata.attrs.ACDD
