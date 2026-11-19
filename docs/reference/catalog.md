# Catalog operations

A catalog is a GeoDataFrame of STAC Item records. The same `.gs` accessor also
handles ordinary vector features. Item operations check the columns they need;
an accessor's type alone does not certify STAC compliance.
See [Store imagery as a catalog](../basics/catalog.md) for
an offline example.

## Publish and load pixels

```python
published = stac.publish_item(item, assets={"image": image, "scl": scl}, root="assets")
rows = stac.create_catalog([published])
path = rows.gs.to_geoparquet("catalog.parquet")
scene = gs.open_vector(path).gs.to_dataset(geobox=image.gs.geobox, chunks={})
```

Choose observation membership and Item metadata explicitly. Publication accepts
Datasets on independent grids; `create_catalog` composes completed native Items.
See [STAC publication](stac.md#publication).

Loading returns a Dataset on the chosen grid. Use `stac.load_native_datasets`
to keep each native grid. Collection metadata, marked Item property coordinates
and band metadata remain separate.

## Select and validate records

Select rows with GeoPandas masks. Selection geometry must use the frame's CRS.
A Series does not inherit that CRS; retain a one-row frame when it matters.

```python title="Adaptation example — use your catalog and region"
from geosave.geodata.core.vector.catalog import dated, valid_geometry
from geosave.geodata.stac.catalog import stac_collections, stac_items

shape = region.extent.to_crs(catalog.crs).geom
selected = catalog.loc[catalog.geometry.intersects(shape)]
selected.gs.validate(valid_geometry, stac_items, stac_collections, dated)
```

`validate(*rules)` applies only the rules passed, reads no assets, and returns
None on success. It raises `VectorValidationError` on failure.

| Rule | Checks |
| --- | --- |
| `valid_geometry` | Every geometry is present, nonempty and valid. |
| `stac_items` | Required Item columns, asset hrefs and unique IDs within collections. |
| `stac_collections` | A stored definition for every named Collection. |
| `one_collection` | At most one named Collection. |
| `dated` | Valid, complete observation time declarations. |

These checks do not validate the full remote STAC schema or scientific
suitability of pixels.

## Table persistence

`.gs.to_geoparquet` selects STAC encoding from the table's `stac-geoparquet`
metadata; ordinary features use GeoPandas encoding. Invalid catalog metadata
raises instead of silently falling back to features. Collection definitions
are retained. Asset hrefs are absolute, in the table and in the file; a file
another writer stored with relative hrefs is resolved against its own
location as it is read.

`concat_catalogs` preserves compatible Collection definitions and rejects
conflicts.

::: geosave.geodata.stac.catalog.concat_catalogs

::: geosave.geodata.core.vector.catalog.GeoCatalog
    options:
      members: [validate, collections, to_items, to_dataset, to_geoparquet]

::: geosave.geodata.core.vector.item.GeoCatalogItem
    options:
      members: [validate, region, to_item, read_pixels]

## Unavailable operations

Row `read_pixels` and vector `rasterize` raise `NotImplementedError` until
they are rebuilt. Load Item rows with `rows.gs.to_dataset(...)` or
`stac.load_dataset(rows.gs.to_items(), ...)`; [chips](chips.md) are cut from
prepared rasters with `gs.RasterChipper`.

## GeoParquet options

Compression applies to feature and catalog output:
`catalog.gs.to_geoparquet("catalog.parquet", compression="zstd")`.
Catalog encoding requires WKB geometry, bbox information and no serialized
pandas index. Conflicting options raise before writing.
`parquet_options={"row_group_size": 10000}` controls catalog row groups.
`store_schema=False` is rejected because it would remove geometry metadata.
