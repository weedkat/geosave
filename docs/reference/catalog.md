# Catalog operations

A catalog is a GeoDataFrame of STAC Item records. The same `.gs` accessor also
handles ordinary vector features. Item operations check the columns they need;
an accessor's type alone does not certify STAC compliance.
See [Create and reopen a raster catalog](../guides/catalogs-and-cuts.md) for
an offline example.

## Publish and load pixels

```python title="Adaptation example — use your dated Dataset"
rows = observations.gs.expand().gs.to_catalog("assets")
path = rows.gs.to_geoparquet("catalog.parquet")
with gs.open_vector(path).gs.to_datatree() as tree:
    scene = tree["default"].gs.combine()
    result = scene.gs.mask_and_scale().mean("time").compute()
```

`expand` needs an Item dimension, `time` by default. Tree publication writes
one COG per asset at `folder/collection/Item/asset.tif` and returns one row per
Item. Save the table separately. Dataset/DataArray `.gs.to_catalog()` and
`.gs.to_item()` are unavailable.

Loading returns the same collection/Item/asset layout on native asset grids.
Combine a **collection node**; resample mixed grids onto an exact GeoBox first.
Collection metadata and Item properties remain separate from band metadata.

## Select and validate records

Select rows with GeoPandas masks. Selection geometry must use the frame's CRS.
A Series does not inherit that CRS; retain a one-row frame when it matters.

```python title="Adaptation example — use your catalog and region"
from geosave.geodata.core.vector.catalog import dated, stac_collections, stac_items, valid_geometry

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
are retained and asset hrefs are stored relative to the table.

`concat_catalogs` preserves compatible Collection definitions and rejects
conflicts. It currently refuses tables read from different source files,
whose relative asset hrefs have different bases.

::: geosave.geodata.stac.catalog.concat_catalogs

::: geosave.geodata.core.vector.catalog.GeoCatalog
    options:
      members: [validate, collections, to_items, to_datatree, to_geoparquet]

::: geosave.geodata.core.vector.item.GeoCatalogItem
    options:
      members: [validate, region, to_item, to_datatree]

## Unavailable operations

`sample`, `GeoSamples`, `to_dataset`, `read_pixels`, and vector `rasterize`
raise `NotImplementedError` while sampling and catalog pixel access are rebuilt.
Use GeoPandas selection and `.gs.to_datatree()` for current reads;
[Chips](chips.md) works with prepared rasters independently.

## GeoParquet options

Compression applies to feature and catalog output:
`catalog.gs.to_geoparquet("catalog.parquet", compression="zstd")`.
Catalog encoding requires WKB geometry, bbox information and no serialized
pandas index. Conflicting options raise before writing.
`parquet_options={"row_group_size": 10000}` controls catalog row groups.
`store_schema=False` is rejected because it would remove geometry metadata.
