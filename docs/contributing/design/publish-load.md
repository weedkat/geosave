# Publish and load: Dataset ⇄ catalog

Updated 2026-11-17. General raster preparation and STAC membership have separate
owners; see the [tree specification](raster-tree.md) and
[catalog tutorial](../../basics/catalog.md).

## Current flow

```python
observation = scene.isel(time=position)
item = stac.StacItem.from_region(observation.gs.region, id=observation_id)
item.set_collection(collection)
assets = {"image": observation[["B04", "B08"]], "scl": observation[["SCL"]]}
published = stac.publish_item(item, assets=assets, root="out/assets")
rows = stac.create_catalog([published])
rows.gs.to_geoparquet("out/catalog.parquet")
loaded = gs.open_vector("out/catalog.parquet").gs.to_dataset(geobox=grid, chunks={})
```

Datasets can instead come from independent tree leaves and retain different grids.
Tree `split`/`merge`/`concat`/`ungroup` prepare structure; effective `to_dataset`
extracts one node. They assign no Item identity. Publication consumes explicit
membership and a native Item, preserves its geometry/properties/Collection links,
and returns a clone with actual written asset hrefs. No prescribed staging tree,
Item formatter or automatic property/Collection inference remains.

Each Item is one observation or derived product. Static reference inputs may
coexist with imagery in an analysis tree but must not enter the dated asset map.
Scalar times/periods must agree with the Item; unsliced dimensions and timeless
assets fail before writing. Explicit membership asserts provenance; copied scalar
timestamps cannot establish it. No mandatory provenance declarations are added.

```text
Dataset / extracted leaf -> explicit Item + asset map -> COGs + native Item
native Items -> create_catalog -> GeoParquet
selected rows -> native Items -> ODC + metadata stamps -> Dataset
```

## Who does what

| Step | Owner |
| --- | --- |
| Select observation positions and group variables | Native xarray or GeoDataTree operations |
| Define identity, properties, geometry and Collection | Native Item, optionally `StacItem.from_region` |
| Write files and describe assets | `stac.publish_item`, existing `geotiff.write_files`, extension translators |
| Items → rows | `stac.create_catalog` |
| Rows ⇄ file | `.gs.to_geoparquet`, `open_vector` |
| Rows → Dataset | `.gs.to_dataset`, ODC and metadata stamps |
| Items → grouped grids | `stac.load_native_datasets`, yielding `(crs, resolution, dataset)` |

All metadata, encoding and destination preflight happens before file writes.
Runtime failures can leave earlier files; publication is eager and nontransactional.
Loading assembles analysis data. It does not restore the published tree hierarchy;
use Zarr/NetCDF for structural persistence. Catalog-to-tree convenience is deferred
by agreement. Native-grid loading groups by CRS/resolution, so differing pixel
origins may align to an output grid; opening individual assets preserves file grids.

## What round trips

COG publication is not a promise of xarray object identity. Select the grid,
properties and decode policy when loading.

| Part | Current behavior |
| --- | --- |
| Pixels and grid | Each asset states its grid; load onto that exact GeoBox to avoid resampling. The catalog tutorial checks pixel values and placement. |
| Time | Items state instants or periods. A period reloads using its start; a distinct original time label is not preserved. |
| Packing and nodata | Asset raster metadata and file encoding describe stored values; loading retains stored values until explicitly decoded. |
| Item properties | Supplied native Item properties are preserved; explicitly select any values from marked coordinates. Request them with `with_properties` on load. |
| Shared scalar property | Repeated in Items and loaded as a coordinate along time. |
| Collection | Carried separately from Item properties and stored in the table metadata. Its extent is not recomputed from written Items. |
| Arbitrary Dataset attrs | Not automatically published as Item properties. |
| Static variables | Independent static/reference rasters stay outside observation asset maps; timeless assets are rejected. Copied scalar dates cannot prove provenance. |

## Gaps to settle

- Reference products without observation dates need a separate publication contract.
- Derived Collections still use explicit native PySTAC metadata and extent updates.
- Deferred publication, standalone STAC JSON and alternate asset formats are not
  part of the initial publisher.
- A standalone tree writer returning path-to-href mapping has no settled entry
  point/destination policy; publication associates assets by explicit keys.

Classification metadata, complete destination preflight and encoded packing are
implemented and tested. Runtime write failure remains nontransactional.
For large catalogs, select rows before `to_items`; workers can open the catalog
by path instead of receiving a pickled copy.
