# Publish and load: Dataset ⇄ catalog

Updated against library source on 2026-11-15. This page records implemented
boundaries and unresolved choices. For a runnable round trip, use the
[catalog tutorial](../../guides/catalogs-and-cuts.md).

## Current flow

```python title="Adaptation example — use a dated Dataset and destination grid"
rows = scene.gs.to_catalog("out/assets")
rows.gs.to_geoparquet("out/catalog.parquet")
rows = gs.open_vector("out/catalog.parquet")
selected = rows.loc[rows["eo:cloud_cover"] < 20]
scene = selected.gs.to_dataset(geobox=grid, chunks={}, with_properties=["eo:cloud_cover"])
```

```text
Dataset -> RasterStac Item/asset tree -> COGs + native Items -> Item table -> GeoParquet
Dataset <- ODC + metadata stamps <- native Items <- selected Item rows <- open_vector
```

`RasterStac` owns the prescribed publication layout and optional native
Collection. General DataTrees have caller-defined hierarchies. Loading returns
Datasets, with one array per band across observations; it does not construct
one tree node per Item and asset. Zarr and NetCDF store whole Datasets or trees
separately from this COG catalog route.

## Who does what

| Step | Owner |
| --- | --- |
| Dataset → staged Item/asset tree | `RasterStac.from_dataset(item_dim="time", id_format=...)`; no pixel reads |
| Join staged grids | `RasterStac.merge`; Items match by id, assets keep their grids |
| Write assets and describe them | `RasterStac.to_cog` writes `<root>/<item>/<asset>.tif`; `StacItem.from_region` and `add_raster_asset` build metadata |
| One-grid publication shortcut | `Dataset.gs.to_catalog` stages, writes and converts to rows |
| Items → rows | `stac.create_catalog` |
| Rows ⇄ file | `.gs.to_geoparquet`, `open_vector`; relative asset hrefs in the file |
| Rows → native Items | `.gs.to_items`, resolving hrefs against the source file |
| Rows → Dataset | `.gs.to_dataset`, validating Items and at most one collection |
| Items → Dataset | `stac.load_dataset`, ODC and registered stamps |
| Items → native-grid groups | `stac.load_native_datasets`, yielding `(crs, resolution, dataset)` |

## What round trips

COG publication is not a promise of xarray object identity. Select the grid,
properties and decode policy when loading.

| Part | Current behavior |
| --- | --- |
| Pixels and grid | Each asset states its grid; load onto that exact GeoBox to avoid resampling. The catalog tutorial checks pixel values and placement. |
| Time | Items state instants or periods. A period reloads using its start; a distinct original time label is not preserved. |
| Packing and nodata | Asset raster metadata and file encoding describe stored values; loading retains stored values until explicitly decoded. |
| Item properties | Only marked coordinates publish as properties; request them with `with_properties` on load. |
| Shared scalar property | Repeated in Items and loaded as a coordinate along time. |
| Collection | Carried separately from Item properties and stored in the table metadata. Its extent is not recomputed from written Items. |
| Arbitrary Dataset attrs | Not automatically published as Item properties. |
| Static variables | Written as an asset in every Item; they are not dropped. |

## Gaps to settle

- **Item identity:** default formatting can collide for repeated observations
  or differ between grids intended for one Item. Carry source identity, allow
  property-based formatting, or retain explicit formatting and reject collisions?
  Fused observations and derived products need their own identity rule.
- **Item dimension:** `from_dataset` requires an existing dimension, normally
  `time`. Scalar-time and timeless publication need a deliberate contract.
- **Property names:** ODC can turn `eo:cloud_cover` into `eo_cloud_cover`;
  the `stac_property` mark remains its identity. Should loaders rename coordinates,
  or continue comparing by the marked key?
- **Property selection:** retain explicit `with_properties`, or offer all scalar
  properties by default? `stac.list_properties(items)` already helps callers select.
- **Dataset attrs:** decide whether descriptive attrs such as `title` belong to
  Item properties, the Collection, or neither unless explicitly marked.
- **Native-grid results:** keep `(crs, resolution, dataset)` tuples or return only
  Datasets, which already state their grids?
- **Derived Collections:** callers currently replace source Collection metadata
  themselves. Decide whether publication should take an explicit Collection.
- **Catalog concatenation:** different source-file bases are refused. Resolving
  all hrefs on read could remove that restriction, at the cost of changing in-memory rows.
- **Publication options:** lazy writes, standalone STAC JSON, Zarr assets and
  classification extension output are not implemented in this path.
- **Write failure boundaries:** consider validating all destinations before the
  first asset write; current publication writes assets in sequence.
- **Decoded packing:** confirm asset declarations and stored encoding agree
  for decoded variables carrying packing only in `encoding`.

Staging reads metadata only; `to_cog` writes and therefore computes pixels.
The former promise of `RasterStac.to_cog(compute=False)` was never implemented.
Attach assets to their Item before adding extension declarations.

For large catalogs, select rows before `to_items`. Workers should open a catalog
by path instead of receiving a pickled copy; that ownership lesson survives the
older prototype without requiring its benchmark numbers as current guarantees.
