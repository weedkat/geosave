# Vector and raster catalog API

## Public interface

```python
import geosave as gs

catalog = ds.gs.to_catalog(
    "data", template="{collection}/{time:%Y%m%d}/{band}.tif", collection="s2"
)
catalog.gs.to_stac_geoparquet("data/catalog.parquet")
restored = gs.open_vector("data/catalog.parquet")
samples = restored.gs.sample(pair=["s2"], window=2)
for key, rows in samples.gs.samples:
    tree = rows.gs.to_xarray()

plots.gs.to_geoparquet("plots.parquet")
```

Every GeoDataFrame has one `GeoVector` accessor. Column changes do not change
the accessor class. Remove `GeoFrame` and `GeoCatalog`, without aliases. Keep
the existing Item-row accessor `GeoCatalogRow`; it represents an Item rather
than choosing between kinds of mutable frame. Catalog operations check their
required fields when invoked. Column recognition is not format validation.
`GeoDataFrame.gs` has one concrete static type; its generic schema parameter
is unnecessary.

`to_geoparquet` writes ordinary vectors; `to_stac_geoparquet` explicitly writes
Item rows, including Collections and portable relative asset hrefs. Reject
ordinary export of Item rows instead of silently losing their STAC metadata.
`GeoVector.concat` preserves Collection definitions when present. Sampling
remains in `core/vector/sample.py`, with its operation-specific keys.

`to_catalog` is shared by DataArray, Dataset and DataTree. A path names raster
assets, never the catalog file. The suffix selects COG, Zarr or NetCDF. Reuse
`io/raster/layout.py` for `{collection}`, `{time}` and `{band}`, validation and
collision detection. A Dataset's `collection=` supplies its collection name;
a DataTree uses its group names. A supplied PySTAC Collection is also retained
in the returned frame metadata. Writer options are passed to the selected
format; a template mapping must use one format. Writing completes before
constructing the catalog. The return value is always a native GeoDataFrame.

`path=None` raises ValueError explaining that unchanged-source verification is
unavailable. A source href survives native xarray arithmetic, variable renames
and selection, and cannot certify that the current object matches that file.
Existing files can still be described through `gs.create_catalog` and
`gs.stac.create_items` from explicit paths without writing pixels.

## Approaches

Keep schema-selected subclasses: less migration, but a cached accessor's type
can become stale after column edits. Use one stable accessor with explicit
STAC operations: removes the redundant hierarchy and makes persistence policy
visible. Implement the latter, following this conversation's simplification.

Direct xarray-to-Arrow construction could remove the intermediate PySTAC Item,
but would duplicate the existing Item/Asset/band metadata translation. Keep
those translators and the dependency's Arrow conversion for now. Do not add
an ambiguous `table.py` or a second metadata implementation.

## Scope and verification

Preserve native values, lazy catalog reads, grids, nodata, packing, band names,
Collection metadata and href portability. No notebook edits, compatibility
aliases, automatic source reuse, ModelSpec redesign or new dependencies.
Exercise stable accessor behavior, malformed Item input, ordinary/STAC export
separation, template splitting and collisions, writer failures, and a real
write/catalog/read pixel-and-metadata round trip. Migrate callers and docs.
