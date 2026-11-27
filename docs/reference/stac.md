# STAC discovery, loading, and publication

Use `StacClient` and `StacQuery` to choose observations, then load their pixels
on a supplied grid, or use `load_native_datasets` for native grids.
See [Search and load STAC data](../advanced/stac.md) for examples.

## Search and loading

::: geosave.geodata.stac
    options:
      members: [StacClient, StacQuery, StacTable, StacLoadOptions, load_dataset, load_native_datasets, group_bands_by_grid, create_catalog]

### Metadata stamps

Every load applies `stamp_nodata` to declare ODC's output fill. Without
`stamps`, it runs just that step. Pass stamps in execution order, each a name
or a function, such as `stamps=["eo", my_stamp]`.
Each receives the current Dataset, the selected PySTAC Items and the load's
`stac_cfg`, and returns the Dataset for the next step.

`geosave.geodata.stac.stamps.STAMPS` is the dictionary of built-in names.

| Name | Metadata written |
| --- | --- |
| `nodata` | Effective output fill for each band. The loader applies this automatically. |
| `raster` | Declared scale, offset and units for each band. |
| `eo` | Spectral band metadata and descriptions. |
| `classification` | Class values, names and declared colours. |
| `properties` | Item properties requested through `with_properties`, as marked coordinates. |
| `observation_period` | Start and end coordinates for each time plane. |
| `rgb` | RGB composite from band common names. |
| `title` | A single-band asset's title as its band's long name. |
| `collection` | Linked STAC Collection and Item extension schemas. |

::: geosave.geodata.stac.stamps
    options:
      members: [STAMPS, stamp_nodata, stamp_raster, stamp_eo, stamp_classification, stamp_properties, stamp_observation_period, stamp_rgb, stamp_title, stamp_collection]

## Publication

`stac.publish_item` consumes a native Item and an explicit asset mapping. Each
Item describes one observation or derived product. Each Dataset becomes one COG;
assets may have different grids and pixel representations.

```python
image = bands_10m.isel(time=0)
scl = bands_20m[["SCL"]].isel(time=0)
item = stac.StacItem.from_region(image.gs.region, id="observation-001")
# Set properties and an optional native Collection on the Item explicitly.
published = stac.publish_item(item, assets={"image": image, "scl": scl}, root="assets")
catalog = stac.create_catalog([published])
catalog.gs.to_geoparquet("catalog.parquet")
```

Files use `<root>/<item.id>/<asset key>.tif`. Publication returns a cloned Item
with actual written hrefs, projection, storage, band names and classification
metadata. Supplied Item geometry, properties and Collection links remain intact.
No tree shape, Item IDs, Collection membership or properties are inferred.

Slice observation dimensions explicitly, including length-one dimensions.
Scalar time/period metadata must agree with the Item; timeless/reference assets
are rejected. Explicit membership asserts provenance: copied timestamps cannot
prove it. All known metadata, representation and destination checks precede writes;
runtime I/O failures can leave earlier files. Publication is eager.

Use tree `split`, `merge`, `concat`, `ungroup` and effective `to_dataset` for
preparation when useful; the publisher consumes the resulting Datasets. It does
not merge independent asset grids. `RasterStac` and Dataset `to_catalog` have been
removed. Catalog loading returns Datasets and does not reconstruct a tree layout.

::: geosave.geodata.stac.publish.publish_item

::: geosave.geodata.stac.publish.StacItem

## Collection and observation metadata

Set the native Collection with `item.set_collection(collection)`. Loading may
stamp Collection metadata on Dataset attrs and properties on marked coordinates;
publishing requires explicitly choosing which of those describe the output.
Band declarations remain on variables. See [metadata rules](../concepts/metadata.md).

::: geosave.geodata.attrs.StacCollection

## Band and extension translation

These lower-level APIs translate native PySTAC values and xarray attrs.
Use accessor methods for everyday metadata edits.

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
