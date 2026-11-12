# Catalogs

A catalog is a native GeoDataFrame of spatial records, read through the
`GeoCatalog` accessor. Records can describe boundaries or STAC Items with
raster assets. A spatial record selected as a pandas Series uses
`GeoCatalogItem`; an ordinary field column does not. Item operations check
the STAC fields they need when called. Neither accessor's type certifies
STAC GeoParquet compliance.

A selected Series does not inherit the GeoDataFrame's CRS. Item conversion
expects WGS84 coordinates; use a frame selection when reprojection or CRS
context is needed. A search matching nothing returns an empty item table with
its schema and CRS, although the explorer currently requires geometry to draw.

Select rows using native GeoPandas masks. Spatial targets must use the frame's
CRS; GeoSave does not infer a temporal filter from the target:

```python
shape = region.extent.to_crs(catalog.crs).geom
selected = catalog.loc[catalog.geometry.intersects(shape)]
selected.gs.validate(crs=True, temporal=True)
```

`temporal=True` requires paired, ordered bounds and a nominal `datetime` within
coverage, while permitting timeless rows. `area=True` requires valid geometries
with positive area. Geometry, CRS, temporal and Item-column checks can be
requested independently; `geometry=False` skips the default geometry check.

## Raster persistence and publication

Raster `to_item` and `to_catalog` currently raise `NotImplementedError` on
DataArray, Dataset and DataTree while publication moves to the Item/asset tree.

Write one selected array to an explicit file independently of STAC:

```python
path = ds["red"].isel(time=0).gs.to_cog("data/red.tif")
pending = ds["red"].isel(time=0).gs.to_gtiff("data/red-copy.tif", compute=False)
pending.compute()
```

These array writers do not split observations or format paths. Existing native
Items can still become a catalog with `gs.create_catalog(items)`. Save the
catalog with `catalog.gs.to_geoparquet(...)`; ordinary feature tables use the
same method, with format selection based on the table's STAC metadata.

## Group samples and open pixels

```python
samples = catalog.gs.sample(collections=["s2"], window=2)
tree = samples[0].gs.to_datatree().gs.reproject(grid)
scene = tree["s2"].gs.combine_stac()
chips = gs.Chips([scene], size=128, constant_value=0)
```

Sampling groups Items without opening pixels and returns `GeoSamples`: the
item table, slot membership and each sample's row positions. `samples[i]` returns
one sample's native rows,
whose `gs.to_datatree()` opens a lazy DataTree, `/{collection}/{item}/{asset}`:
one node per STAC asset on the grid it states. `tree.gs.reproject(grid)` aligns
every asset onto one grid. `tree["s2"].gs.combine_stac()` then merges each Item's
assets and stacks the Items along `time`, with their IDs as an `id` coordinate;
on an Item it merges that Item's assets. Collections keep independent timelines,
so combine one at a time. To go straight from a catalog to one grid, use
`catalog.gs.to_dataset(geobox)`, which also mosaics tiles of one date.
Chips accepts prepared Datasets or DataArrays, each on one regular grid, and
returns each chip as a lazy xarray object from `gs.crop_to_grid`. Select and align
the scene before chipping; convert chips with `gs.to_numpy(dtype=...)`.
`chips.xarray[i]` retains the original prepared source; `chips.merge(...)`
returns NumPy arrays in the same source order.
Construct output xarray objects explicitly with the coordinates and metadata
appropriate for predictions.

Persist the Item catalog and sample it again after opening. `GeoCatalog.concat`
merges Collection metadata; for repeated Collection ids, the first definition
wins.

::: geosave.geodata.core.vector.catalog.GeoCatalog
    options:
      members: [collections, concat, sample, to_items, to_datatree, to_geoparquet]

::: geosave.geodata.core.vector.samples.GeoSamples
    options:
      members: [indices, catalog, footprints, explore]

::: geosave.geodata.core.vector.item.GeoCatalogItem

## GeoParquet options

Compression and Parquet options apply to both feature and catalog output:

```python
catalog.gs.to_geoparquet("data/catalog.parquet", compression="zstd")
```

Catalog encoding requires WKB geometry, bbox information and no serialized
pandas index. Options conflicting with those constraints raise before writing.
`parquet_options={"row_group_size": 10000}` bounds catalog row groups through
the dependency's Arrow-stream input. `store_schema=False` is rejected for all
GeoParquet output because it would remove required geometry metadata.
`collections=` overrides definitions by id and is valid only for catalogs.

Catalog constructors and file readers attach the metadata marker. Selected
columns retain it, so rewriting an incomplete catalog raises for missing fields.
Item-shaped rows whose metadata was lost must be reconstructed explicitly with
`stac.create_catalog(...)`; the writer does not guess a catalog from column names.
