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

## Publish raster assets

```python
import geosave as gs

catalog = ds.gs.to_catalog(
    "data", format_paths="{collection}/{time:%Y%m%d}/{variable}.tif", collection="s2"
)
catalog.gs.to_geoparquet("data/catalog.parquet")
```

The path names raster assets. Its suffix chooses COG, Zarr or NetCDF. Templates
use the same path fields as `to_cog`, such as `{collection}`, `{time}` and
`{variable}`; `split_item` and `split_asset` control file splitting. DataTrees use group names as collections and accept
one template or a mapping of group names to templates. A Dataset can receive
a collection id or a PySTAC Collection; the latter is retained in the GDF.
Each Item describes one observation, so multiple dates require `{time}` even
for Zarr and NetCDF publication.

Writes complete before the catalog is returned. Save the catalog separately
with `to_geoparquet`, the same method ordinary vector features use. The writer
selects STAC encoding when `attrs["stac-geoparquet"]` is present. It preserves
Collection definitions and portable asset references; malformed metadata or
missing Item fields raise instead of falling back to feature encoding.
A catalog comes from xarray: `to_catalog` writes the raster's files and builds
one Item per date from the raster itself, with the STAC metadata it carries.
A saved catalog opens again with `open_vector`.

## Group samples and open pixels

```python
samples = catalog.gs.sample(collections=["s2"], window=2)
tree = samples[0].gs.to_datatree().gs.group_grid()
scene = tree["s2"].gs.to_dataset(grid="r10m")
chips = gs.Chips([scene], size=128, constant_value=0)
```

Sampling groups Items without opening pixels and returns `GeoSamples`: the
item table, slot membership and each sample's row positions. `samples[i]` returns
one sample's native rows,
whose `gs.to_datatree()` opens a lazy DataTree, `/{collection}/{asset}`: one
node per STAC asset on the grid it states. `tree.gs.group_grid()` merges assets
sharing one grid into nodes named by pixel size (`r10m`, `r20m`).
One Item row's `gs.to_datatree()` opens its scene the same way, and
`.gs.to_dataset()` combines all raster nodes in the selected subtree, including
organizational groups, only when they share an exact grid. Select `tree["s2"]`
to limit the inputs and explicitly align mixed grids before conversion.
Variables must have unique names and compatible coordinates.
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
