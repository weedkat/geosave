# Nested raster trees: collections, then grids

```python
tree = sample.gs.to_datatree()
# /s2
#   /r10m   (time, y, x)  blue green red nir
#   /r20m   (time, y, x)  rededge1 ... swir22
# /dem
#   /r30m   (y, x)        height

tree.gs.groups                       # ('s2', 'dem')
tree.gs.rasters                      # {'s2/r10m': Dataset, 's2/r20m': Dataset, 'dem/r30m': Dataset}
tree["s2"].gs.to_dataset()           # s2's bands on its finest grid, along time
tree["s2"].gs.to_dataset(res="r20m", resampling="average")  # the 10 m bands averaged onto 20 m
tree.gs.to_dataset()                 # ValueError: the tree holds collections ['s2', 'dem']; convert one

row.gs.to_datatree()                 # /r10m /r20m: one scene on its native grids
row.gs.to_datatree().gs.to_dataset() # the scene as one Dataset

tree.gs.to_catalog(root, template="{collection}/{time:%Y%m%d}/{band}.tif")
# one Item per collection and date; its assets come from every grid of that date
```

## Why nest

xarray's DataTree is meant for data at several resolutions and from several
sensors. Children must align exactly with their parents, and data that cannot
align, such as different grids, sits in sibling nodes. A collection is the
parent; its grids are siblings. ESA's EOPF Sentinel-2 product groups bands the
same way (`r10m`, `r20m`, `r60m`).

GeoSave's trees were flat: one level of groups, each a Dataset. A collection
whose assets differ in resolution could not be read without resampling, and a
multi-resolution product had to be named `s2-l2a-10m`, `s2-l2a-20m`, which
`to_catalog` could not map back to one collection. The flat invariant is replaced.

STAC stays flat. An Item's assets are one mapping, each stating its own grid
in `proj:shape` and `proj:transform`; asset keys may encode resolution, as the
Copernicus catalogue's `B02_10m`, `B02_20m`. The tree's grid groups exist only
in memory, derived from those stated grids, and are flattened back on writing.

## Shape

A raster tree has two levels below its root:

- **Collection nodes**, `/{collection}`. They hold no data variables; their
  attrs carry the collection's `StacMetadata`.
- **Grid nodes**, `/{collection}/{res}`. Each is a Dataset on one grid. `res`
  follows EOPF: `r` + pixel size + unit, `r10m`, `r20m`, `r60m`.

A scene, `row.gs.to_datatree()`, is one level: its root carries the Item's
date, properties and `StacMetadata`; its children are grid nodes.

Every leaf keeps its own `time`. Hoisting a collection's shared `time` to its
node is xarray-legal but not required, and is left out.

## Accessor contract

| Member | Meaning |
|---|---|
| `groups` | collection names, in order |
| `rasters` | every grid node as a Dataset, keyed by path (`"s2/r10m"`) |
| `variables` | `"s2/r10m/red"`, every leaf variable |
| `timespan`, `times` | over every leaf, as today over every group |
| `align(...)` | every leaf onto one grid; the tree is kept |
| `to_dataset(res=None, resampling="nearest")` | one collection as one Dataset |
| `map_groups`, `plot`, `to_numpy`, `to_tensor` | over leaves |

### `to_dataset`

`to_dataset` converts one collection: a node whose children are grid nodes,
such as `tree["s2"]` or a scene tree. A node whose children are collections
refuses, listing them: select one, `tree["s2"]`.

A collection's grid nodes are one grid at several zoom levels: the same CRS,
origin and extent, pixel sizes in whole-number ratios. Sentinel-2's 20 m grid
is its 10 m grid `zoom_out(2)`, its 60 m grid `zoom_out(6)` (odc-geo). Within
one collection the nodes come from the same Items, so dates, `stac_properties`
and variable names never clash; choosing a member of the family is the only
decision.

| Parameter | Meaning | Default |
|---|---|---|
| `res` | the grid node whose grid every node is brought onto: `"r20m"` | the finest node |
| `resampling` | a GDAL kernel, or one per variable with `"*"` for the rest, as `warp.reproject`, `align` and `StacSource` take | `"nearest"`, as everywhere in GeoSave and odc |

Nodes are brought onto the chosen member with `warp.reproject`, lazily. On
nesting grids GDAL's warp is the whole-pixel operation: `"nearest"` up to a
finer grid repeats each pixel, `"average"` down to a coarser grid takes each
block's mean, `"mode"` its most common value. The finest default keeps every
value, at the cost of memory: a 60 m band at 10 m holds 36 times its pixels.
Reflectance onto a coarser member reads
`to_dataset(res="r20m", resampling={"*": "average", "scl": "mode"})`.

Nodes that do not nest (other CRS, shifted origin, a ratio that is no whole
number) refuse and point to `align(target=...)`, which reprojects. A grid
from outside the collection, such as an anchor's, is `align(target=geobox)`
then `to_dataset()`: every node is resampled once, straight onto the target.

The Dataset carries the attrs every grid node shares, after each node takes
the attrs of the nodes above it up to the converted one, so a collection's
`StacMetadata` reaches its Dataset.

Joining collections is plain xarray:

```python
s2 = tree["s2"].gs.to_dataset()
dem = tree["dem"].gs.align(target=s2).gs.to_dataset()
prepared = s2.assign(height=dem.height)
```

`create_datatree` accepts `{collection: {res: Dataset}}`. A bare Dataset for a
collection becomes its one grid node, named from its grid.

Per-variable mappings (`resampling` in `to_dataset`, `align`, `reproject`)
key a variable as `"<collection>/<variable>"`, with `"*"` for the rest; the
grid is left out, since names are unique within a collection.

`open_tree` reads each entry of a folder or mapping as a collection with its
one grid node, as a bare key in `create_datatree`; a Zarr or NetCDF store
written as a tree reads back with its own group levels.

## Reading

```python
row.gs.to_datatree()     # each asset opened on its grid; assets sharing a grid merged into one node
rows.gs.to_datatree()    # per collection: its rows' scene trees concatenated along time, node by node
```

The only check the methods add: every row names a collection. Everything
else is enforced by the operations: `combine.merge` (a node's files share a
grid, no variable twice) and `concat` (rows share CRS, variables, dtypes and grid per node).

## Writing

Only GeoTIFF splits a raster into files: a GeoTIFF holds one grid at one date.
Zarr and NetCDF hold a whole tree natively, so they take no template.

```python
tree.gs.to_cog(root, template="{collection}/{res}/{time:%Y%m%d}/{band}.tif")
tree.gs.to_catalog(root, template="{collection}/{time:%Y%m%d}/{band}.tif")   # COGs + catalog rows
tree.gs.to_zarr("scene.zarr")     # one store: /s2/r10m ... with time axes, via DataTree.to_zarr
tree.gs.to_netcdf("scene.nc")     # one file with the same groups
open_tree("scene.zarr")           # reads its own group levels back
```

`expand_paths(raster, template)` lays a raster out as a flat
`{relative path: Dataset}`, one splitter per template field:

| Field | Splitter | Absent |
|---|---|---|
| `{collection}` | `split_by_collection`: a tree's collections | several collections collide, naming the field |
| `{res}` | `split_by_res`: a collection's grid nodes | a collection on several grids is refused, naming the field |
| `{time}` | `split_by_time`: dates | several dates are refused by the GeoTIFF writer |
| `{band}` | `split_by_band`: variables | all variables in one file |

A template may be one per collection. One collision check names the missing
field. `write_raster_files(files, writer)` writes `{UPath: Dataset}`, checking
each file with the format's own rules; it is the only file writer.

Catalog assets are files rasterio reads, written as COGs; STAC support for
Zarr is still immature. `publish_rasters` loops `split_by_collection`, lays
out and writes each collection's COGs, and builds one Item per collection and
date, its assets that date's files across every grid node, each stating its
grid in `proj:*`. A `.zarr` or `.nc` catalog template is refused, pointing to
`to_zarr` and `to_netcdf`.

## StacSource

`source.search(anchor)` lists the Items as the API stores them: whole tiles,
each on its own grid. `source.load_dataset(anchor)` stitches tiles of one date
and grids everything onto the anchor's GeoBox. `source.load_datatree(anchor)`
does the same, but keeps each band family at its native pixel size: the
anchor's grid zoomed to each family's resolution, `/{collection}/r10m` on the
anchor's GeoBox, `/r20m` on `anchor.geobox.zoom_out(2)`. odc-stac loads each
family with `bands=` and that zoomed GeoBox. Pixels are untouched only where
the anchor lies on the tiles' own grid.

## Removed

`GeoCatalogItem.to_xarray`, `GeoCatalog.to_xarray`, `open_scene(resolution=)`,
`resolve_group_paths`, `resolve_paths`, `write_group_files`, `template` on
`to_zarr` and `to_netcdf`, Zarr and NetCDF as catalog assets, and the
flat-tree rule in `create_datatree`.

## Out of scope

`ModelSpec`, which reads groups by collection name. Zarr or NetCDF stores as
catalog assets (STAC's xarray-assets `group` open option).

## Grids

A grid node's grid comes from, in order:

1. the file's own CRS and transform;
2. otherwise the asset's `proj:code`, `proj:shape` and `proj:transform`,
   assigned on read with `xr_coords(projection.to_geobox(asset))` when the
   stated shape equals the file's; a plain PNG becomes a georeferenced raster
   without reading pixels;
3. otherwise none: the raster is refused in a tree, naming the asset and
   saying that `proj:*` fields place it. `open_raster(path)` still opens it.

When the file and the asset state different grids, the file wins and a
warning names both, as publication treats the written file as the truth.

A grid node is named after EOPF: `r` + pixel size + the CRS's unit,
`r10m`, `r20m`; `r0.0001deg` for a CRS in degrees.
