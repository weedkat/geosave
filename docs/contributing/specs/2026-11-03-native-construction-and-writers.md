# Native construction, folder writers, native tree mapping

```python
pred = xr.DataArray(labels, dims=("y", "x"), name="label")
pred = pred.gs.write_grid(chip.gs.geobox).gs.write_nodata(255)

scene = xr.Dataset({"red": (("time", "y", "x"), red)}, coords={"time": dates})
scene = scene.gs.write_grid(anchor.geobox)

tree = xr.DataTree.from_dict({
    "/s2": xr.Dataset(attrs=scene.attrs),
    "/s2/r10m": scene,
    "/dem": xr.Dataset(attrs=dem.attrs),
    "/dem/r10m": dem,
})
tree["s2"].gs.to_dataset()

scene.gs.to_cog("out")                                   # out/20250601T000000/red.tif ...
tree.gs.to_cog("s3://bucket/out", collections=["s2"], storage_options={...}) # out/s2/20250601T000000/red.tif ...
scene.isel(time=0).gs.to_cog("out", path_format="scene.tif")
rows = tree.gs.to_catalog("out", collections=["s2"])                         # same files, plus one Item per collection and date
tree.gs.to_zarr("scene.zarr")
open_raster("out/scene.tif")                             # one file or store
```

## Construction

GeoSave builds no xarray object. Rasters are built with xarray's own
constructors; GeoSave writes what xarray cannot express on its own, as it
already writes nodata, packing and legends.

Spatial meaning has two parts (CF, GDAL, odc-geo):

| Part | Where it lives | Written by |
|---|---|---|
| Placement: where each pixel is | `y`/`x` coordinates, pixel centres | `write_grid` |
| Reference: what the numbers measure | `spatial_ref` coordinate, linked from each variable by `grid_mapping` | `write_grid`, `write_crs` |

rioxarray can keep placement in a `GeoTransform` attr alone; GeoSave cannot,
since crop, align, chips and concat work through coordinates.

- `write_grid(geobox)` on DataArray and Dataset: checks the trailing `y, x`
  dimensions and their sizes against the grid, assigns `xr_coords(geobox)`,
  links every variable to `spatial_ref`, writes CF axis attrs. A `GCPGeoBox`
  is placed the same way, without CF axes.
- `write_crs(crs)` on DataArray and Dataset: the reference only, for
  coordinates that exist.
- Build trees with native `xr.DataTree` or `xr.DataTree.from_dict` and explicit
  paths. Put collection attrs on explicit parent Datasets when needed; raster
  accessors do not construct tree hierarchies.
- `Dataset.gs.to_dataarray()` stacks variables with GeoSave metadata.
  `DataArray.gs.to_dataset()` wraps a named array as one variable; explicitly
  pass `dim="band"` to split band coordinates into variables.
- `DataTree.gs.to_dataset()` merges compatible same-grid nodes. Align mixed
  grids explicitly with `tree.gs.align(how=reference, resampling=...)` first.

A grid is a `GeoBox` (CRS, transform, shape), and `write_grid` takes only that.
Dates are a coordinate, set where the xarray object is built.

Removed: `create_dataset`, `create_dataarray`, `create_datatree`, the
reference-raster branch of grid assignment, `create_catalog` and `core/ctor/`.

## Item tables

`GeoCatalog` is an accessor over any GeoDataFrame and changes none. Spatial
methods (`footprint`, `query`, `sample`, `plot`, `rasterize`, `concat`) need only a
geometry; Item methods (`to_items`, `to_datatree`, `collections`, a row's
`anchor`) need the stac-geoparquet columns and name what is missing, and where
Item tables come from, when called. A GeoJSON is an ordinary vector file:
`open_vector` returns plain features and does not look for STAC.

Item tables come from `to_catalog`, `StacClient.search`, reading a stac-geoparquet
file, and `stac_geoparquet.create_catalog(items, collections=())`, the one PySTAC
conversion, which search, `to_catalog`, `ModelSpec` and the flows share. With no
Items it returns the empty table with the Item columns that a search matching
nothing needs.

## Writing

Two kinds of persistence:

- **Stores** hold a whole object: `to_zarr(path)`, `to_netcdf(path)` on
  DataArray, Dataset and DataTree. One store, one `UPath`.
- **Files** hold one grid at one date: `to_cog(path, path_format=None)` and
  `to_gtiff` lay a raster out under the folder `path`; `to_catalog(path,
  path_format=None)` writes the same files and returns their Items.

`path_format` is filled with `str.format`, as `id_format` is. Its default is
`{collection}/{time:%Y%m%dT%H%M%S}/{band}.tif`, dropping `{collection}/` for a
raster of no collection and `{time:...}/` for one without dates. One file per
band is one STAC asset per band, and never needs `{grid}`.

Catalog assets are GeoTIFFs. `writers.py` is removed: layout lives in
`paths.py`, the GeoTIFF batch writer in `geotiff.py`, store writing in
`zarr.py` and `netcdf.py` (which take a Dataset or a DataTree), publication in
`publish.py`.

## Trees

Trees keep a bare root. Every tree operation maps the grid nodes with native
`DataTree.map_over_datasets`, skipping nodes without variables. `tree.gs.geobox`
reads the grid nodes: their one grid, or None. `align` returns grid nodes on
one grid; it no longer publishes the grid at the root.

Removed: `map_groups`, `replace_rasters`, root hoisting, `align(inplace=)`.

## Reading

`open_raster(path)` opens one file or store. A folder laid out by `to_cog`
reads back through its catalog. Removed: `combine_rasters`, `open_raster`'s
sequence, `geobox` and `resampling` arguments.
