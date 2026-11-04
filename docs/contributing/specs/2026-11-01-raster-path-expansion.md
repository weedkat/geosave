# Raster path expansion

```python
# Raster layout has no filesystem knowledge.
parts = expand_paths(ds, "{time:%Y%m%d}/{band}.tif")
# {'20250601/red.tif': <Dataset ...>, ...}

# The writer resolves a parent and its filesystem options separately.
files = resolve_paths(ds, "s3://bucket/out", template="{time:%Y%m%d}/{band}.tif")
paths = write_raster_files(files, geotiff.write_cog)

# Public APIs distinguish an exact destination from a parent plus template.
ds.gs.to_zarr("out/scene.zarr")
ds.gs.to_zarr("out", template="{time:%Y%m%d}/{band}.zarr")
scene.gs.to_cog("out", template={"s2": "s2/{time:%Y%m%d}.tif", "dem": "dem.tif"})
```

`io/raster/paths.py` owns `expand_paths(raster, template, *, collection=None)
-> dict[str, xr.Dataset]`. It fills `{collection}`, `{time}` and `{band}` with
relative filenames mapped to their raster selections. It knows no UPath,
filesystem, credentials, or parent location. It preserves lazy pixels and all
coordinates and leaves an axis whole when its field is absent. It rejects
colliding filenames, missing or repeated dates, absolute names, URLs, and parent
traversal, including names introduced by field values.

`io/raster/writers.py` owns storage resolution. `resolve_paths(raster, path,
*, template=None, collection=None, storage_options=None)` treats `path` as an
exact destination when no template is supplied. It never formats that path.
With a template it converts the parent to UPath once, expands relative names,
joins them to the parent, and checks normalized destinations for collisions.
Storage options carried by a UPath are retained and explicit options merged.
`resolve_group_paths` does the same for tree groups, accepting group mappings
for paths and templates and rejecting missing groups and cross-group collisions.

`write_raster_files(files, writer)` writes the resolved selections and returns
their paths; `write_group_files(groups, writer)` returns paths keyed by group.
With `compute=False` each returns one native Dask Delayed: a single task writes
every file in turn because netCDF4 and HDF5 crash when two threads write at once.
`select_writer_by_suffix` selects the format only for `to_catalog`.

Every `gs` writer takes `path` and optional keyword `template`. It returns a list
for Dataset or DataArray and lists keyed by group for DataTree. An accessor
resolves destinations and binds writer options; it holds no write loop or
format preflight. The batch writer checks every destination before writing,
using the chosen format's supported suffixes. GeoTIFF owns grid, dimension,
one-instant and suffix checks in `validate`. Raster publication builds and
validates its Items and catalog table before the batch writer writes assets.
The Item translator owns observation metadata; the vector converter owns table
representation and duplicate Item identity checks.

This intentionally replaces embedded path templates: `to_cog("out/{band}.tif")`
becomes `to_cog("out", template="{band}.tif")`. Braces in an exact path or parent
are literal; braces within a template use Python format-string escaping.
