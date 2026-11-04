# Native raster publication and loading

```python
paths = ds.gs.to_cog("data", template="{time:%Y%m%d}/{band}.tif")
pending = ds.gs.to_cog("later", template="{band}.tif", compute=False)
paths = pending.compute()
with gs.open_raster(paths) as restored:
    values = restored.gs.to_numpy()

catalog = tree.gs.to_catalog("data", template="{collection}/{time:%Y%m%d}/{band}.tif")
catalog.gs.to_geoparquet("data/catalog.parquet")
with gs.open_vector("data/catalog.parquet").gs.to_xarray() as restored:
    ...
```

## Ownership and public values

Remove io/raster/file.py and assemble.py, without aliases or replacement
wrapper classes. Format writers return a UPath of the artifact actually saved.
Dataset/DataArray to_cog returns list[UPath]; to_zarr/to_netcdf returns UPath.
DataTree writers return dictionaries keyed by group, with lists of COG paths
or one store path per group. compute=False returns one native dask.Delayed
whose result has the same eager shape; never a custom pending state machine.

open_raster accepts a path or sequence of paths. open_tree accepts a folder
or a mapping of groups to a path or sequence. Both I/O loaders own their open
resources: close sources on any failure and attach cleanup to returned native
Dataset/DataTree values. STAC loading applies the same lifetime ownership,
including exceptions while interpreting asset metadata. Pure transforms never
take ownership of caller-supplied Dataset resources.

transform/combine.py owns combining opened Datasets: grid alignment/windowing,
variable merging and temporal concatenation. Reuse the existing merge,
temporal.concat and warp operations. io/raster/stac.py keeps real STAC-specific
selection, href/band/date interpretation and actual-asset header reads. It
does not retain a parallel synthetic-header reconstruction path.

STAC Item/Asset translators accept actual native Datasets and asset paths.
create_items accepts explicit asset-name-to-path mappings for one observation,
or a path/sequence of paths grouped by their stored observation and grid.
Tree output dictionaries are group inventories: callers iterate groups and
supply collection= explicitly when describing them outside to_catalog.

to_catalog retains source observation coordinates internally during writing;
formats' actual saved band metadata remains authoritative. Use the actual
writer-returned inventory, then open asset headers lazily with deterministic
cleanup. This preserves cloud cover, nominal/composite dates, time bounds,
platform and other published properties that not every raster format stores.
Collections stay in the native GDF's stac-geoparquet attrs/footer; row.to_item
and the unified to_geoparquet contract remain intact. path=None stays rejected.

## Constraints and validation

Keep global path collision and all-file format/observation preflight. Preserve
grid, dtype, nodata, packing, band identities, native grids and lazy pixels.
Deferred writes must perform no writes/reads until compute, while still doing
metadata/path validation at construction. No compatibility aliases, new
dependencies, notebook edits, raster source-reuse detection or ModelSpec
redesign. Migrate callers/tests/docs to native inventories; do not preserve old
header-overlay tests by inventing another fake metadata object.

Verify the whole publication -> catalog persistence -> reopen -> raster load
-> close flow, plus failure cleanup, laziness, portable hrefs, composite
metadata, deferred writes, grouped trees and format round trips. Report the
38 recorded ModelSpec/template failures separately and compare exact names.
