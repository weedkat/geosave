# DataTree accessor

These methods are available as `data.gs` on an xarray `DataTree`.
Raster data belongs on leaves; grouping nodes carry attrs and coordinates.
Each leaf owns its grid coordinates; leaves may have different grids and timelines. Select a leaf and extract its
Dataset to read its grid: `tree["s2/image"].gs.to_dataset().gs.geobox`.

```python
tree = ds.gs.to_tree("s2", groups={"image": ["B04", "B03", "B02"], "mask": ["SCL"]})
image = tree["s2/image"].gs.to_dataset()
merged = tree["s2"].gs.merge()  # requires identical grids and shared indexes
flat = tree.gs.ungroup("s2")
```

Without `groups`, `ds.gs.to_tree()` keeps the whole Dataset under `/ds`.
Explicit groups partition every data variable exactly once. Shared Dataset attrs
stay on the source group. Grid coordinates are never automatically promoted;
all source coordinates, including auxiliary bounds,
remain on the leaves without changing variable dimensions. `.gs.to_dataset()` resolves ancestor attrs with local
overrides and includes inherited coordinates. Native `to_dataset()` is unchanged.

`group(groups)` partitions an existing leaf using the same complete,
exclusive groups as construction. `split("time", names=None)` selects each
position, retaining its original scalar coordinate. Names default to label
strings; duplicates require explicit unique names in positional order.

```python
from geosave.geodata.core.raster.rules.datatree import compatible_concat

split = ds.gs.to_tree("s2")["s2"].gs.split("time", names=["first", "second"])
# s2/time/first and s2/time/second; optional s2/static holds variables without time.
split["time"].gs.validate(compatible_concat("time"))  # optional preflight
split["time"] = split["time"].gs.concat(dim="time")
# If a static sibling exists, split.gs.merge() reconstructs the complete Dataset.
```

`merge()` combines distinct variables from every immediate child. `concat(dim=)`
assembles matching variables in child order, requiring matching units, packing,
flags, dtypes and storage encoding. Both require exact grid/shared-index alignment.
They return a new leaf named after the selected group. No pixel conversion occurs.

`ungroup(name)` returns the containing subtree with that direct group's children
promoted in place. For example, assign a split subtree back and use its parent's
`ungroup` to dissolve the source grouping. This rearranges hierarchy without
merging variables or choosing STAC membership. Source trees stay unchanged.

The default `raster_leaves` validation policy requires complete leaf-owned grids.
`same_grid` and `same_geobox` are additional rules when an operation needs alignment.
Rules live in `geosave.geodata.core.raster.rules.datatree`. Concat uses
`compatible_concat(dim)` through the same validation mechanism automatically.
Validation reads metadata and coordinate indexes, never raster pixels; it does
not change the tree or prepare incompatible representations.
Crop/reprojection use native tree mapping without changing parent coordinates.
Crop-to-grid clips declared extent attrs at their existing owner, including groups. Numerical conversion allows different channel
units, but requires matching dimensions and an explicit dtype when source dtypes
differ. Prepare individual Dataset leaves before tree conversion if needed.

::: geosave.geodata.core.raster.datatree.GeoDataTree
