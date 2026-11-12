# Build catalogs and read samples

Use a catalog when you need to select observations across files or dates.
For one image and label file, start with [training chips](../quickstart.md).

This example creates three monthly observations, writes their assets, and groups
them into two-observation windows. Raster publication returns a native
GeoDataFrame whose rows describe those assets:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pystac
import xarray as xr
from odc.geo.xr import xr_coords

import geosave as gs

root = Path(tempfile.mkdtemp())
from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 500320, 5000320),
    crs='EPSG:32633', resolution=10,
)
dates = pd.date_range("2025-01-01", periods=3, freq="MS")
optical = (
    xr.Dataset(
        {"red": (("time", "y", "x"), np.ones((3, 32, 32), dtype="uint16"))},
        coords={"time": dates},
    )
    .assign_coords(xr_coords(grid))
    .gs.write_nodata(0)
)
# Any extent will do: to_catalog sets it to cover the Items it writes.
everywhere = pystac.Extent(
    pystac.SpatialExtent([[-180, -90, 180, 90]]), pystac.TemporalExtent([[None, None]])
)
collection = pystac.Collection("optical", "Monthly imagery", everywhere)
items = optical.gs.to_catalog(
    root, format_paths="{collection}/{time:%Y%m%d}/{variable}.tif", collection=collection
)
items[["id", "collection", "datetime"]]
```

`{collection}` uses the supplied collection id. `split_item=("time",)` makes one
Item per date and `split_asset=("variable",)` one asset per variable; the
template only names them, here by each asset's `{time}` and `{variable}`.
The same path collision checks as `to_cog` reject filename collisions before writing.
A DataTree uses its collection names for `{collection}` and can take a mapping
of collection names to path formats when dated and undated collections need
different paths. A collection on several grids, such as Sentinel-2's 10 m and
20 m bands, publishes one Item per date whose assets come from every grid, each
stating its own grid in `proj:*`; `{asset}` keeps the grids' files apart, since a
tree's asset names start with the node path.

Catalog assets are COGs, so path formats end in `.tif`; a Zarr or netCDF store is
written whole with `to_zarr` or `to_netcdf`. A catalog Item represents one
observation or composite, so multi-date publication needs `{time}`.

## Persist the Item table

```python
path = items.gs.to_geoparquet(root / "catalog.parquet")
items = gs.open_vector(path)
```

The writer retains Collection definitions, updates their extents, and stores
asset hrefs relative to the catalog so a folder can move with its assets.
Ordinary features and Item catalogs both use `gs.to_geoparquet(...)`. The
frame's `stac-geoparquet` metadata selects the STAC encoder. Invalid catalog
metadata or missing Item fields raise instead of silently losing information.

Every catalog row comes from a raster `to_catalog` wrote, or from a saved
catalog opened with `gs.open_vector`. A raster loaded from STAC carries its
Collection and Item properties, so its catalog keeps them; see
[STAC](stac.md#preserve-properties-and-collection-definitions).

## Group samples

```python
samples = items.gs.sample(collections=["optical"], window=2, freq="MS")
trees = []
for rows in samples:
    trees.append(rows.gs.to_datatree())
```

Sampling groups Items without opening pixels. Collections independently populate
calendar slots; no reference collection or nearest-date pairing is needed.
`window` and `step` select consecutive slots. Include static collections explicitly
when they should accompany windows. Crossing composite intervals raise by default;
choose `interval="first"`, `"last"`, `"overlap"`, or `"drop"` deliberately.

`GeoSamples` retains source positions, slot membership, and founding footprints.
`samples[i]` returns native source rows; `samples[i].gs.to_datatree()` opens their
observations with original timestamps and native grids under `/{collection}/{grid}`.
Use explicit temporal transforms after loading when reduction or interpolation
is needed. Grouping does not prepare pixels or align grids. See [calendar sampling](../concepts/catalog-windows.md)
for interval policies, missing slots, and forecasting constraints.

Reading one Item row with
`items.iloc[0].gs.to_datatree()` opens one node per asset of each Item.
`tree["s2"].gs.combine_stac()` merges each Item's assets and stacks the Items along
`time`, on a grid they already share. For mixed grids, choose one explicitly:

```python title="Adaptation example — a collection with several grids"
scene = tree.gs.reproject("EPSG:32633", resolution=20, resampling="average")["s2"].gs.combine_stac()
```

Combining never chooses a grid for you; tiles of one date need a mosaic, as
`rows.gs.to_dataset(geobox)` loads.
`samples.footprints` holds each sample's footprint and Item count;
`samples.footprints.gs.explore()` maps them, and `samples[i].gs.explore()` maps
one sample's rows, whose raster column opens each row's raster. Persist the
Item catalog and sample it again after reading.

## Choose the reading grid

Inspect the actual georeferencing without loading raster pixels:

```python
tree = rows.gs.to_datatree()
tree.gs.spatial_dims  # Shared spatial names, or None when nodes disagree.
nodes = tree.gs.datasets  # {node path: Dataset}, with lazy pixels preserved.
image = nodes["s2/<item>/B04"]
grid = image.gs.geobox  # CRS, shape, resolution, transform, and bounds.

# Split explicitly: one lazy slice per position, labels kept as scalar coords.
dates = image.gs.expand_rasters("time")
levels = temperature.gs.expand_rasters(("time", "level"))

# Request only the guarantees needed for the next operation.
tree.gs.validate(georeferenced=True)
scene.gs.validate(georeferenced=True, same_grid=True)
```

`expand_rasters` returns a tuple of slices of the same type, one per position.
Each slice carries its labels as scalar coordinates, or its position where a
dimension has none; other coordinates and spatial metadata stay as they are.
Validation raises `RasterValidationError`, a `ValueError` subclass available from
`geosave.geodata`. The error identifies the node and variable and suggests a
correction, such as assigning a known CRS with `raster.odc.assign_crs(...)`.
Calling operations may add context, such as a requested window that cannot be
read. Validation returns `None` on success, reads no raster pixels, and does not
cache results across coordinate or metadata changes.

Splitting needs explicit dimensions, which each slice must have; spatial axes
cannot be split. Spatial dimensions are the ones ODC names: `y`/`x` for a
projected grid, `latitude`/`longitude` for a geographic one. Rename other
layouts before using `.gs`.

`tree.gs.datasets` includes raster parents and children at any depth, keyed by
relative node path, with inherited coordinates. Metadata-only nodes are left
out. Inspection assigns no Collection membership and computes no pixels.
Different grids within a tree are valid unless `same_grid=True` is requested.

Publication selects a subtree as one observation series:

```python
catalog = tree["s2"].gs.to_catalog("output", collection=collection)
item = tree["s2"].gs.to_item("one-observation", collection=collection)
```

The selected root and all data-bearing descendants participate. Their grids can
differ, but their observation coordinates must agree. `split_item=("time",)`
partitions observations; `split_asset`, `("variable",)` by default, partitions
their assets.
`to_item` requires exactly one resulting observation.

Collection membership comes from `stac_collection` metadata or the supplied
native PySTAC Collection. A node path is an asset identity, not a Collection ID.
Use `{asset}`, `{variable}`, and each asset's scalar coordinates in
`format_paths`; a tree's asset names start with the node path.
Formatting never changes partitioning, and collisions fail before writes.

Each grid Dataset owns its spatial reference. Native Zarr preserves hierarchy;
STAC reopening reconstructs spatial grid groups rather than organizational paths.
Group enumeration order is not a persistent identifier.

`rows.gs.to_datatree()` preserves original asset grids. Successive Items of a
Collection must agree on their grid nodes, coordinates and variables so they
can concatenate exactly. Geographic overlap alone does not imply that their
pixels align. An accepted temporal sample can therefore fail native opening
when an acquisition is shifted. For a single Collection with differing grids,
select a native GeoBox and use `stac.load_dataset(items, geobox=grid)` explicitly.

## Cut raster chips

```python
scenes = []
for tree in trees:
    scenes.append(tree.gs.reproject(grid)["optical"].gs.combine_stac())
chips = gs.Chips(scenes, size=16, overlap=0.25, mode="reflect")
chip = chips[0, 0]  # (sample_index, tile_id): a lazy Dataset
chip.time                         # observation times, unchanged
chip.gs.region.extent.centroid    # this chip's location
pixels = chip.gs.to_numpy(dtype="float32")
```

`overlap` accepts integer pixels or a float fraction in `[0.0, 1.0)`.
Tiler rounds fractional overlap up to whole pixels; the resulting stride must
remain positive. For example, `size=16, overlap=0.25` shares four pixels.
`Chips` uses one Tiler per sample. `chips[index]` selects a tile across all
samples; `chips.key(index)` gives its `(sample_index, tile_id)` identity. Explicit
tuple indexing also works, and iteration yields keys.

A chip is `scene.gs.crop_to_grid(chips.geobox(key), mode=...)`: the source's own
xarray type on exactly the chip's grid. Its coordinates, grid mapping and
declared extent describe the chip; time and other coordinates, nodata and
attrs come from the source. Model context such as observation dates and
location is read from the chip itself. Converting to NumPy or tensors is the
caller's step, usually in a torch Dataset.

Construction reads metadata only, and indexing computes nothing. Keep source
files open until chip reads finish. No reprojection or extent selection happens
during chipping; every sample needs one regular grid.

`mode` accepts `"constant"`, `"reflect"`, `"edge"`, `"wrap"`, and `"drop"`.
Every tile has spatial shape `(size, size)`. Constant padding fills
`constant_value=0` by default, keeping the dtype. Reflection and wrapping apply
to the pixels inside the boundary tile, as in Tiler. `"drop"` omits incomplete
tiles; use padding when inference needs full coverage. Overlap changes tile
spacing and does not add a frame around the source.

Pass NumPy predictions and their keys to a fresh merge operation:

```python
def predictions():
    for key in chips:
        yield key, predict(chips[key].gs.to_numpy(dtype="float32"))

outputs = chips.merge(predictions(), window="hamming")
```

`window=None` uses uniform weighting. Named Tiler windows and NumPy weights
matching the native tile shape (flattened channels, height, width) are accepted.
Tiler owns normalization: repeated keys accumulate, missing tiles are allowed,
and pixels without contributions become zero. Edge-zero windows and `"drop"`
can leave gaps; Chips adds no frame padding or completeness policy.

Outputs are float32 NumPy arrays in source order. Leading prediction shapes must
agree within each sample; a source with no predictions returns None. Each call
starts fresh merger state. Place an output back on its source grid:

```python
source = chips.xarray[0]
y, x = source.gs.spatial_dims
output = xr.DataArray(outputs[0], dims=("class", y, x)).assign_coords(xr_coords(source.gs.geobox))
```

This example assumes class scores with one leading axis. Choose output dimensions,
variable names, labels, units and nodata to match your model. Input band/time
coordinates and attributes are not automatically appropriate for predictions.

`chips.xarray[i]` retains the original source by reference; do not change its
grid after Chips construction. Merging allocates full output arrays in memory,
even when input reading is lazy.

See the [catalog API](../reference/catalog.md) for the public interface and
[model input guide](model-spec.md) for preprocessing contracts.

```python
aligned.close()
for tree in trees:
    tree.close()
```

Item IDs are a publication choice. Defaults derive from observation time or
interval and additional split coordinates. Use `format_id` to customize them,
or `id` for one Item. Duplicate IDs within one Collection are rejected.
