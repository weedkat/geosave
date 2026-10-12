# Build catalogs and read windows

Save images and labels, then describe their files in an item table:

```python
import tempfile
from pathlib import Path

import numpy as np

import geosave as gs

root = Path(tempfile.mkdtemp())
anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=128, resolution=10,
    crs="EPSG:32633", timespan="2025-01-01",
)
days = [np.datetime64("2025-01-01"), np.datetime64("2025-01-06")]
rng = np.random.default_rng(0)
optical = gs.raster(
    {"red": (("y", "x"), rng.integers(1, 3000, (128, 128), dtype="uint16"))},
    anchor.geobox, nodata=0,
).expand_dims(time=days)
labels = gs.raster(
    {"label": (("y", "x"), rng.integers(0, 3, (128, 128), dtype="uint8"))},
    anchor.geobox,
).expand_dims(time=days[:1])

sample = gs.stack({"optical": optical, "label": labels})
files = sample.gs.to_cog(root / "s0/{collection}_{time:%Y%m%d}.tif")
rows = gs.stac.create_items(files, id="s0")
path = rows.gs.to_geoparquet(root / "catalog.parquet")
items = gs.open_vector(path)
items[["collection", "datetime"]]
```

Output:

```text
  collection                  datetime
0    optical 2025-01-01 00:00:00+00:00
1    optical 2025-01-06 00:00:00+00:00
2      label 2025-01-01 00:00:00+00:00
```

This local example repeats one synthetic optical image on two dates and stores
labels on one date. The stack is an xarray `DataTree`; the item table is a
GeoPandas `GeoDataFrame`. Each row describes files from one collection at one
instant on one grid. The catalog contains metadata and file addresses, while
the TIFFs contain the pixels.

If you already have a catalog, start with `items = gs.open_vector(path)`.
See [STAC](stac.md) to search remote catalogs or describe Collection metadata.

## Group sources around a label

```python
windows = gs.cuts.stacks(items, on="label")
windows[["anchor", "proj:shape", "start_datetime", "end_datetime"]]
```

Output:

```text
              anchor  proj:shape start_datetime end_datetime
0  s0/label_20250101  [128, 128]     2025-01-01   2025-01-06
```

A **window** describes a model input: a grid and the source Items to read on it.
`stacks` creates one window per label Item, using the label's grid. Items from
other collections join when they cover the entire label footprint. This stage
operates on table metadata; it does not open image files.

Partially covering Items are omitted with a warning. A source collection with
two covering Items at the same instant keeps the first in table order and
warns. Check those warnings before using the table for training.

## Select time and pixel windows

```python
frames = gs.cuts.frames(windows, 2, tolerance="10D")
chips = gs.cuts.chips(frames, 64, overlap=16)
chips[["proj:shape", "start_datetime", "end_datetime"]].head(4)
```

Output:

```text
  proj:shape start_datetime end_datetime
0   [64, 64]     2025-01-01   2025-01-06
1   [64, 64]     2025-01-01   2025-01-06
2   [64, 64]     2025-01-01   2025-01-06
3   [64, 64]     2025-01-01   2025-01-06
```

`frames` selects sequences of two joint instants, matching observations within
10 days. A collection with one Item, such as these labels, joins every frame.
`chips` cuts each frame into 64 × 64 pixel regions. Neighbouring chips overlap
by 16 pixels; padding makes edge regions usable too. Both return ordinary
GeoDataFrames and still read no pixels.

Time matching selects observations; it does not interpolate their pixels or
change their acquisition dates. The [window rules](../concepts/catalog-windows.md)
explain joint instants, tolerance, incomplete frames, and padding.

## Open one window

```python
chip = chips.iloc[0].gs.to_stack()
chip
```

Output:

```text
<xarray.DataTree>
Group: /
│   Dimensions:      (y: 64, x: 64)
│   Coordinates:
│     * y            (y) float64 512B 4.988e+06 4.988e+06 ... 4.988e+06 4.988e+06
│     * x            (x) float64 512B 2.627e+05 2.627e+05 ... 2.633e+05 2.633e+05
│       spatial_ref  int32 4B 32633
├── Group: /label
│       Dimensions:      (time: 1, y: 64, x: 64)
│       Coordinates:
│         * time         (time) datetime64[s] 8B 2025-01-01
│           spatial_ref  int32 4B 32633
│       Data variables:
│           label        (time, y, x) uint8 4kB dask.array<chunksize=(1, 40, 40), meta=np.ndarray>
│       Attributes:
│           TIFFTAG_DATETIME:  2025:01:01 00:00:00
│           GEOSAVE_DATETIME:  2025-01-01
│           AREA_OR_POINT:     Area
└── Group: /optical
        Dimensions:      (time: 2, y: 64, x: 64)
        Coordinates:
          * time         (time) datetime64[s] 16B 2025-01-01 2025-01-06
            spatial_ref  int32 4B 32633
        Data variables:
            red          (time, y, x) uint16 16kB dask.array<chunksize=(1, 40, 40), meta=np.ndarray>
        Attributes:
            AREA_OR_POINT:  Area
```

This chip contains one label date and two optical dates. Its arrays are lazy,
and all groups are placed on the chip's grid. A source already on that grid is
sliced; a source on a different grid is reprojected. Backend reads may include
whole storage blocks covering the requested region.

Joining the dated optical files emits `DroppedAttrsWarning` for their differing
per-file date tags. The time coordinate retains both acquisition dates;
those single-file tags cannot describe the joined raster.

```python
chip.close()
```

Close a file-backed DataTree after using it, including after computing tensors.
Keep the row position alongside each model input so a prediction can be placed
back using `chips.iloc[position]`.

## Save the windows for another process

```python
path = chips.gs.to_geoparquet(root / "chips.parquet")
saved = gs.open_vector(path)
saved[["parent", "chip", "proj:shape"]].head(4)
```

Output:

```text
   parent  chip proj:shape
0       0     0   [64, 64]
1       0     1   [64, 64]
2       0     2   [64, 64]
3       0     3   [64, 64]
```

The saved window table carries its grid and sources, so another process can
open `saved.iloc[0].gs.to_stack()`. Its nested source addresses are absolute:
the source files must remain at those locations. An item table instead stores
asset addresses relative to its catalog, allowing that catalog and its assets
to move together. Cutting and saving windows does not give them the same
relocation behavior.

## Maintain and use a catalog

Append a batch with `gs.GeoVector.concat([old, new])`, then write the complete
table with `overwrite=True`. If you need existing Collection descriptions,
read them with `gs.stac.get_collections(path)` and pass them to the writer.
The current writer produces one table file; updating a collection rewrites
that file.

Store training and validation Items in separate catalogs, or partition the
tables yourself. GeoSave does not add a `split` column. A `ModelSpec` can apply
the configured cuts with `spec.frames.cut(windows)` and
`spec.chips.cut(frames)`; see [model specifications](model-spec-and-release.md).

For reconstruction, `gs.cuts.merge(frames, chips, predictions, taper=...)`
blends chip outputs back into each frame. See the
[cuts reference](../reference/cuts.md) for its expected prediction shape and
[window rules](../concepts/catalog-windows.md) for layout metadata.
