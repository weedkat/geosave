# Build catalogs and read samples

Save monthly images and a season's labels, then describe their files in an
item table:

```python
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import geosave as gs

root = Path(tempfile.mkdtemp())
anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=128, resolution=10,
    crs="EPSG:32633", timespan="2025-01-01",
)
months = pd.date_range("2025-01-01", periods=4, freq="MS")
rng = np.random.default_rng(0)
optical = gs.raster(
    {"red": (("time", "y", "x"), rng.integers(1, 3000, (4, 128, 128), dtype="uint16"))},
    anchor.geobox, nodata=0, time=months,
)
labels = gs.raster(
    {"label": (("y", "x"), rng.integers(0, 3, (128, 128), dtype="uint8"))},
    anchor.geobox,
)
season = (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 4, 30, 23, 59, 59, tzinfo=UTC))
optical_items = gs.stac.create_items(
    optical.gs.to_cog(root / "s0/optical_{time:%Y%m}.tif"),
    collection="optical", id_format="s0_optical_{start:%Y%m}",
)
label_items = gs.stac.create_items(
    labels.gs.to_cog(root / "s0/label.tif"),
    collection="label", id_format="s0_label", datetime=season,
)
rows = gs.stac.create_catalog(
    [*optical_items, *label_items],
    collections=[gs.stac.create_collection(name, description=name) for name in ("optical", "label")],
)
items = gs.open_vector(rows.gs.to_geoparquet(root / "catalog.parquet"))
items[["id", "collection"]]
```

Output:

```text
                  id collection
0  s0_optical_202501    optical
1  s0_optical_202502    optical
2  s0_optical_202503    optical
3  s0_optical_202504    optical
4           s0_label      label
```

Each row of the item table is one STAC Item: files of one collection on one
grid, dated by an instant or a span. `items.iloc[0].gs.to_xarray()` opens one
Item lazily on its native grid.

| Step | One row describes |
| --- | --- |
| Item table | One Item and its date or span. |
| `items.gs.sample()` | All observations sharing a spatial footprint, grouped by collection. |
| `samples.gs.frame(windows={"optical": {"slots": ["-1M", "0M"]}})` | Two consecutive months of a sample. |
| `frames.gs.chip(64, target=anchor.geobox, overlap=16)` | One pixel window on the supplied grid. |

No step opens a file; each reads Item metadata only. `frame` and `chip` give
the same rows in either order.

## Group observations sharing a footprint

```python
samples = items.gs.sample()
samples.iloc[0].gs.to_xarray()
```

The four images and season label share a footprint, so they form one sample.
Each collection keeps its native grid and dates. Sampling does not align grids,
match dates or open files. Different footprints remain separate samples,
including containing or partly overlapping footprints.

Image-only inference uses the same operation:

```python
optical_samples = items[items.collection == "optical"].gs.sample()
optical_frames = optical_samples.gs.frame(windows={"optical": {"slots": ["-1M", "0M"]}})
```

No target rows are needed to define the timeline.

## Request temporal bins

```python
frames = samples.gs.frame(
    windows={"optical": {"slots": ["-1M", "0M"], "reduce": "first"}},
    stride=1,
)
frames[["reference_time", "start_datetime", "end_datetime"]]
```

Each optical acquisition timestamp anchors a candidate frame, newest first.
The two slots cover the preceding and current relative calendar month.
`first` selects the earliest whole acquisition in each bin; missing bins
omit the frame with a warning. NaN pixels do not affect eligibility.
A composite must fit wholly inside its requested bin.

To anchor inputs to labels, set `reference="label"`; that reference observation
is attached without a separate slot declaration. Without a reference, only
collections in `windows` supply candidate timestamps. Unrequested dated
collections are not implicitly joined. Timeless auxiliary rasters stay attached.
See the [temporal contract](../concepts/catalog-windows.md) for reduction,
interpolation and interval boundaries.

## Cut space into chips

Supply `target=grid` to choose the pixel lattice. `grid` is an explicit GeoBox. Its
CRS, resolution and pixel alignment apply over every frame's footprint;
its original extent does not limit inference to the training location.
Alternatively, supply `resolution=10` to build a local grid in each row's
shared source CRS, with pixel edges snapped to multiples of that resolution.
Resolution is in CRS units; different rows may use different CRSs. If a row's
sources disagree on CRS, supply an explicit target grid. Do not combine target
and resolution. Existing chips can omit both to reuse their recorded grid.

```python
chips = frames.gs.chip(64, target=anchor.geobox, overlap=16)
chips[["start_datetime", "proj:shape", "pad_widths"]].head(3)
```

Output:

```text
  start_datetime proj:shape          pad_widths
0     2025-01-01   [64, 64]  [[24, 0], [24, 0]]
1     2025-01-01   [64, 64]   [[24, 0], [0, 0]]
2     2025-01-01   [64, 64]   [[24, 0], [0, 8]]
```

With overlap, a halo keeps the sample's border away from chip edges, so a
chip may reach past the sample. `pad_widths` counts those pixels; reading fills
them as `pad_mode` says, `reflect` by default.
Chipping records metadata only. `row.gs.to_xarray()` aligns the selected
rasters onto the recorded chip grid before combining observations in time.
The chip call can record per-variable resampling, for example
`resampling={"*": "nearest", "optical/red": "bilinear"}`. Nearest is the default
and preserves categorical labels.
Merging reconstructs the full layout grid stored in the chips, which can
differ from any source grid.

## Open one sample

```python
tree = chips.iloc[0].gs.to_xarray()
tree["optical"].dataset.red
```

Output:

```text
<xarray.DataArray 'red' (time: 2, y: 64, x: 64)> dask.array<chunksize=(1, 40, 40)>
```

Every collection uses the chip's grid, retaining the Items selected by framing.
A collection with a coarser source grid is resampled lazily to the requested
pixel size. The label, static over the frame, reads without a `time` dimension.
Unchipped samples and frames still keep native grids unless explicitly aligned.

## Save samples for another process

```python
saved = gs.open_vector(chips.gs.to_geoparquet(root / "chips.parquet"))
sorted(saved.gs.collections)
```

Output:

```text
['label', 'optical']
```

A sample table is plain GeoParquet; its Collection definitions sit in the
file's `geosave:collections` metadata, as an item table's sit under
`stac-geoparquet`. The Items inside a sample carry absolute asset addresses,
so the files must stay where they are.

## Maintain and merge

Append a batch with `gs.GeoCatalog.concat([old, new])`, then write the whole
table with `overwrite=True`. Collection definitions merge by id; the first
definition of each id is kept.

`frames.gs.merge(chips, outputs, bands=[...], taper="hann")` puts chip outputs
back on each frame's grid. Chips record their parent's spatial key and match
its time span, so nested chips remain distinct after reordering or persistence.
Parent and chip row labels must be unique, and chip grids must occupy the
parent's pixel lattice. See the [catalog reference](../reference/catalog.md).
