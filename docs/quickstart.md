# Quickstart

This example uses synthetic data and needs no network access. It builds a small synthetic image and label, saves them
as Cloud Optimized GeoTIFFs, lists them in a STAC item table, cuts model-sized
chips, and reads one chip as tensors.

## 1. Make a sample

An anchor says where and when: a grid of pixels on the ground plus a time span.

```python
import tempfile
from pathlib import Path

import numpy as np

import geosave_engine as gs

root = Path(tempfile.mkdtemp())
anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=256, resolution=10,
    crs="EPSG:32633", timespan="2025-01-01",
)
rng = np.random.default_rng(0)
pixels = lambda hi: rng.integers(0, hi, (256, 256), dtype="uint16")

optical = gs.raster(
    {"red": (("y", "x"), pixels(3000)), "nir": (("y", "x"), pixels(3000))},
    anchor.geobox,
    nodata=0,
)
label = gs.raster(
    {"label": (("y", "x"), pixels(3).astype("uint8"))}, anchor.geobox
)
optical
```

Output:

```text
<xarray.Dataset> Size: 266kB
Dimensions:      (y: 256, x: 256)
Coordinates:
  * y            (y) float64 2kB 4.989e+06 4.989e+06 ... 4.986e+06 4.986e+06
  * x            (x) float64 2kB 2.623e+05 2.623e+05 ... 2.648e+05 2.648e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 131kB 1527 2551 2763 1910 ... 1966 1038 1042 561
    nir          (y, x) uint16 131kB 797 813 1662 1534 ... 2028 1189 2057 611
```

`gs.raster` returns a plain `xarray.Dataset` with the grid stored as CF/GDAL
metadata.

## 2. Give each raster a date

<!-- PENDING-SCALAR-TIME -->

Writers name files after the `time` coordinate, so the raster needs a `time`
dimension. A scalar `time` coordinate is not accepted by `to_cog` yet.

```python
day = np.datetime64("2025-01-01")
optical = optical.expand_dims(time=[day])
label = label.expand_dims(time=[day])
```

## 3. Stack, save, and catalogue

A stack groups rasters on one grid. Each group is a STAC collection, and
`to_cog` writes one file per collection and date.

```python
sample = gs.stack({"optical": optical, "label": label})
files = sample.gs.to_cog(str(root / "s0/{collection}_{time:%Y%m%d}.tif"))
rows = gs.stac.create_items(files, id="s0")
rows.gs.to_geoparquet(str(root / "catalog.parquet"))
rows[["collection", "datetime"]]
```

Output:

```text
  collection                  datetime
0    optical 2025-01-01 00:00:00+00:00
1      label 2025-01-01 00:00:00+00:00
```

`files` are the records the writer returned; the item table is built from them,
so what is catalogued is exactly what was saved.

## 4. Cut windows

```python
items = gs.open_vector(str(root / "catalog.parquet"))
windows = gs.cuts.stacks(items, on="label")   # one window per label item
windows = gs.cuts.chips(windows, 128)         # split in space
windows[["proj:shape", "start_datetime", "end_datetime"]]
```

Output:

```text
   proj:shape start_datetime end_datetime
0  [128, 128]     2025-01-01   2025-01-01
1  [128, 128]     2025-01-01   2025-01-01
2  [128, 128]     2025-01-01   2025-01-01
3  [128, 128]     2025-01-01   2025-01-01
```

A cut takes a table and returns one; no pixels are opened.

## 5. Read a chip as tensors

```python
chip = windows.iloc[0].gs.to_stack()          # lazy DataTree
tensors = chip.gs.to_tensor()
chip.close()
{name: tuple(t.shape) for name, t in tensors.items()}
```

Output:

```text
{'label': (1, 1, 128, 128), 'optical': (1, 2, 128, 128)}
```

Pixels load here, and only the chip's own window is read. The tensors are
`[time, band, y, x]`.

## Where next

- The core objects: [Rasters](basics/rasters.md), [Vectors](basics/vectors.md).
- Search real imagery: [STAC](guides/stac.md).
- Feed the chips to a model: [Training](guides/training.md).
