# From images to training chips

Create a small image and label, save them, then read a spatial chip as tensors.
This example uses synthetic data and works offline. Read
[Your first raster](first-raster.md) first if xarray rasters and grids are new to you.

The data moves through three forms: pixels in xarray, file metadata in a catalog,
and selected pixels in PyTorch tensors.

## 1. Make a sample

An anchor says where and when: a grid of pixels on the ground plus a time span.

```python
import tempfile
from pathlib import Path

import numpy as np

import geosave as gs

root = Path(tempfile.mkdtemp())
anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=256, resolution=10,
    crs="EPSG:32633", timespan="2025-01-01",
)
rng = np.random.default_rng(0)
def pixels(hi):
    return rng.integers(0, hi, (256, 256), dtype="uint16")

optical = gs.raster(
    {"red": (("y", "x"), pixels(3000)), "nir": (("y", "x"), pixels(3000))},
    anchor.geobox,
    nodata=0,
)
label = gs.raster(
    {"label": (("y", "x"), pixels(2).astype("uint8"))}, anchor.geobox
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

The `time` coordinate gives the saved files their acquisition date. Here a
singleton dimension keeps one-date and multi-date model inputs in the same
form. `to_cog` also accepts a scalar `time` coordinate for one image.

```python
day = np.datetime64("2025-01-01")
optical = optical.expand_dims(time=[day])
label = label.expand_dims(time=[day])
```

## 3. Stack, save, and catalogue

These two rasters share a grid. `gs.stack` groups them without resampling;
other groups can retain different grids. Here each group becomes a STAC
collection, and `to_cog` writes one file per collection and date.

```python
sample = gs.stack({"optical": optical, "label": label})
files = sample.gs.to_cog(str(root / "s0/{collection}_{time:%Y%m%d}.tif"))
rows = gs.stac.create_item(files, id="s0")
rows.gs.to_geoparquet(str(root / "catalog.parquet"))
rows[["collection", "datetime"]]
```

Output:

```text
  collection                  datetime
0    optical 2025-01-01 00:00:00+00:00
1      label 2025-01-01 00:00:00+00:00
```

`files` is a tuple of the destination `UPath` objects. `create_item` reads their
headers to build the item table, without reading pixels. The table describes
the files this call saved; it does not scan the output folder.

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

The four rows describe 128 × 128 windows on the saved scene. A cut takes a
table and returns one; it does not read the pixel arrays.

| Step | Input | Output | Execution |
| --- | --- | --- | --- |
| Save | Named raster groups in a DataTree | TIFF paths | Computes and writes pixels. |
| Describe | Saved TIFF paths | Item GeoDataFrame | Reads headers. |
| Cut | Item/window GeoDataFrame | Window GeoDataFrame | Uses grid, time, and source metadata. |
| Open | One window row | DataTree | Builds lazy raster reads on the window grid. |
| Convert | Raster or stack | Tensor or tensor mapping | Reads the selected pixels. |

## 5. Read a chip as tensors

```python
chip = windows.iloc[0].gs.to_stack()          # lazy DataTree
decoded = chip.gs.mask_and_scale()
tensors = decoded.gs.to_tensor()
chip.close()
{name: tuple(t.shape) for name, t in tensors.items()}
```

Output:

```text
{'label': (1, 1, 128, 128), 'optical': (1, 2, 128, 128)}
```

Decoding turns optical nodata (`0`) into NaN; any declared scale and offset
would also be applied. Conversion to tensors reads the selected chip window.
Here the axes are `[time, band, y, x]`.

## Use the single-date sample

For a model taking one image, remove the singleton time axis and make the
label an integer class map:

```python
image = tensors["optical"][0]
target = tensors["label"][0, 0].long()
valid = image.isfinite().all(dim=0)
target[~valid] = 255
image.shape, target.shape
```

Output:

```text
(torch.Size([2, 128, 128]), torch.Size([128, 128]))
```

The two input channels are red and near-infrared. Labels are class `0` or `1`,
with `255` marking pixels whose inputs are missing. These are the `image`,
`target`, and `valid` values a segmentation Dataset returns. The
[training guide](guides/training.md) shows that sample contract and the
DataModule's remaining input preparation.

## Where next

- The core objects: [Rasters](basics/rasters.md), [Vectors](basics/vectors.md).
- Start with your files: [File ingestion](guides/ingest-files.md).
- Search real imagery: [STAC](guides/stac.md).
- Feed the chips to a model: [Training](guides/training.md).
