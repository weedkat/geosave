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

The `time` coordinate gives the saved files their acquisition date. One image
holds one date, so `time` is a scalar coordinate rather than an axis; a series
of dates would span a `time` axis instead.

```python
day = np.datetime64("2025-01-01")
optical = optical.assign_coords(time=day)
label = label.assign_coords(time=day)
```

## 3. Group, save, and catalogue

These two rasters share a grid. `gs.tree` groups them without resampling;
other groups can retain different grids. Here each group becomes a STAC
collection, and `to_cog` writes one file per collection, date, and band.

```python
sample = gs.tree({"optical": optical, "label": label})
scenes = sample.gs.to_cog(root / "s0/{collection}_{time:%Y%m%d}_{band}.tif")
rows = gs.stac.create_catalog(gs.stac.create_items(scenes, id_format="s0"))
rows.gs.to_geoparquet(str(root / "catalog.parquet"))
rows[["collection", "datetime"]]
```

Output:

```text
  collection                  datetime
0    optical 2025-01-01 00:00:00+00:00
1      label 2025-01-01 00:00:00+00:00
```

`scenes` is a list of `Scene`s, one per collection and date, each mapping asset
keys to `RasterFile`s. `create_items` builds one Item from each, from the
headers they carry, reading nothing again and no pixels; `create_catalog` lays
them out as the item table. The table describes
the files this call saved; it does not scan the output folder.
Separate band files give the STAC assets the names `red`, `nir`, and `label`,
so loading through the catalog retains those variable names.

## 4. Cut windows

```python
items = gs.open_vector(str(root / "catalog.parquet"))
windows = items.gs.sample()   # one sample per shared footprint
windows = windows.gs.chip(128, target=anchor.geobox)         # split in space
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
| Save | Named raster groups in a DataTree | `Scene`s of `RasterFile`s | Computes and writes pixels; captures file headers. |
| Describe | Saved `Scene`s | Item GeoDataFrame | Uses their headers without reopening files. |
| Cut | Item/window GeoDataFrame | Window GeoDataFrame | Uses grid, time, and source metadata. |
| Open | One stack row | DataTree | Builds lazy raster reads on the window grid. |
| Convert | Raster or tree | Tensor or tensor mapping | Reads the selected pixels. |

## 5. Read a chip as tensors

```python
chip = windows.iloc[0].gs.to_xarray()          # lazy DataTree
optical = chip["optical"].dataset[["red", "nir"]].gs.mask_and_scale()
label = chip["label"].dataset[["label"]]
tensors = {
    "optical": optical.gs.to_tensor(dtype="float32"),
    "label": label.gs.to_tensor(),
}
chip.close()
{name: tuple(t.shape) for name, t in tensors.items()}
```

Output:

```text
{'optical': (2, 128, 128), 'label': (1, 128, 128)}
```

Decoding turns optical nodata (`0`) into NaN; any declared scale and offset
would also be applied. Label class codes stay unchanged. Conversion reads the
selected chip window; here the axes are `[band, y, x]`.

A Dataset identifies bands by variable name. Select them explicitly when
stacking them into positional tensor channels; their order after a storage or
catalog round trip is not the model's channel contract. Here the selection
sets channel 0 to red and channel 1 to near-infrared.

## Use the single-date sample

For a model taking one image, take the label's one band as an integer class
map:

```python
image = tensors["optical"]
target = tensors["label"][0].long()
known = image.isfinite().all(dim=0)
target[~known] = 255
image = image.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
image.shape, target.shape
```

Output:

```text
(torch.Size([2, 128, 128]), torch.Size([128, 128]))
```

The two input channels are red and near-infrared. Labels are class `0` or `1`,
with `255` marking pixels whose inputs are missing. This example explicitly
fills missing image values before returning model-ready inputs and the target;
the DataModule does not infer availability or repair input values. The
[training guide](guides/training.md) shows that sample contract and the
DataModule's remaining input preparation.

## Where next

- The core objects: [Rasters](basics/rasters.md), [Vectors](basics/vectors.md).
- Start with your files: [File ingestion](guides/ingest-files.md).
- Search real imagery: [STAC](guides/stac.md).
- Feed the chips to a model: [Training](guides/training.md).
