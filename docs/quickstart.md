# From images to training chips

Create a small image and label, save them, then read a spatial chip as tensors.
This example uses synthetic data and works offline. Install `geosave[ml]`
for the final tensor conversion. Read
[Your first raster](first-raster.md) first if xarray rasters and grids are new to you.

The data moves through three forms: pixels in xarray, file metadata in a catalog,
and selected pixels in PyTorch tensors.

## 1. Make a sample

A GeoBox defines the pixel grid. Acquisition dates are coordinates on the raster; a GeoAnchor describes geometry and time for spatial queries.

```python
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

import geosave as gs

root = Path(tempfile.mkdtemp())
from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 502560, 5002560),
    crs='EPSG:32633', resolution=10,
)
rng = np.random.default_rng(0)
def pixels(hi):
    return rng.integers(0, hi, (256, 256), dtype="uint16")

optical = (
    xr.Dataset({"red": (("y", "x"), pixels(3000)), "nir": (("y", "x"), pixels(3000))})
    .gs.write_grid(grid)
    .gs.write_nodata(0)
)
label = xr.Dataset(
    {"label": (("y", "x"), pixels(2).astype("uint8"))}
).gs.write_grid(grid)
optical
```

Output:

```text
<xarray.Dataset> Size: 266kB
Dimensions:      (y: 256, x: 256)
Coordinates:
  * y            (y) float64 2kB 5.003e+06 5.003e+06 5.003e+06 ... 5e+06 5e+06
  * x            (x) float64 2kB 5e+05 5e+05 5e+05 ... 5.025e+05 5.026e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 131kB 1527 2551 2763 1910 ... 1966 1038 1042 561
    nir          (y, x) uint16 131kB 797 813 1662 1534 ... 2028 1189 2057 611
```

`xr.Dataset` is built as usual; `gs.write_grid` stores the supplied grid on it as
CF/GDAL metadata.

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

These two rasters share a grid. `to_datatree` makes each one a collection, and
`xr.merge` groups them without resampling; other collections can retain different
grids. Each collection becomes a STAC collection, and `to_catalog` writes one file
per collection, date, and band.

```python
sample = xr.merge([optical.gs.to_datatree("optical"), label.gs.to_datatree("label")])
rows = sample.gs.to_catalog(root / "s0", id_format="s0")
rows.gs.to_geoparquet(str(root / "catalog.parquet"))
rows[["collection", "datetime"]]
```

Output:

```text
  collection                   datetime
0    optical  2025-01-01 00:00:00+00:00
1      label  2025-01-01 00:00:00+00:00
```

`to_catalog` writes the rasters and reads their actual file metadata to build
one Item per collection and date. It retains source observation properties
that the raster format cannot store. The table describes the returned written
paths; it does not scan the output folder.
Separate band files give the STAC assets the names `red`, `nir`, and `label`,
so loading through the catalog retains those variable names.

## 4. Cut windows

```python
items = gs.open_vector(str(root / "catalog.parquet"))
samples = items.gs.sample()   # one sample per shared footprint
opened = samples[0].gs.to_datatree()
chips = gs.Chips([opened], size=128, resolution=10)
len(chips)
```

Output:

```text
4
```

The four chips cover 128 × 128 windows on the saved scene. Creating their
layout uses grid metadata and leaves the pixel arrays lazy.

| Step | Input | Output | Execution |
| --- | --- | --- | --- |
| Publish | Named raster groups in a DataTree | Item GeoDataFrame | Writes pixels; reads actual file metadata. |
| Group | Item GeoDataFrame | Grouped Item rows | Uses footprints, dates and collection names. |
| Open | One sample group | DataTree | Builds lazy raster reads. |
| Chip | Raster trees | `Chips` | Calculates windows from grid metadata. |
| Convert | Raster or tree | Tensor or tensor mapping | Reads the selected pixels. |

## 5. Read a chip as tensors

```python
tree_index, chip = chips[0]          # lazy DataTree
optical = chip["optical"].gs.to_dataset()[["red", "nir"]].gs.mask_and_scale()
label = chip["label"].gs.to_dataset()[["label"]]
tensors = {
    "optical": optical.gs.to_tensor(dtype="float32"),
    "label": label.gs.to_tensor(),
}
opened.close()
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
