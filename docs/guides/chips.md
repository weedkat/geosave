# Read overlapping chips and merge predictions

Split the [sample scene](../first-raster.md#open-the-sample) into windows,
process each one, and place the merged result back on its original grid.
No catalog, model, or GPU is needed.

**You need:** the base package and the sample TIFF. This example copies one
band through the tiling workflow to check reconstruction before substituting a model.

## Read one window

```python
import numpy as np
import xarray as xr

import geosave as gs
from geosave.geodata import io

source = gs.open_raster("sample-scene.tif")
image = source[["red", "nir"]]
tree = xr.DataTree.from_dict({"image": image})
chips = gs.RasterChipper([tree], size=128, stride=96, mode="reflect")
chip = chips[0]["image"].to_dataset()
pixels = chip.gs.to_numpy(dtype="float32")
pixels.shape, chips.layout.locate(0)
```

```text
((2, 128, 128), ChipLocation(raster_index=0, top=0, left=0))
```

RasterChipper takes a list of DataTrees, each with its raster nodes on one exact
GeoBox. Here the tree holds one image; it could also hold a label Dataset.
Construction reads grids only. Indexing returns a lazy tree cropped to the
chip grid. Selecting the image Dataset and calling `to_numpy` reads pixels;
storage blocks or upstream calculations may require reads beyond the window.
Keep `source` open until all chip reads finish.

The integer index identifies a chip across all input rasters.
`chips.layout.locate(index)` gives its source raster and pixel offsets.
Select bands explicitly to fix channel order. Tensor conversion does not
decode scaling; call `.gs.mask_and_scale()` first when physical values are needed.

## Merge a result

```python
merger = gs.RasterMerger(chips.layout, window="hamming")
for index in range(len(chips)):
    values = chips[index]["image"].to_dataset().gs.to_numpy(dtype="float32")
    merger.add(index, values[:1])  # Stand-in prediction: copy the red band.

red = merger.merge(0).isel(band=0, drop=True).rename("red_copy")
np.allclose(red.values, image["red"].values)
```

```text
True
```

Predictions are NumPy arrays shaped `(band, size, size)`, even for one band.
They may arrive in any order; pass the original chip index to `add`.
A weighted merge blends overlaps and removes padding outside the source grid.
For class logits, merge scores first and choose the class afterward.

`merge(raster_index)` returns a float32 DataArray with dimensions `(band, y, x)`
on that source's GeoBox. Uncovered pixels are NaN; a source with no predictions
raises ValueError. Repeated indices accumulate contributions. Give predictions
their own names, units, class legend and nodata policy.

```python
from pathlib import Path

path = Path("output/red-copy.tif")
path.parent.mkdir(parents=True, exist_ok=True)
io.write_cog(red.to_dataset(), path)
source.close()
```

## Choose overlap and edge behavior

| Option | Behavior |
| --- | --- |
| `size=128, stride=96` | Share 32 pixels between adjacent chips. |
| `stride=None` | Use the chip size as stride, with no overlap. |
| `mode="constant", constant_value=0` | Fill outside the source with an explicit value. |
| `mode="reflect"`, `"edge"`, or `"wrap"` | Extend the whole source raster using that padding rule. |
| `mode="shift"` | Move the final chips inward to fit; requires a source at least one chip wide and high. |
| `mode="drop"` | Omit incomplete boundary chips. |
| merger `window="boxcar"` | Average overlaps uniformly. |
| merger `window="hamming"` | Blend overlaps with tapered weights. |

Every chip has the requested spatial size. Merge weights have a positive
floor so covered border pixels remain represented. Dropped edges can leave NaN gaps.
Merging holds full output accumulators in memory and allocates the result
beside them. Lazy input reads do not make output memory bounded.
`chips.rasters` retains sources by reference; keep their grids unchanged.

**Next:** [training chips](../quickstart.md) for image/label pairs, or
[the sampling reference](../reference/chips.md) for exact arguments and return types.
