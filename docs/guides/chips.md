# Read overlapping chips and merge predictions

Split the [sample scene](../first-raster.md#open-the-sample) into windows,
process each one, and place the merged result back on its original grid.
No catalog, model, or GPU is needed.

**You need:** the base package and the sample TIFF. The calculation below
copies one band through the tiling workflow so you can check reconstruction
before substituting a model.

## Read one window

```python
import numpy as np
import xarray as xr
from odc.geo.xr import xr_coords

import geosave as gs

source = gs.open_raster("sample-scene.tif")
image = source[["red", "nir"]]
chips = gs.Chips([image], size=128, overlap=32, mode="reflect")
chip = chips[0]
pixels = chip.gs.to_numpy(dtype="float32")
pixels.shape, chips.key(0)
```

```text
((2, 128, 128), (0, 0))
```

Construction plans windows without reading pixels. Indexing returns a lazy
Dataset on that window's grid; `to_numpy` reads its pixels. Storage blocks or
upstream calculations can require reads beyond the requested window.
Keep `source` open until all chip reads finish.

`chips[index]` indexes across all source images. `chips.key(index)` returns
`(sample_index, tile_id)`; tuple indexing and iteration use those same keys.
Select bands explicitly to fix channel order. Tensor conversion does not
decode scaling; use `.gs.mask_and_scale()` first when the model needs physical values.

## Merge a result

```python
def predictions():
    for key in chips:
        values = chips[key].gs.to_numpy(dtype="float32")
        yield key, values[0]  # Stand-in prediction: copy the red band.


outputs = chips.merge(predictions(), window="hamming")
red = xr.DataArray(outputs[0], dims=("y", "x"), name="red_copy")
red = red.assign_coords(xr_coords(image.gs.geobox))
np.allclose(red.values, image["red"].values)
```

```text
True
```

A weighted float32 reconstruction can differ from its input by rounding.
A model can replace `values[0]`. Return NumPy predictions with spatial axes
last, together with the original chip key. A weighted merge blends overlaps.
For class logits, merge scores first and choose the class afterward; averaging
integer class IDs has no useful class meaning.

Outputs are float32 NumPy arrays in source order, or None for a source with
no predictions. Choose dimensions appropriate for your output, such as
`("class", "y", "x")` for class scores. Give predictions their own name, units,
class legend and nodata policy; source band metadata may not describe them.

```python
path = red.gs.to_cog("output/red-copy.tif", overwrite=True)
source.close()
```

## Choose overlap and edge behavior

| Option | Behavior |
| --- | --- |
| `overlap=32` | Share 32 pixels between adjacent chips. |
| `overlap=0.25` | Share one quarter of chip size, rounded up to pixels. |
| `mode="constant", constant_value=0` | Pad boundary chips with an explicit value, retaining dtype. |
| `mode="reflect"`, `"edge"`, or `"wrap"` | Extend values from the boundary tile using that padding rule. |
| `mode="drop"` | Omit incomplete boundary chips. |
| `window=None` | Merge with uniform weighting. |
| `window="hamming"` | Blend overlapping predictions with a Tiler window. |

Every returned chip has the requested spatial size. Missing tiles are allowed;
uncovered output pixels become zero. Dropped edges or windows with zero edge
weights can therefore leave gaps. Repeated keys accumulate contributions.

Merging allocates full output accumulators in memory. Lazy input reads do not
make output memory bounded. `chips.xarray[i]` retains each source by reference;
do not change its grid after constructing Chips.

**Next:** [training chips](../quickstart.md) for image/label pairs, or
[the Chips reference](../reference/chips.md) for exact arguments and return types.
