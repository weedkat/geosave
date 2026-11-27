# Prepare model inputs

A model author defines the bands, their order, physical units and treatment of
missing pixels in one preprocessing function. A training Dataset and an inference
caller can use the same function. This example uses the
[sample scene](../basics/first-raster.md#open-the-sample).

```python
import xarray as xr

import geosave as gs


def preprocess(sources: xr.DataTree) -> xr.DataTree:
    bands = sources["scene"].gs.to_dataset()[["red", "nir"]]
    image = bands.gs.mask_and_scale().astype("float32").fillna(0)
    return xr.DataTree.from_dict({"image": image})


scene = gs.open_raster("sample-scene.tif", chunks={})
sources = xr.DataTree.from_dict({"scene": scene})
prepared = preprocess(sources)
pixels = prepared.gs.to_tensor()
pixels["image"].shape
```

```text
torch.Size([2, 337, 267])
```

`image` has red in channel 0 and near-infrared in channel 1.
On a large area, select a spatial chip before calling `to_tensor`.
A different model can standardize bands with its training statistics or
keep an explicit validity layer. [Training chips](overlapping-chips.md) ·
[Preparing multiple collections](../contributing/design/release-inference.md)
