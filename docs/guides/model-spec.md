# Declare and prepare model inputs

## Describe the raster a model reads

Save this as `model_spec.yaml`:

```yaml
schema_version: 2
rasters:
  optical:
    variables: [red, nir]
    coordinates: [x, y]
    require_crs: true
preprocessing:
  image:
    call: !ref optical.gs.mask_and_scale
inputs:
  image: !ref image
```

The model needs red and near-infrared bands, in that order, on a georeferenced
grid. `optical` names the supplied raster. Preprocessing decodes its stored
nodata, scale, and offset into physical values and names the result `image`.
`inputs.image` binds that result to the model argument `image`.

```python
from geosave_engine.model.spec import ModelSpec

spec = ModelSpec.load("model_spec.yaml")
```

Loading validates the declaration without importing or running its calls. The
spec describes data preparation; model architecture is configured separately
through [model stages](models.md).

## Prepare a raster in memory

Here is a small raster with packed reflectance values. The same `preprocess`
call works with a Dataset opened from GeoTIFF, Zarr, or STAC.

```python
import numpy as np
from odc.geo.geobox import GeoBox

import geosave_engine as gs

grid = GeoBox.from_bbox(
    (500_000, 9_000_000, 500_040, 9_000_040),
    "EPSG:32748",
    resolution=10,
)
scene = gs.raster({
    "red": (("y", "x"), np.full((4, 4), 1000, dtype="uint16")),
    "nir": (("y", "x"), np.full((4, 4), 3000, dtype="uint16")),
}, grid)
for name in scene.data_vars:
    scene[name].attrs["scale_factor"] = 0.0001

prepared = spec.preprocess({"optical": scene})
prepared["image"]
```

```text
<xarray.Dataset> Size: 196B
Dimensions:      (y: 4, x: 4)
Coordinates:
  * y            (y) float64 32B 9e+06 9e+06 9e+06 9e+06
  * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
    spatial_ref  int32 4B 32748
Data variables:
    red          (y, x) float32 64B ...
    nir          (y, x) float32 64B ...
```

Red is now 0.1 and near-infrared is 0.3. The coordinates and CRS remain with
the Dataset. Selection and this preprocessing preserve Dask laziness when the
source is lazy. A declared preprocessing operation can still require eager
work; choose its behavior explicitly.

## Turn prepared pixels into a model tensor

```python
inputs = spec.model_inputs(prepared)
image = inputs["image"].gs.to_tensor(dtype="float32").unsqueeze(0)
image.shape
```

```text
torch.Size([1, 2, 4, 4])
```

`model_inputs` binds names to prepared rasters and adds declared context.
Raster inputs remain xarray objects until you call `to_tensor`. This call
reads the pixels, stacks variables in their selected order, and the
`unsqueeze(0)` adds one batch dimension. With time as its only nonspatial axis,
a raster becomes `(time, band, y, x)` before batching. Other nonspatial axes
retain their order; the model must support that layout.

Feed `image` into the two-band chain from [Build and use a model](models.md):
`chain(image=image)`. For a DataLoader, your dataset performs the raster-to-tensor
conversion per sample and batching adds the batch dimension. See
[Training](training.md) for the dataset and DataModule contracts.

## Save prepared inputs for repeated training

`prepare` reads windows from a raw item table and writes the preprocessed model
inputs. Labels stay beside them. To try it with the raster above, first create
a matching label and a raw catalog:

```python
from datetime import UTC, datetime

label = gs.raster({
    "label": (("y", "x"), np.zeros((4, 4), dtype="uint8")),
}, grid)
files = gs.stack({"optical": scene, "label": label}).gs.to_cog(
    "data/raw/{collection}.tif"
)
raw = gs.stac.create_items(
    files, id="sample", datetime=datetime(2025, 6, 1, tzinfo=UTC)
)

catalog_path = spec.prepare(raw, "data/prepared", on="label")
catalog = gs.open_vector(catalog_path)
```

`on="label"` uses each label Item as a window. For each window, `prepare`
selects the declared rasters, runs preprocessing, and saves the raster roots
read by `inputs`. Here those are `image` and the automatically retained label.
It writes COG assets, `catalog.parquet`, and `model_spec.yaml` into a new output
folder. Choose a fresh path on subsequent runs; an existing output folder
raises `FileExistsError`.

```python
windows = gs.cuts.stacks(catalog, on="label")
sample = windows.iloc[0].gs.to_stack()
inputs = spec.model_inputs(sample.gs.rasters)
image = inputs["image"].gs.to_tensor(dtype="float32").unsqueeze(0)
sample.close()
```

The prepared catalog holds physical values, so bind its inputs directly.
`prepare` already ran preprocessing; do not rerun it on the prepared rasters.
`keep=["another_collection"]` copies additional collections through unchanged.

Temporal frames and spatial chips are separate cuts after preparation. Add a
`chips` section for a training DataModule, then cut windows with
`spec.chips.cut(windows)`. [Catalogs and cuts](catalogs-and-cuts.md) explains
window tables, lazy reads, and frame/chip construction.

Continue with [model input contracts](../concepts/model-inputs.md) for optional
metadata requirements, STAC acquisition, tensor normalization, and geospatial
context. The [spec reference](../reference/model-spec.md) lists the public
fields and methods.
