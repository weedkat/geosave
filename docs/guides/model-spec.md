# Declare and prepare model inputs

ModelSpec is under active development in Alpha. This guide describes the
current implementation; its API and preparation pattern may change. To learn
model development with plain Python preparation, start with the
[notebooks](../examples.md).

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
from geosave.model.spec import ModelSpec

spec = ModelSpec.load("model_spec.yaml")
```

Loading validates the declaration without importing or running its calls. The
spec describes data preparation; model architecture is configured separately
through [model stages](models.md).

`rasters` checks and selects the supplied data; it does not rename bands or
reproject them to meet a requirement. `preprocessing` holds the explicit calls
that change pixels. `inputs` maps their results to the chain's argument names.
Only `schema_version` and `rasters` are required to load a spec; this example
also declares preparation and a model input so it can supply a model call.

## Prepare a raster in memory

Here is a small raster with packed reflectance values. The same `preprocess`
call works with a Dataset opened from GeoTIFF, Zarr, or STAC.

```python
import numpy as np
from odc.geo.geobox import GeoBox

import geosave as gs

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

The stages run at different times:

| Call | Takes | Returns or applies |
| --- | --- | --- |
| `spec.preprocess(rasters)` | Named raw xarray Datasets | Selected rasters and explicit preprocessing results |
| `spec.model_inputs(prepared)` | Named prepared rasters | Pixel inputs as xarray objects, plus declared context tensors |
| `raster.gs.to_tensor()` | One selected raster | Pixel tensor; this reads lazy data |
| The DataModule's batch-transfer hook | A collated tensor batch | Training augmentation, then `spec.transforms` in every split |

`model_inputs` binds prepared values without rerunning preprocessing or raster
validation. `preprocess` and `model_inputs` also leave `transforms` for the
DataModule or your own inference code to apply. A direct model call needs that
same tensor normalization. [Pretrained weights](pretrained.md) shows an image
input with time/location context and explicit inference normalization.

## Save prepared inputs for repeated training

`prepare` reads windows from a raw item table and writes the preprocessed model
inputs. Labels stay beside them. To try it with the raster above, first create
a matching label and a raw catalog:

```python
from datetime import UTC, datetime

label = gs.raster({
    "label": (("y", "x"), np.zeros((4, 4), dtype="uint8")),
}, grid)
scenes = gs.tree({"optical": scene, "label": label}).gs.to_cog(
    "data/raw/{collection}/{band}.tif"
)
raw = gs.stac.create_catalog(
    gs.stac.create_items(scenes, id_format="sample", datetime=datetime(2025, 6, 1, tzinfo=UTC))
)

catalog_path = spec.prepare(raw, "data/prepared", keep=["label"])
catalog = gs.open_vector(catalog_path)
```

`{band}` creates a file per variable. Each single-band STAC asset is named
after that variable, so catalog reads keep `red` and `nir` as the spec expects.

`prepare` groups equal-footprint Items and applies the declared frames. For each sample, it
selects the declared rasters, runs preprocessing, and saves the raster roots
read by `inputs`. Here those are `image` and the explicitly retained label.
It writes COG assets, `catalog.parquet`, and `model_spec.yaml` into a new output
folder. Choose a fresh path on subsequent runs; an existing output folder
raises `FileExistsError`.

```python
windows = catalog.gs.sample()
sample = windows.iloc[0].gs.to_xarray()
inputs = spec.model_inputs(sample.gs.rasters)
image = inputs["image"].gs.to_tensor(dtype="float32").unsqueeze(0)
sample.close()
```

The prepared catalog holds physical values, so bind its inputs directly.
`prepare` already ran preprocessing; do not rerun it on the prepared rasters.
`keep=["another_collection"]` copies additional collections through unchanged.

Temporal frames and spatial chips are separate cuts after preparation. Add a
`chips` section for a training DataModule, then cut windows with
`spec.chips.cut(windows, target=grid)`. [Catalogs and cuts](catalogs-and-cuts.md) explains
sample tables, lazy reads, and frame/chip construction.

Continue with [model input contracts](../concepts/model-inputs.md) for optional
metadata requirements, STAC acquisition, tensor normalization, and geospatial
context. The [spec reference](../reference/model-spec.md) lists the public
fields and methods.
