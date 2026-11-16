# How the library works

```python
from pathlib import Path

import geosave as gs
from geosave.geodata import io

scene = gs.open_raster("sample-scene.tif")
image = scene[["red", "nir"]]
ndvi = gs.features.ndvi(image, nir="nir", red="red").rename("ndvi")

Path("output").mkdir(parents=True, exist_ok=True)
io.write_cog(ndvi.to_dataset(), "output/ndvi.tif")
```

With the [sample scene](../basics/first-raster.md), this is a complete workflow:
open, select, calculate, save. `scene` and `image` are native xarray Datasets;
`ndvi` is a DataArray. Importing GeoSave registers `.gs` accessors on native
xarray and GeoPandas objects.

## Choose the object for the job

| What you have | Python object | Purpose |
| --- | --- | --- |
| One band or array | DataArray | Values with named axes, coordinates and attrs. |
| Bands on a shared grid | Dataset | Named variables on shared coordinates. |
| Several observations or grids | DataTree | Datasets in a caller-defined hierarchy; each node can have its own grid. |
| Features or STAC Items | GeoDataFrame | Geometry and attributes; Item assets point to files. |

`gs.Dataset`, `gs.DataArray`, and `gs.DataTree` are the native xarray types.
Normal indexing, arithmetic, concatenation and merging remain available.
Decoding values, changing grids, converting dtype and computing pixels should
be explicit. See [metadata](metadata.md) before combining packed imagery.

## Add only the stages your workflow needs

| Need | Operation | Result |
| --- | --- | --- |
| Describe a place and time | `GeoRegion` | Geometry, CRS and optional period, without a pixel grid. |
| Discover observations | `StacClient.search` | Item table, without raster pixels. |
| Load on an exact grid | `stac.load_dataset(items, geobox=grid)` | Dataset on the supplied GeoBox. |
| Retain native asset grids | `stac.load_native_datasets(items)` | One `(crs, resolution, Dataset)` per native grid. |
| Read model-sized windows | `RasterChipper([tree], size=256)` | Lazy raster windows; explicit conversion reads pixels. |
| Publish dated raster assets | `scene.gs.to_catalog(folder)` | Written TIFFs and an Item table. |

Catalog sampling is currently unavailable. Select observations with GeoPandas
or select dates on a prepared Dataset. [Catalogs](../advanced/raster-catalog.md)
and [chips](../advanced/overlapping-chips.md) work independently.

Lazy execution delays pixel work; it does not guarantee small downloads or bounded peak memory.
See [lazy execution](lazy-data.md) and [remote storage](storage.md#use-remote-storage).

## Add training when you need a model

```mermaid
flowchart LR
    rasters[Prepared rasters] --> samples[Project Dataset: inputs and targets]
    samples --> batches[DataLoader: batches]
    batches --> training[LightningModule + Trainer]
    training --> checkpoint[Training checkpoint]
    training --> release[Model release + pipeline]
    release --> product[Prediction raster on the source grid]
```

Your Dataset owns reads, band order, target codes and split selection. RasterChipper
is optional; it cuts prepared one-grid DataTrees. A DataModule can organize loaders and device-side transforms;
the LightningModule owns loss and metrics, and Trainer runs the loop.
Start with the [Lightning training guide](../advanced/training.md).

Prepare inputs with the same code at training and inference. A native model
release saves registered stage construction and weights, and optionally a
pipeline: ingest, preparation, prediction and decoding classes that run the
model chip by chip over a raster. See [model inputs](../advanced/model-inputs.md)
and [releases](../advanced/release.md#run-with-a-raster-pipeline).
ModelSpec is unavailable. Generated recipes now use explicit preprocessors;
their Dataset and integration readiness varies. See
[workspace method status](../reference/training.md).

For a real project, version [data, experiments and model releases](../mlops/index.md)
separately and link their immutable revisions.

Package ownership belongs in [the contributor architecture](../contributing/architecture.md).
