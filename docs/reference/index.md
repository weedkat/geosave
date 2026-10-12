# API reference

Look up a public function, class, or accessor method here. Entries are generated
from the implementation's docstrings and include signatures, parameters,
returns, and examples where available.

| Area | APIs |
| --- | --- |
| [Data](geodata.md) | Raster and vector accessors, I/O, transforms, features, metadata, STAC, and windows. |
| [Models](model.md) | Chains, registry, stages, input specifications, and releases. |
| [Training](ml.md) | Lightning tasks, data modules, metrics, and training tools. |
| [Workflows](workflow.md) | Validated job inputs, Prefect flows, and reusable tasks. |

Start with the [practical guides](../index.md#find-your-task) when you want a
complete example. Read [Concepts](../concepts/index.md) for behavior shared
by several operations.

## Accessors and native objects

For `scene.gs.reproject(...)`, look under the [Dataset accessor](raster-accessor.md).
For a single band, use the [DataArray accessor](array-accessor.md); for a group
of rasters, use the [DataTree accessor](stack-accessor.md). Their inherited
GeoSave methods are included. [GeoDataFrame methods](vectors.md) are separate.

The reference documents GeoSave's own APIs. Native xarray, GeoPandas, PyTorch,
Lightning, and Transformers methods keep their upstream contracts.
