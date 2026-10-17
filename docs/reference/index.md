# API reference

Look up a public function, class, or accessor method here. Entries are generated
from the implementation's docstrings and include signatures, parameters,
returns, and examples where available.

The [complete public API index](public-api.md) also includes module helpers,
configuration fields, and inherited GeoSave methods. Its contracts are refreshed
from the current source with `uv run python scripts/api_reference.py`.

| Area | APIs |
| --- | --- |
| [Data](geodata.md) | Raster and vector accessors, I/O, transforms, features, metadata, STAC, and windows. |
| [Models](model.md) | Chains, registry, stages, input specifications, and releases. |
| [Training](ml.md) | Lightning tasks, data modules, metrics, and training tools. |
| [Workflows](workflow.md) | `geosave_flows`: validated job inputs, Prefect flows, and reusable tasks. |
| [CLI](cli.md) | Workspace commands, options, and generated files. |

## Inputs and outputs

Each entry shows its signature with parameter names, annotated types, and
defaults. Read the parameter and return descriptions for accepted values,
units, array axes, band order, and whether the operation changes or computes
its input. A tensor annotation alone does not describe its shape.

For validated Pydantic models, the listed fields are keyword inputs.
TypedDict fields describe dictionary keys. Field entries show their types and
declared defaults; a required field has no default. `Field(...)` expressions
also show validation constraints and default factories.

Readers and writers can forward `**options` to their format backend. Follow
the corresponding [format options](io.md) for the supported keys and their
defaults. An output annotated `None` has no returned value; read its description
for file writes or changes made to the input.

The complete contracts are grouped by [data](generated/data.md),
[models](generated/models.md), [training](generated/training.md), and
[flows](generated/flows.md). A missing type or description is a source
documentation gap. [Documentation maintenance](../contributing/documentation.md)
explains how to refresh the reference and audit those gaps.

Start with the [practical guides](../index.md#find-your-task) when you want a
complete example. Read [Concepts](../concepts/index.md) for behavior shared
by several operations.

## Accessors and native objects

For `scene.gs.reproject(...)`, look under the [Dataset accessor](dataset-accessor.md).
For a single band, use the [DataArray accessor](dataarray-accessor.md); for a group
of rasters, use the [DataTree accessor](datatree-accessor.md). Their inherited
GeoSave methods are included. [GeoDataFrame methods](vectors.md) are separate.

The reference documents GeoSave's own APIs. Native xarray, GeoPandas, PyTorch,
Lightning, and Transformers methods keep their upstream contracts.
