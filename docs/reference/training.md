# Training recipes

ModelSpec-based workflows on this page are paused pending redesign and raise
`NotImplementedError`. Use caller-owned data preparation and DataLoaders.

Lightning Modules, DataModules, Datasets, and sample definitions are generated
project code. They are no longer imported from `geosave.ml.tasks`.
Geospatial recipes are paused while ModelSpec is rebuilt.
Training Modules still accept their sample contracts from caller-owned DataLoaders.
See [workspaces](../guides/workspaces.md) for generation and
[training behavior](../concepts/training.md) for the recipes' initial contracts.

| Task | Methods | Generated package |
| --- | --- | --- |
| `segmentation` | `supervised`, `fixmatch` | `modules.segmentation.<method>` |
| `regression` | `supervised`, `mean-teacher` | `modules.regression.supervised` or `modules.regression.mean_teacher` |
| `classification` | `supervised` | `modules.classification.supervised` |
| `pretraining` | `mim` (experimental) | `modules.pretraining.mim` |
| `lightning` | `boilerplate` | `modules.lightning.boilerplate` |

Each package contains `module.py`, `data.py`, and `datasets.py`. Semi-supervised
recipes include their own supervised base. These implementations are editable;
training behavior does not need a library PR. New inference architectures still
need deployment support under the existing registered release contract.

Reusable metrics, builders, transforms, losses, and callbacks remain library
APIs; see [training tools](training-tools.md) and [metrics](metrics.md).
