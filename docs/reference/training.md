# Training recipes

Lightning Modules, DataModules, Datasets, and sample definitions are generated
project code. They are no longer imported from `geosave.ml.tasks`.
DataModules build explicit preprocessors and take caller-owned Datasets.
See [workspaces](../advanced/training.md) for generation and
[training behavior](../concepts/training.md) for the recipes' initial contracts.

| Task / method | Generated package | Current starting point |
| --- | --- | --- |
| `lightning` / `boilerplate` | `modules.boilerplate` | Offline synthetic Dataset and training configuration. |
| `segmentation` / `supervised` | `modules.supervised` | Paired raster Dataset; its flat samples still conflict with Module steps expecting nested `inputs`. Integration is unfinished. |
| `segmentation` / `fixmatch` | `modules.fixmatch` | Implement labeled and unlabeled Datasets; supplied examples raise `NotImplementedError`. |
| `regression` / `supervised`, `mean-teacher` | `modules.supervised`, `modules.mean_teacher` | Implement the supplied Dataset placeholders. |
| `classification` / `supervised` | `modules.supervised` | Implement the supplied Dataset placeholder. |
| `pretraining` / `mim` | `modules.mim` | Experimental reconstruction method with a Dataset placeholder. |

Each package contains `module.py`, `data.py`, and `datasets.py`. Semi-supervised
recipes include their own supervised base. These implementations are editable;
training behavior does not need a library PR. New inference architectures still
need deployment support under the existing registered release contract.

The [training guide](../advanced/training.md) runs the Lightning boilerplate.
ModelSpec remains unavailable; it is no longer the dependency blocking the
template DataModules.

Reusable metrics, builders, transforms, losses, and callbacks remain library
APIs; see [training tools](training-tools.md) and [metrics](metrics.md).
