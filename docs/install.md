# Installation

GeoSave requires Python 3.12 or newer.

```bash
pip install geosave
```

For a project managed with uv:

```bash
uv add geosave
```

`pip install geosave` covers opening, inspecting, plotting, STAC search and
loading, writing, model specs, and the `geosave` CLI. Add extras for the rest:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `ml` | torch, Lightning, Kornia, timm, TerraTorch, torchmetrics, TensorBoard | Model stages, training tasks, `.gs.to_tensor()` |
| `hub` | `ml` + Transformers | Saving and loading native model releases |
| `flows` | Prefect | `geosave_flows` and `geosave-flows ingest` |
| `cloud` | s3fs, gcsfs | `s3://` and `gs://` paths |
| `mlflow` | MLflow | The MLflow logger added when `MLFLOW_TRACKING_URI` is set |
| `all` | every extra above | Notebooks and the examples series |

```bash
pip install "geosave[all]"
```

Without an extra, the feature that needs it fails with Python's own
`ModuleNotFoundError` naming the missing package, such as `torch` or `prefect`.
Importing GeoSave never imports torch.

## Model releases

Save and load native Transformers releases with the `hub` extra, which includes `ml`:

```bash
pip install "geosave[hub]"
```

Individual pretrained encoders may download weights on first use. The
[First raster](first-raster.md) and [training example](guides/training.md) work
offline once their dependencies are installed.

## Development installation

From a clone of this repository:

```bash
uv sync
```

This installs the project with every extra (the `dev` group pulls
`geosave[all]`) and its development tools. `uv run pytest` runs the
local test suite; `uv run zensical build` builds these docs.
