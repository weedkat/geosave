# Installation

GeoSave requires Python 3.12 or newer. Run installation commands in your
project's virtual environment; in a notebook, select that environment as the
kernel. Start with the base package for raster and vector processing.

## Install to follow these docs

These Alpha docs follow the repository's `main` branch. Use the source install
for the closest match to the examples. It requires Git as well as Python:

```bash
python -m pip install --upgrade "geosave @ git+https://github.com/weedkat/geosave.git@main"
```

For tensors, model stages and the CPU training tutorial, use the complete ML command:

```bash
python -m pip install --upgrade "geosave[ml] @ git+https://github.com/weedkat/geosave.git@main"
```

Choose `geosave[hub]` for native model releases or `geosave[all]` for every
optional dependency. To reproduce a project, replace `main` with its commit
SHA and record the environment. An earlier packaged preview may have different APIs.

## Other installation sources

### Test build · TestPyPI

Download GeoSave from [TestPyPI](https://test.pypi.org/project/geosave/), then
install its dependencies from your normal index (usually PyPI):

```bash
python -m pip download --no-deps --index-url https://test.pypi.org/simple/ --dest wheels "geosave==0.3.0.dev38"
python -m pip install --upgrade ./wheels/geosave-0.3.0.dev38-py3-none-any.whl
```

`0.3.0.dev38` is the build checked for this guide. Replace the version in **both**
commands to try a newer build listed on TestPyPI. The separate download keeps
TestPyPI's unrelated test packages out of dependency resolution.

For the real-image training examples, install the same wheel with extras:

```bash
python -m pip install --upgrade "./wheels/geosave-0.3.0.dev38-py3-none-any.whl[all]"
```

### Published releases · PyPI

When a release is available on [PyPI](https://pypi.org/project/geosave/), install
it with `python -m pip install geosave`. Check its version against the example
you are following; these docs describe source on `main`.

## Choose what you need

The base package covers opening, inspecting, plotting, STAC search and
loading, writing, and the `geosave` CLI. ModelSpec is currently disabled. Add extras for the rest:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `ml` | torch, Lightning, Kornia, timm, TerraTorch, torchmetrics, TensorBoard | Model stages, training tasks, `.gs.to_tensor()` |
| `hub` | `ml` + Transformers | Saving and loading native model releases |
| `flows` | Prefect | Project Prefect flows and `geosave_flows` tasks |
| `cloud` | s3fs, gcsfs | `s3://` and `gs://` paths |
| `mlflow` | MLflow | The MLflow logger added when `MLFLOW_TRACKING_URI` is set |
| `all` | every extra above | Notebooks and the examples series |

Request extras on your chosen installation source as shown above.

Without an extra, the feature that needs it fails with Python's own
`ModuleNotFoundError` naming the missing package, such as `torch` or `prefect`.
Importing GeoSave never imports torch.

## Model releases

Save and load native Transformers releases with the `hub` extra, which includes
`ml`. Use `geosave[hub]` on the source install command or `[hub]` on the wheel.

Individual pretrained encoders may download weights on first use. The
[first raster](first-raster.md) works offline after downloading its small sample;
the [training tutorial](guides/training.md) creates its data locally and needs
no model download.

## Development installation

From a clone of this repository:

```bash
uv sync
```

This installs the project with every extra (the `dev` group pulls
`geosave[all]`) and its development tools. `uv run pytest` runs the
local test suite; `uv run zensical build` builds these docs.

## Check the installation

```bash
python -c "import geosave; from importlib.metadata import version; print(version('geosave'))"
```

Then follow [your first vegetation map](first-raster.md), or use
[the offline chip tutorial](quickstart.md) if you installed the `ml` extra
and want to begin with training data. If an example does not match your
installed API, see [version troubleshooting](troubleshooting.md#installation-and-versions).
