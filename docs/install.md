# Installation

GeoSave requires Python 3.12 or newer. Run installation commands in your
project's virtual environment; in a notebook, select that environment as the
kernel. Start with the base package for raster and vector processing.

## Choose an installation source

GeoSave is in Alpha. These docs follow the repository's `main` branch.
Choose **TestPyPI** for a packaged preview or **GitHub** for the latest source.

### Stable release · PyPI

The release command is:

```bash
python -m pip install geosave
```

A public PyPI release was not available when this guide was checked on
6 November 2026. Until one is published, use one of the preview paths below.
See [PyPI](https://pypi.org/project/geosave/) for release availability.

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

### Latest source · GitHub

Requires Git in addition to Python:

```bash
python -m pip install --upgrade "geosave @ git+https://github.com/weedkat/geosave.git@main"
```

For training, replace `geosave @` with `geosave[ml] @`; for all notebook
dependencies, use `geosave[all] @`. To reproduce an experiment, replace `main`
with the commit SHA you used. Test builds and `main` may have different APIs;
use GitHub when following a newly updated example.

## Choose what you need

The base package covers opening, inspecting, plotting, STAC search and
loading, writing, model specs, and the `geosave` CLI. Add extras for the rest:

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

Save and load native Transformers releases with the `hub` extra, which includes `ml`:

Use `[hub]` on your chosen wheel or GitHub install command.

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
