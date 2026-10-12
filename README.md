# geosave-engine

GeoSave Engine is a local-first product for building geospatial AI workflows end to end. It standardizes the full path from data acquisition, environment setup, model training, and prediction to serving-ready outputs, so teams do not need to reinvent a different workflow for every project.

It generates editable workspace starters that compose geospatial data APIs, model stages, and Lightning training. Users can prepare datasets and develop models through ordinary Python objects and configuration.

Visit official Documentation : <https://weedkat.github.io/geosave-engine/>

## Features

- **Geospatial core.** CF-conformant `xr.Dataset` rasters and same-grid
  `xr.DataTree` stacks, plus `GeoDataFrame` vectors, use `.gs` accessors.
  Persistence covers Zarr, NetCDF, GeoTIFF/COG, GeoJSON, GeoPackage and
  GeoParquet; raster work stays lazy until you compute.
- **STAC catalogs.** Items are built from the files a writer saved and stored as
  stac-geoparquet item tables, searchable with rustac.
- **Cuts.** Frames and chips are tables of windows; each row reads itself as a
  lazy stack.
- **Model chains and releases.** Registered encoders (DINOv3, Prithvi,
  Prithvi-TL, Clay), decoders (DPT, UNet) and heads compose into one
  `ModelChain`, released as a Transformers model with its `model_spec.yaml`.
- **Lightning training.** Task and method pairs of `LightningModule` and
  `LightningDataModule` in `ml/`, run through `LightningCLI`.
- **MLflow.** Set `MLFLOW_TRACKING_URI` and training also logs to MLflow.
- **Editable workspaces.** `geosave create` hands you real files; no base class
  is required.

## Installations

Requires Python 3.12+.

```bash
pip install geosave-engine
# or
uv add geosave-engine
```

Want the rolling dev build (rebuilt on every push to `main`) instead of the
latest stable tag:

```bash
pip install --pre --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ geosave-engine
```

Working on GeoSave Engine itself (clone + `uv sync`), or installing an
exact dev build off a GitHub release — see the
[documentation](https://weedkat.github.io/geosave-engine/).

## Quick Start

```bash
uv run geosave create my-project
cd my-project
# fill in .env with your CDSE (or other STAC provider) credentials
```

`geosave create` prompts for a workspace starter; pass `-w segmentation`,
`-w regression`, `-w custom_lightning`, or `-w blank` to select it, and `-d` to set the description.
See the [workspace guide](docs/guides/workspaces.md) and follow the
[documentation](https://weedkat.github.io/geosave-engine/) for the full
step-by-step — explore a pipeline, build a dataset, train.

## Generated Workspace

```text
my-project/
├── artifacts/     # checkpoints, logs, saved configs (created by training)
├── configs/       # LightningCLI YAML configs
├── data/          # project datasets
├── logs/
├── modules/       # editable project modules
├── notebooks/
├── predictions/
├── scripts/       # `geosave make scripts <file>` copies optional scaffolds here
├── .env           # CDSE credentials, filled in with placeholders
├── geosave.toml   # project information, starter template, engine version
└── main.py        # LightningCLI entry point — do not need to touch this
```

## A first look in Python

```python
import numpy as np
import geosave_engine as gs

anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=64, resolution=10, crs="EPSG:32633"
)
pixels = np.random.default_rng(0).integers(1, 3000, (64, 64), dtype="uint16")
scene = gs.raster({"red": (("y", "x"), pixels)}, anchor.geobox, nodata=0)
scene
```

Output:

```text
<xarray.Dataset> Size: 9kB
Dimensions:      (y: 64, x: 64)
Coordinates:
  * y            (y) float64 512B 4.988e+06 4.988e+06 ... 4.987e+06 4.987e+06
  * x            (x) float64 512B 2.632e+05 2.632e+05 ... 2.639e+05 2.639e+05
    spatial_ref  int32 4B 32633
Data variables:
    red          (y, x) uint16 8kB 1528 2551 2763 1911 2521 ... 65 818 2788 735
```

The [quickstart](docs/quickstart.md) continues from pixels to model-ready
tensors, offline.

## Development

```bash
uv sync
uv run pytest                    # fast tests
uv run pytest -m integration     # network tests
uv run pytest -m slow
uv run ruff check .
uv run pre-commit run --all-files
uv run zensical build            # documentation, written to site/
```
