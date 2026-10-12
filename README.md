# geosave

GeoSave is a Python library for geospatial data preparation and Lightning-based
model development. Work with native xarray rasters and GeoDataFrames, preserve
geospatial metadata, prepare training chips, and compose ordinary PyTorch models.

Use the library directly, or create an editable workspace with run configurations
and your own Dataset. Install `geosave[flows]` to use the optional Prefect ingestion jobs through
`geosave_flows`.

Visit official Documentation : <https://weedkat.github.io/geosave-engine/>

## Features

- **Geospatial core.** CF-conformant `xr.Dataset` rasters and named
  `xr.DataTree` groups that can retain their native grids, plus `GeoDataFrame` vectors, use `.gs` accessors.
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
pip install geosave
# or
uv add geosave
```

Training, Hub releases, Prefect flows and cloud paths are extras;
`pip install "geosave[all]"` installs everything the examples use.

Want the rolling dev build (rebuilt on every push to `main`) instead of the
latest stable tag:

```bash
pip install --pre --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ geosave
```

Working on GeoSave itself (clone + `uv sync`), or installing an
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
Start with [Your first raster](docs/first-raster.md), then
[prepare training chips](docs/quickstart.md) and [train a model](docs/guides/training.md).
The [workspace guide](docs/guides/workspaces.md) explains generated projects;
the [API reference](docs/reference/index.md) lists public functions and classes.

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
import geosave as gs

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
