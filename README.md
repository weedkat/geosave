# GeoSave

GeoSave is an Alpha Python library for geospatial data preparation and
Lightning-based model development. Work with native xarray rasters and
GeoDataFrames, preserve their geospatial metadata, and prepare model inputs
without replacing the ecosystem's objects.

## Install

Requires Python 3.12 or newer:

```bash
pip install geosave
```

Add `geosave[ml]` for tensors and training, `geosave[hub]` for native model
releases, `geosave[flows]` for Prefect ingestion, or `geosave[cloud]` for S3
and Google Cloud filesystems. See the [installation guide](docs/install.md).

## Start with a raster

```python
import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave as gs

grid = GeoBox.from_bbox(
    (500000, 5000000, 500040, 5000040), crs="EPSG:32633", resolution=10
)
scene = xr.Dataset({
    "red": (("y", "x"), np.full((4, 4), 0.25, dtype="float32")),
    "nir": (("y", "x"), np.full((4, 4), 0.75, dtype="float32")),
}).gs.write_grid(grid)
ndvi = gs.features.ndvi(scene, nir="nir", red="red").rename("ndvi")
paths = ndvi.gs.to_cog("output", path_format="ndvi.tif")
```

Every NDVI pixel is `0.5`; the output retains the image's grid. The writer's
first argument is a directory, and `path_format` names the file. Existing files
raise unless replacement is explicitly requested.

Follow [your first raster](docs/first-raster.md) for a complete offline round
trip, then [images to training chips](docs/quickstart.md). File-backed rasters
can stay lazy until an operation needs their pixels.

## Train an editable starter

```bash
pip install 'geosave[ml]'
geosave create demo --task lightning --method boilerplate
cd demo
python main.py fit --config configs/lightning-boilerplate.yaml \
  --trainer.accelerator cpu --trainer.devices 1 --trainer.max_epochs 1
```

The starter supplies project-owned Dataset, DataModule, and LightningModule
files. Reusable builders, metrics, losses, transforms, and callbacks live in
GeoSave. Geospatial task recipes currently have a ModelSpec migration blocker;
see [training status](docs/guides/training.md#geospatial-recipes).

## Find the next step

- [Documentation](https://weedkat.github.io/geosave-engine/): guides and API reference.
- [Catalogs and samples](docs/guides/catalogs-and-cuts.md): publish imagery,
  group Item rows, and cut lazy raster trees into chips.
- [Training workspaces](docs/guides/workspaces.md): generate and customize code.
- [Model releases](docs/guides/release.md): save registered chains, weights,
  and optional preprocessing declarations.
- [Notebook examples](docs/examples.md): a land-cover project using real data;
  these require downloads and are separate from the offline starters.
- [Project review](docs/contributing/review-2026-11-04.md): current findings and
  prioritized implementation recommendations.

## Development

```bash
uv sync
uv run pytest                    # excludes slow and integration tests
uv run pytest -m integration     # external-service tests
uv run pytest -m slow
uv run ruff check .
uv run pre-commit run --all-files
uv run zensical build
```

Read [architecture](docs/contributing/architecture.md) and
[documentation maintenance](docs/contributing/documentation.md) before changing
public behavior or guides. Edit `src/templates/` to change future generated
workspaces; existing projects own their copied code.
