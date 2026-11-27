# GeoSave

**From satellite imagery to geospatial machine learning, in ordinary Python.**

GeoSave connects imagery discovery, raster preparation, interactive maps, and
Lightning-based model development. Work with xarray Datasets and GeoDataFrames
while keeping the coordinates, CRS, nodata, and band metadata needed to turn
results back into usable maps.

[Documentation](https://weedkat.github.io/geosave/) ·
[Examples](https://github.com/weedkat/geosave/tree/main/examples) ·
[Installation](https://weedkat.github.io/geosave/basics/install/)

![Sentinel-2 imagery and vegetation index near Davis, California](https://raw.githubusercontent.com/weedkat/geosave/main/docs/assets/vegetation.jpg)

## What can I do with it?

- **Find and download imagery.** Search STAC collections and load the bands,
  dates, area, and pixel grid you need.
- **Explore before processing.** Inspect rasters, vectors, and catalogs on
  interactive maps or create static plots.
- **Prepare useful datasets.** Align inputs explicitly, calculate spectral
  indices, preserve nodata, and save GeoTIFF, Zarr, NetCDF, or GeoParquet.
- **Use pretrained modules.** Build models with supported encoders,
  decoders, and heads, including geospatial foundation-model integrations.
- **Train and map predictions.** Make chips from aligned xarray Datasets,
  train with Lightning, and merge chip predictions onto their source grids.

GeoSave is **Alpha**: APIs are evolving. These examples track `main`.

## Install

Python **3.12+**. Choose a source:

**Published releases · [PyPI](https://pypi.org/project/geosave/)** — install an
available release with `python -m pip install geosave`. These Alpha docs track
`main`; use the source installation below for the closest API match.

**Test build · [TestPyPI](https://test.pypi.org/project/geosave/)**

```bash
python -m pip download --no-deps --index-url https://test.pypi.org/simple/ --dest wheels "geosave==0.3.0.dev38"
python -m pip install --upgrade ./wheels/geosave-0.3.0.dev38-py3-none-any.whl
```

This downloads only GeoSave from TestPyPI; dependencies use your normal index.
For training, add `[ml]` to the quoted wheel path. Use `[all]` for the notebooks.
Replace the version in both commands when choosing a newer test build.

**Latest source · GitHub** — requires Git:

```bash
python -m pip install --upgrade "geosave @ git+https://github.com/weedkat/geosave.git@main"
```

Use `geosave[ml] @` for training or `geosave[all] @` for every extra.
Replace `main` with a commit SHA to pin an experiment. See the
[installation guide](https://weedkat.github.io/geosave/basics/install/) for
extras, development setup, and version checks.

## Make a vegetation map

Download a small real Sentinel-2 crop, calculate NDVI, and save a georeferenced
result. No account or GPU is needed for this example.

```python
from pathlib import Path
from urllib.request import urlretrieve

import geosave as gs
from geosave.geodata import io

urlretrieve(
    "https://raw.githubusercontent.com/weedkat/geosave/main/docs/assets/sample-scene.tif",
    "scene.tif",
)
Path("output").mkdir(exist_ok=True)
scene = gs.open_raster("scene.tif")
ndvi = gs.features.ndvi(scene, nir="nir", red="red").rename("ndvi")
io.write_cog(ndvi.to_dataset(), "output/ndvi.tif")
```

Open `output/ndvi.tif` in your GIS. The sample is a 752 KiB crop of agricultural
land near Davis, California. `ndvi` decodes the bands' nodata and scale before
calculating. An index shows spectral variation; one scene does not establish
crop health, drought, or change.

Follow the [vegetation tutorial](https://weedkat.github.io/geosave/basics/first-raster/)
for the map and interpretation, or the
[zone comparison](https://weedkat.github.io/geosave/advanced/vegetation-zones/)
to summarize NDVI and valid coverage for two areas in a CSV.

## Follow a land-cover project

The notebooks use expert-annotated Sentinel-2 scenes to train a baseline and
fine-tune Prithvi on image chips. Save the trained model and its class legend,
then load them in a GeoSave pipeline to publish a dated forest map. The
teaching download is about 840 MB. The Jakarta NDVI lesson runs independently
and needs only a small satellite crop. [Start the notebook series](examples/README.md).

| Your next task | Guide |
| --- | --- |
| Download imagery for your area | [Search and download](https://weedkat.github.io/geosave/advanced/download-imagery/) |
| Explore imagery on a map | [Explorer](https://weedkat.github.io/geosave/basics/explore/) |
| Start with pretrained weights | [Pretrained models](https://weedkat.github.io/geosave/advanced/pretrained/) |
| Create an editable training project | [Workspaces](https://weedkat.github.io/geosave/advanced/training/) |
| Organize data, experiments and releases | [Three-repository workflow](https://weedkat.github.io/geosave/mlops/) |
| Look up a function | [API reference](https://weedkat.github.io/geosave/reference/) |

## Development

```bash
git clone https://github.com/weedkat/geosave.git
cd geosave
uv sync
uv run pytest
uv run zensical build
```

See [contributing](https://weedkat.github.io/geosave/contributing/architecture/)
for architecture and contracts. Training workspaces own their Dataset,
DataModule, and LightningModule; edit `src/templates/` to change future
workspace generation.
