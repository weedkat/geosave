# Installation

GeoSave requires Python 3.12 or newer.

```bash
pip install geosave-engine
```

For a project managed with uv:

```bash
uv add geosave-engine
```

The package includes its geospatial and Lightning dependencies. You can use the
data APIs on their own; importing GeoSave does not import torch.

## Model releases

Install the Hub extra when saving or loading native Transformers releases:

```bash
pip install 'geosave-engine[hub]'
```

Individual pretrained encoders may download weights on first use. The
[First raster](first-raster.md) and [training example](guides/training.md) work
offline once their dependencies are installed.

## Development installation

From a clone of this repository:

```bash
uv sync
```

This installs the project and its development tools. `uv run pytest` runs the
local test suite; `uv run zensical build` builds these docs.
