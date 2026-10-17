# Learn by example

```bash
git clone https://github.com/weedkat/geosave.git
cd geosave
uv sync
```

Open a notebook in `examples/`. Select the repository's `.venv` Python as the
kernel and use `examples/` as the working directory. Run cells in order.
Each notebook starts with a fresh kernel and reuses files from earlier lessons.

## Follow the project

Build a land-cover map from satellite images using ordinary PyTorch Datasets
and Lightning modules.

| Notebook | What you do |
| --- | --- |
| [0 · Basics](https://github.com/weedkat/geosave/blob/main/examples/00_basics.ipynb) | Explore images and labels, train a small model, and plot a prediction. |
| [1 · Model](https://github.com/weedkat/geosave/blob/main/examples/01_model.ipynb) | Swap model parts and compare training runs. |
| [2 · Build a dataset](https://github.com/weedkat/geosave/blob/main/examples/02_sources.ipynb) | Fetch images for labels and a forest outline, then save the files and catalogs. |
| [3 · Train and predict](https://github.com/weedkat/geosave/blob/main/examples/03_inputs.ipynb) | Select time windows, make chips, train, and join predictions into maps. |

The [teaching dataset](https://huggingface.co/datasets/fatmur/dynamicworld-300)
is about 400 MB. Later notebooks reuse that download. Notebook 2 also needs
Planetary Computer access; notebook 3 uses the files it saves.

Pass TIFF paths to the `DynamicWorld` Dataset. For training, supply images and
labels. For prediction, supply images only. Notebook 3 adds catalog, frame and
chip tables inside the Dataset, keeping the same tensor outputs.

The short training runs show how the steps fit together. Use more data and a
separate test set before drawing conclusions about model accuracy.

## Next steps

| I want to… | Read |
| --- | --- |
| Start with pretrained weights. | [Pretrained models](guides/pretrained.md) |
| Use GeoSave's training modules. | [Training](guides/training.md) |
| Save training settings in a project. | [Workspaces](guides/workspaces.md) |
| Declare a model's data requirements. | [ModelSpec](guides/model-spec.md), an Alpha API |
| Prepare spatial and temporal windows. | [Catalogs and cuts](guides/catalogs-and-cuts.md) |
| Publish a model. | [Model releases](guides/release.md) |
