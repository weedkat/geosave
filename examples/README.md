# Map land cover, one project at a time

Start with labeled satellite images. Train a baseline, investigate a different
model, acquire imagery for your own forest, and turn predictions into maps.
Each notebook tackles the next problem in that project and leaves files or
results you can use in the next lesson.

## Follow the project

| Lesson | The problem you solve | What you leave with |
| --- | --- | --- |
| [0 · Explore and train](00_basics.ipynb) | Can these images and labels teach a model to recognize land cover? | A baseline prediction and training logs; optional per-class review. |
| [1 · Compare models](01_model.ipynb) | What should I change when the baseline misses things? | A second training run and concrete experiments to compare. |
| [2 · Build a dataset](02_sources.ipynb) | How do I find matching imagery for my labels and my forest? | Aligned training pairs, six forest observations, and catalogs. |
| [3 · Train and predict](03_inputs.ipynb) | How do I turn those images into complete land-cover maps? | Dated forest maps, evaluation results, and the trained model. |
| [4 · Keep the input recipe](04_recipe.ipynb) | How do I map the next observation without changing what the model sees? | A saved preparation recipe and an image/prediction pair for review. |

Run them in order. Notebook 1 can reuse the downloaded teaching data without
rerunning notebook 0; notebook 3 needs notebook 2's acquired files; notebook 4
needs notebook 3's **Keep this trained model** cell. Start a fresh kernel for
each lesson: files carry forward, Python variables do not.

Each lesson ends with **More things you can do**: ways to extend this land-cover
project, such as reviewing difficult classes, expanding the observation period,
or preparing a map for a forest manager. There are no quizzes or answer cells.

## Before you start

From a repository checkout:

```bash
uv sync
```

Open the notebooks in Jupyter or VS Code, select the repository `.venv` as the
kernel, and use `examples/` as the working directory. Run cells top to bottom.
The teaching-data download is about 400 MB. CPU training is supported; the short
runs demonstrate the workflow rather than produce a validated operational map.
Notebook 2 additionally makes live Planetary Computer queries and reads imagery;
this can take longer than training. Notebook 4 uses existing files with no new
download or training run.

Plots render in the notebook. TensorBoard and the Explorer server are optional
snippets you can run separately. Rerunning training starts a fresh run unless
you explicitly supply a checkpoint. Use a different logger version to keep
experiments separate. Cells with `overwrite=True` replace their example outputs.

## The data and what it means

The pinned teaching subset pairs Sentinel-2 imagery with **expert-annotated
Dynamic World training labels**, not the automated Dynamic World product.
It has ten classes including cloud, nodata 255, and four bands in the order
B02, B03, B04, B08. Each notebook links the dataset card and original source;
retain their attribution when sharing derived work.

Notebooks 0 and 1 use the same eight complete training pairs and two validation
pairs for a short experiment. Their Datasets read prepared data and convert it
to tensors. Notebook 3 handles missing coverage in newly acquired imagery before
constructing its Dataset; categorical labels are never reflectance-scaled.

Keep the train/validation/test splits fixed before cutting chips. Inspect
geographic overlap before claiming generalization. Compare experiments on
validation data and reserve the test split for final evaluation. The forest
observations have no reference labels: use them to apply and inspect the model,
not to measure its accuracy. Predicted differences between dates do not by
themselves establish forest loss.

## Files that connect the lessons

| Folder | What it carries |
| --- | --- |
| `data/dynamic_world/` | Downloaded teaching images, labels, and class definitions |
| `data/my_dataset/` | Notebook 2's acquired pairs and fixed split catalog |
| `data/khdtk/` | The forest observations from notebook 2 |
| `data/khdtk_pred/` | Notebook 3's dated land-cover maps and catalog |
| `data/landcover/` | Notebook 3's trained architecture, weights, and classes; notebook 4 adds the input recipe and review maps |
| `lightning_logs/` | Training curves and checkpoints for resuming experiments |

Keep catalogs with the image folders their relative paths reference. Notebook 3
loads its small collection before preparation. `Chips` reads windows on demand;
merging predictions allocates full output maps. Prepare larger projects in batches.

## Where the project goes next

The [series plan](../docs/contributing/specs/2026-10-09-examples-series-design.md)
continues with improving this model using pretrained weights, handling cloudy
months with additional observations, moving the experiment into a workspace,
and handing the trained model to a colleague. These later lessons are planned,
not yet runnable notebooks. Each will keep the project question, data, and
comparison visible instead of teaching a library subsystem in isolation.
