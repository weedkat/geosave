"""Every example notebook runs from top to bottom."""

from __future__ import annotations

import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from geosave.geodata import attrs

EXAMPLES = Path(__file__).parents[2] / "examples"
DATASET = "fatmur/dynamicworld-300"


@pytest.fixture(scope="session")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download the example dataset once for every notebook that reads it."""
    from huggingface_hub import snapshot_download

    folder = tmp_path_factory.mktemp("data")
    snapshot_download(
        DATASET,
        repo_type="dataset",
        revision="5e65026ad47e1f02f9a48a3efee1028db3bb0d1a",
        local_dir=folder / "dynamic_world",
    )
    return folder


def _run(name: str, workdir: Path, data: Path) -> list[nbformat.NotebookNode]:
    """Execute one notebook in a working directory of its own.

    Args:
        name: Notebook file in `examples/`.
        workdir: Empty directory the notebook runs in.
        data: Downloaded data folder, linked as `data/`; the notebook's own
            download then only checks it.

    Returns:
        The notebook's executed code cells and outputs.
    """
    (workdir / "data").symlink_to(data, target_is_directory=True)
    # Notebooks 2 and 3 read the forest outline shipped beside them.
    shutil.copy(EXAMPLES / "khdtk.geojson", workdir)
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": workdir}},
    )
    client.execute()
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def _scalars(run: Path) -> dict[str, int]:
    """Count the points logged per TensorBoard scalar of one named run."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    events = EventAccumulator(str(run))
    events.Reload()
    return {tag: len(events.Scalars(tag)) for tag in events.Tags()["scalars"]}


@pytest.mark.slow
@pytest.mark.integration
def test_basics_runs(tmp_path: Path, data: Path) -> None:
    code = _run("00_basics.ipynb", tmp_path, data)

    assert all(cell.execution_count is not None for cell in code)
    scalars = _scalars(tmp_path / "lightning_logs/00_basics/version_0")
    for tag in ("train_loss_step", "val_loss", "val_miou", "val_accuracy"):
        assert tag in scalars
    # One validation point per epoch: a single epoch would plot as a dot.
    assert scalars["val_miou"] > 1


@pytest.mark.slow
@pytest.mark.integration
def test_model_runs(tmp_path: Path, data: Path) -> None:
    code = _run("01_model.ipynb", tmp_path, data)

    assert all(cell.execution_count is not None for cell in code)
    assert "val_miou" in _scalars(tmp_path / "lightning_logs/01_model/version_0")


@pytest.mark.slow
@pytest.mark.integration
def test_sources_runs(tmp_path: Path, data: Path) -> None:
    code = _run("02_sources.ipynb", tmp_path, data)

    assert all(cell.execution_count is not None for cell in code)
    # The project's catalog and the pairs it lists.
    import geosave as gs

    dataset = tmp_path / "data" / "my_dataset"
    table = gs.StacClient.open(dataset / "catalog.parquet")
    assert table.collections() == {"label", "sentinel-2-l2a"}
    # Each split keeps image and label filenames paired in its own folders.
    for split, count in [("train", 6), ("val", 2), ("test", 2)]:
        labels = sorted(path.name for path in (dataset / split / "label").glob("*.tif"))
        images = sorted(
            path.name for path in (dataset / split / "sentinel-2-l2a").glob("*.tif")
        )
        assert len(labels) == count and images == labels
    splits = gs.open_vector(dataset / "catalog.parquet")["split"].value_counts()
    assert splits.to_dict() == {"train": 12, "val": 4, "test": 4}
    # The forest season notebook 3 predicts over, one picture per month.
    forest = gs.StacClient.open(tmp_path / "data" / "khdtk" / "catalog.parquet")
    assert forest.collections() == {"sentinel-2-l2a"}


@pytest.mark.slow
@pytest.mark.integration
def test_inputs_runs(tmp_path: Path, data: Path) -> None:
    # Notebook 3 uses both the paired tiles and inference imagery from notebook 2.
    if not all(
        (data / folder / "catalog.parquet").is_file()
        for folder in ("khdtk", "my_dataset")
    ):
        (tmp_path / "sources").mkdir()
        test_sources_runs(tmp_path / "sources", data)

    workdir = tmp_path / "inputs"
    workdir.mkdir()
    code = _run("03_inputs.ipynb", workdir, data)

    assert all(cell.execution_count is not None for cell in code)
    assert "val_miou" in _scalars(workdir / "lightning_logs/03_train/version_0")
    # The forest's predicted land cover, one map per month of the season.
    import geosave as gs

    cover = gs.open_vector(data / "khdtk_pred" / "catalog.parquet")
    assert set(cover["collection"]) == {"land_cover"} and len(cover) == 6


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="STAC publication is being rebuilt on expand_stac and tree.gs.to_cog",
)
def test_catalog_examples_load_chips_and_restore_pixels(tmp_path):
    """Execute the lessons' data functions with small real local STAC assets."""
    import numpy as np
    import torch
    import xarray as xr
    from odc.geo.geobox import GeoBox
    from torch.utils.data import Dataset

    import geosave as gs
    from geosave.geodata import io
    from tests.conftest import build_dataset
    from tests.geodata.conftest import create_test_collection

    grid = GeoBox.from_bbox(
        (500000, 9200000, 500080, 9200080), crs="EPSG:32749", resolution=10
    )
    bands = ["B02", "B03", "B04", "B08"]
    pixels = {}
    for band in bands:
        pixels[band] = (("y", "x"), np.full((8, 8), 2000, dtype="uint16"))
    image = build_dataset(pixels, grid, nodata=0).assign_coords(
        time=np.datetime64("2025-01-15")
    )
    image["B08"].values[0, 0] = 0
    collection = create_test_collection(
        "sentinel-2-l2a", description="Local test imagery"
    )
    source = image.gs.to_catalog(tmp_path / "source", collection=collection)
    path = source.gs.to_geoparquet(tmp_path / "source.parquet")
    client = gs.StacClient.open(path)
    namespace = {
        "gs": gs,
        "io": io,
        "xr": xr,
        "BANDS": bands,
    }
    sources = nbformat.read(EXAMPLES / "02_sources.ipynb", as_version=4)
    for cell in sources.cells:
        if cell.cell_type != "code":
            continue
        if any(
            name in cell.source
            for name in ("def fetch_image(", "def save_tile(", "def build_dataset(")
        ):
            exec(compile(cell.source, "02_sources.ipynb", "exec"), namespace)
    label = build_dataset(
        {"label": (("y", "x"), np.ones((8, 8), dtype="uint8"))}, grid, nodata=255
    ).assign_coords(time=np.datetime64("2025-01-15"))
    label_path = io.formats.geotiff.write_cog(label, tmp_path / "label.tif")
    table = namespace["build_dataset"](
        [label_path],
        tmp_path / "paired",
        "train",
        client=client,
        collection=collection,
        query=gs.StacQuery(collections=[collection.id], max_items=1),
    )
    assert set(table["split"]) == {"train"}
    sample = table.gs.sample(collections=["sentinel-2-l2a", "label"], freq="D")[0]
    tree = sample.gs.to_datatree()
    assert tree.gs.geobox == grid

    import ast

    inputs = nbformat.read(EXAMPLES / "03_inputs.ipynb", as_version=4)
    namespace.update(Dataset=Dataset, torch=torch)
    for cell in inputs.cells:
        if cell.cell_type != "code":
            continue
        parsed = ast.parse(cell.source)
        definitions = []
        for node in parsed.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name in {
                    "load_pairs",
                    "prepare_pairs",
                    "cut_pairs",
                    "DynamicWorld",
                }:
                    definitions.append(node)
        module = ast.Module(body=definitions, type_ignores=[])
        exec(compile(module, "03_inputs.ipynb", "exec"), namespace)

    images, labels = namespace["load_pairs"](table)
    images, labels = namespace["prepare_pairs"](
        images, labels, bands=bands, ignore_index=255
    )
    image_chips, label_chips = namespace["cut_pairs"](
        images, labels, size=128, ignore_index=255
    )
    dataset = namespace["DynamicWorld"](image_chips, label_chips)
    image_tensor, label_tensor = dataset[0]

    assert image_tensor.shape == (4, 128, 128)
    assert label_tensor.shape == (128, 128)
    assert label_tensor.dtype == torch.int64
    torch.testing.assert_close(image_tensor[:, 1, 1], torch.full((4,), 0.2))
    assert image_tensor[3, 0, 0] == 0
    assert label_tensor[0, 0] == 255
    assert label_tensor[1, 1] == 1
    assert (label_tensor[8:, :] == 255).all()
    assert (label_tensor[:, 8:] == 255).all()

    chips = gs.Chips(images, size=4, overlap=2)
    predictions = []
    for key in chips:
        predictions.append((key, np.ones((2, 4, 4), dtype="float32")))
    (merged,) = chips.merge(predictions, window="hamming")
    assert merged.shape == (2, 8, 8)
    np.testing.assert_allclose(merged, 1)


@pytest.mark.slow
@pytest.mark.integration
def test_recipe_runs(tmp_path: Path, data: Path) -> None:
    """Carry the trained forest model into a saved preparation recipe."""
    if not (data / "landcover" / "weights.pt").is_file():
        workdir = tmp_path / "training"
        workdir.mkdir()
        test_inputs_runs(workdir, data)
    workdir = tmp_path / "recipe"
    workdir.mkdir()
    code = _run("04_recipe.ipynb", workdir, data)
    assert all(cell.execution_count is not None for cell in code)

    import geosave as gs

    review = data / "landcover" / "review"
    image_path = next(review.glob("image-*.tif"))
    cover_path = next(review.glob("land-cover-*.tif"))
    with gs.open_raster(image_path) as image, gs.open_raster(cover_path) as cover:
        assert cover.gs.geobox == image.gs.geobox
        assert cover.gs.timespan == image.gs.timespan
        assert cover["label"].gs.nodata == 255
        assert cover["label"].gs.flags is not None


@pytest.mark.parametrize("name", ["00_basics.ipynb", "01_model.ipynb"])
def test_training_dataset_scales_images_and_preserves_label_ids(name, tmp_path):
    """Prepared pairs become tensors without changing categorical targets."""
    import ast

    import numpy as np
    import torch
    from odc.geo.geobox import GeoBox
    from torch.utils.data import Dataset

    import geosave as gs
    from tests.conftest import build_dataset

    grid = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32633", resolution=10)
    bands = {}
    for index, band in enumerate(["B02", "B03", "B04", "B08"]):
        bands[band] = (("y", "x"), np.full((2, 2), 1000 * (index + 1), dtype="uint16"))

    image = build_dataset(bands, grid).gs.write_packing(attrs.CFPacking(scale_factor=1e-4, fill_value=0))
    image = image[["B08", "B04", "B03", "B02"]]
    labels = build_dataset(
        {"label": (("y", "x"), np.array([[1, 255], [2, 3]], dtype="uint8"))},
        grid,
        nodata=255,
    )
    image_path = gs.geodata.io.formats.geotiff.write_cog(image, tmp_path / "image.tif")
    label_path = gs.geodata.io.formats.geotiff.write_cog(labels, tmp_path / "label.tif")

    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    cell = next(cell for cell in notebook.cells if "class DynamicWorld(" in cell.source)
    parsed = ast.parse(cell.source)
    definition = next(node for node in parsed.body if isinstance(node, ast.ClassDef))
    module = ast.Module(body=[definition], type_ignores=[])
    namespace = {"Dataset": Dataset, "gs": gs}
    exec(compile(module, name, "exec"), namespace)
    dataset = namespace["DynamicWorld"]([image_path], [label_path])
    pixels, target = dataset[0]

    torch.testing.assert_close(pixels[:, 1, 1], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert target.dtype == torch.int64
    assert target.tolist() == [[1, 255], [2, 3]]


@pytest.mark.parametrize("name", ["00_basics.ipynb", "01_model.ipynb"])
def test_prediction_uses_image_grid_without_reference_labels(name, tmp_path):
    """Run the prediction cell with an image and class definitions only."""
    import numpy as np
    import torch
    import yaml
    from odc.geo.geobox import GeoBox

    import geosave as gs
    from geosave.geodata.io.formats.geotiff import write_cog
    from tests.conftest import build_dataset

    grid = GeoBox.from_bbox(
        (500000, 9200000, 500040, 9200040), crs="EPSG:32749", resolution=10
    )
    bands = {}
    for band in ["B02", "B03", "B04", "B08"]:
        pixels = np.arange(16, dtype="uint16").reshape(4, 4) + 2000
        bands[band] = (("y", "x"), pixels)
    image = build_dataset(bands, grid).gs.write_packing(attrs.CFPacking(scale_factor=1e-4, fill_value=0))
    image = image.gs.write_rgb(red="B04", green="B03", blue="B02")
    image_path = write_cog(image, tmp_path / "image.tif")
    classes = {
        "ignore_index": 255,
        "class_map": {0: "water", 1: "trees"},
        "color_map": {0: "#419bdf", 1: "#397d49"},
    }
    (tmp_path / "classes.yaml").write_text(yaml.safe_dump(classes))

    model = torch.nn.Conv2d(4, 2, 1)
    model.device = torch.device("cpu")
    namespace = {
        "gs": gs,
        "torch": torch,
        "root": tmp_path,
        "val_images": [image_path],
        "module": model,
    }
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    cell = next(
        cell for cell in notebook.cells if "predicted = xr.DataArray(" in cell.source
    )
    exec(compile(cell.source, name, "exec"), namespace)

    predicted = namespace["predicted"]
    assert predicted.gs.geobox == grid
    assert predicted.gs.flags.classes == classes["class_map"]
    assert predicted.gs.nodata == 255
    assert predicted.dtype == np.uint8
    assert set(np.unique(predicted.values)) <= {0, 1}
