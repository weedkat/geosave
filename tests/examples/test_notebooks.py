"""Every example notebook runs from top to bottom."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from geosave.geodata import attrs

EXAMPLES = Path(__file__).parents[2] / "examples"


@pytest.fixture(scope="session")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Share the notebooks' teaching-data download and model release."""
    return tmp_path_factory.mktemp("data")


def _run(
    name: str, workdir: Path, data: Path | None = None
) -> list[nbformat.NotebookNode]:
    """Execute one notebook in a fresh kernel.

    Args:
        name: Notebook file in `examples/`.
        workdir: Directory containing the lesson's inputs and any project modules.
        data: Folder linked as `data/` for inputs and outputs shared across lessons.

    Returns:
        The notebook's executed code cells and outputs.
    """
    if data is not None:
        (workdir / "data").symlink_to(data, target_is_directory=True)
    if name == "04_pipeline.ipynb":
        shutil.copy(EXAMPLES / "khdtk.geojson", workdir)
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1200,
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
    counts = {}
    for tag in events.Tags()["scalars"]:
        points = events.Scalars(tag)
        if "loss" in tag:
            assert all(math.isfinite(point.value) for point in points), tag
        counts[tag] = len(points)
    return counts


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
def test_sources_runs(tmp_path: Path) -> None:
    import numpy as np

    import geosave as gs

    code = _run("02_sources.ipynb", tmp_path)

    assert all(cell.execution_count is not None for cell in code)
    with gs.open_raster(tmp_path / "data/jakarta/ndvi.tif") as saved:
        assert list(saved.data_vars) == ["NDVI"]
        assert saved.gs.crs == "EPSG:32748"
        assert str(saved.time.values).startswith("2019-08-")
        assert np.nanstd(saved.NDVI.values) > 0


@pytest.fixture(scope="session")
def inputs_lesson(
    tmp_path_factory: pytest.TempPathFactory, data: Path
) -> tuple[Path, list[nbformat.NotebookNode]]:
    """Run lesson 3 once to produce the release for lesson 4."""
    workdir = tmp_path_factory.mktemp("landcover_lessons")
    code = _run("03_inputs.ipynb", workdir, data)
    return workdir, code


@pytest.mark.slow
@pytest.mark.integration
def test_inputs_runs(inputs_lesson, data: Path) -> None:
    workdir, code = inputs_lesson
    assert all(cell.execution_count is not None for cell in code)
    assert "val_miou" in _scalars(workdir / "lightning_logs/03_inputs/version_0")
    release = data / "landcover_release"
    config = json.loads((release / "config.json").read_text())
    assert list(config) == ["encoder", "decoder", "head"]
    assert config["encoder"]["init_args"]["pretrained"] is False
    assert (release / "model.safetensors").is_file()
    assert (release / "labels.yaml").is_file()


@pytest.mark.slow
@pytest.mark.integration
def test_pipeline_runs(inputs_lesson, tmp_path: Path, data: Path) -> None:
    """Reload the trained network, reproduce its map and publish a forest observation."""
    import pystac
    import geosave as gs

    code = _run("04_pipeline.ipynb", tmp_path, data)
    assert all(cell.execution_count is not None for cell in code)

    (item_path,) = (data / "khdtk_maps").glob("*/item.json")
    item = pystac.Item.from_file(item_path)
    with gs.open_raster(item.assets["class_id"].get_absolute_href()) as product:
        assert len(product.class_id.gs.flags.classes) == 9
        assert product.class_id.gs.nodata == 255


@pytest.mark.parametrize("name", ["00_basics.ipynb", "01_model.ipynb"])
def test_training_dataset_scales_images_and_remaps_label_class_ids(name, tmp_path):
    """Original annotation class IDs become training classes; reflectance is scaled."""
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
    bands["B11"] = (("y", "x"), np.full((2, 2), 6000, dtype="uint16"))

    image = build_dataset(bands, grid).gs.write_packing(
        attrs.CFPacking(scale_factor=1e-4, fill_value=0)
    )
    image["B03"].values[1, 0] = 0
    # Original class IDs: 0 unlabelled, 1 water, 2 trees, 10 cloud.
    labels = build_dataset(
        {"band_1": (("y", "x"), np.array([[1, 0], [2, 10]], dtype="uint8"))}, grid
    )
    # The `dynamic_world` scheme: class IDs 1-9 shift down, 0 and 10 are ignored.
    lookup = np.array([255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype="uint8")
    image_path = gs.geodata.io.raster.geotiff.write_cog(image, tmp_path / "image.tif")
    label_path = gs.geodata.io.raster.geotiff.write_cog(labels, tmp_path / "label.tif")

    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    namespace = {
        "Dataset": Dataset,
        "gs": gs,
        "BANDS": ["B02", "B03", "B04", "B08"],
        "lookup": lookup,
        "ignore_index": 255,
    }
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {
                "remap_labels",
                "fill_missing_pixels",
                "DynamicWorld",
            }:
                exec(compile(ast.Module([node], []), name, "exec"), namespace)
    dataset = namespace["DynamicWorld"]([image_path], [label_path])
    inputs, target = dataset[0]
    pixels = inputs["image"]

    assert list(inputs) == ["image"]
    torch.testing.assert_close(pixels[:, 1, 1], torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert target.dtype == torch.int64
    assert pixels[1, 1, 0] == 0
    assert target.tolist() == [[0, 255], [1, 255]]


@pytest.mark.parametrize("name", ["00_basics.ipynb", "01_model.ipynb"])
def test_prediction_uses_image_grid_without_reference_labels(name, tmp_path):
    """Run the prediction cell with an image and class definitions only."""
    from types import SimpleNamespace

    import numpy as np
    import torch
    from odc.geo.geobox import GeoBox

    import geosave as gs
    from geosave.geodata.io.raster.geotiff import write_cog
    from tests.conftest import build_dataset

    grid = GeoBox.from_bbox(
        (500000, 9200000, 500040, 9200040), crs="EPSG:32749", resolution=10
    )
    bands = {}
    for band in ["B02", "B03", "B04", "B08"]:
        pixels = np.arange(16, dtype="uint16").reshape(4, 4) + 2000
        bands[band] = (("y", "x"), pixels)
    image = build_dataset(bands, grid).gs.write_packing(
        attrs.CFPacking(scale_factor=1e-4, fill_value=0)
    )
    image = image.gs.write_rgb(red="B04", green="B03", blue="B02")
    # Dynamic World scenes are dated by their filename, as this one is.
    image_path = write_cog(image, tmp_path / "20190122_image.tif")
    sample = gs.open_raster(image_path, mask_and_scale=True)
    class_names = {0: "water", 1: "trees"}

    convolution = torch.nn.Conv2d(4, 2, 1)

    # Stands in for the notebook's Segmenter, which reads its inputs by name.
    def model(inputs):
        return convolution(inputs["image"])

    model.device = torch.device("cpu")
    model.eval = convolution.eval
    namespace = {
        "gs": gs,
        "torch": torch,
        "val": SimpleNamespace(read_sample=lambda index: (sample, None)),
        "module": model,
        "ignore_index": 255,
        "class_names": class_names,
        "class_colors": {0: "#419bdf", 1: "#397d49"},
    }
    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    cell = next(
        cell
        for cell in notebook.cells
        if "predicted = wrap_xr(" in cell.source
    )
    exec(compile(cell.source, name, "exec"), namespace)

    predicted = namespace["predicted"]
    assert predicted.gs.geobox == grid
    assert predicted.gs.flags.classes == class_names
    assert predicted.gs.nodata == 255
    assert predicted.dtype == np.uint8
    assert set(np.unique(predicted.values)) <= {0, 1}
