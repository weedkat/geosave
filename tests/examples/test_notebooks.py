"""Every example notebook runs from top to bottom."""

from __future__ import annotations

import math
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
        revision="4d3353a45d76c55a92506ef89aedf43efafc3bf6",
        local_dir=folder / "dynamic_world",
    )
    return folder


def _run(name: str, workdir: Path, data: Path | None = None) -> list[nbformat.NotebookNode]:
    """Execute one notebook in a fresh kernel.

    Args:
        name: Notebook file in `examples/`.
        workdir: Directory containing the lesson's inputs and any project modules.
        data: Downloaded data folder, linked as `data/`; the notebook's own
            download then only checks it.

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
    code = _run("02_sources.ipynb", tmp_path)

    assert all(cell.execution_count is not None for cell in code)

    assert (tmp_path / "data/jakarta/ndvi.tif").is_file()


@pytest.fixture(scope="session")
def inputs_lesson(
    tmp_path_factory: pytest.TempPathFactory, data: Path
) -> tuple[Path, list[nbformat.NotebookNode]]:
    """Run lesson 3 once to produce the checkpoint for lesson 4."""
    workdir = tmp_path_factory.mktemp("landcover_lessons")
    code = _run("03_inputs.ipynb", workdir, data)
    return workdir, code


@pytest.mark.slow
@pytest.mark.integration
def test_inputs_runs(inputs_lesson, data: Path) -> None:
    workdir, code = inputs_lesson
    assert all(cell.execution_count is not None for cell in code)
    assert "val_miou" in _scalars(workdir / "lightning_logs/03_inputs/version_0")
    assert (data / "checkpoints/03_inputs/last.ckpt").is_file()


def test_predictor_bounds_model_batches_and_restores_each_scene(tmp_path):
    import runpy

    import numpy as np
    import torch
    import xarray as xr

    import geosave as gs
    from geosave.geodata.io.raster.geotiff import write_cog
    from tests.conftest import build_dataset
    from tests.geodata.conftest import geobox

    notebook = nbformat.read(EXAMPLES / "03_inputs.ipynb", as_version=4)
    cell = next(cell for cell in notebook.cells if "class LandCoverPredictor(" in cell.source)
    cell_path = tmp_path / "predictor.cell"
    cell_path.write_text(cell.source)
    namespace = runpy.run_path(str(cell_path), init_globals={"gs": gs, "xr": xr, "torch": torch})
    predictor = namespace["LandCoverPredictor"](size=2, stride=1, batch_size=16)

    batch_sizes = []

    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, image, centroid, year, day_of_year):
            batch_sizes.append(len(image))
            assert not torch.is_grad_enabled()
            assert centroid.shape == (len(image), 2)
            assert (year == 2025).all() and (day_of_year == 15).all()
            return image * self.scale

    model = IdentityModel()
    for scene, (shape, expected_batches) in enumerate([((6, 7), [16, 14]), ((3, 4), [6])]):
        values = np.arange(np.prod(shape), dtype="float32").reshape(shape) + scene * 100
        image = build_dataset(
            {"B02": (("y", "x"), values)}, geobox(shape=shape), time=np.datetime64("2025-01-15")
        )
        path = write_cog(image, tmp_path / f"{scene}.tif")
        batch_sizes.clear()
        with gs.open_raster(path) as raster:
            logits = predictor(model, raster.gs.to_tree("image"))
        assert batch_sizes == expected_batches
        assert logits.gs.geobox == image.gs.geobox
        np.testing.assert_allclose(logits.values[0], values, atol=1e-5)


@pytest.mark.skip(
    reason="the lesson functions call the removed gs.Chips and "
    "old recipe contract; rewrite with RasterChipper and explicit STAC Items"
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
    from geosave.geodata import stac

    item = stac.StacItem.from_region(image.gs.region, id="source")
    item.set_collection(collection)
    published = stac.publish_item(
        item, assets={"image": image}, root=tmp_path / "source"
    )
    source = stac.create_catalog([published])
    path = source.gs.to_geoparquet(tmp_path / "source.parquet")
    client = gs.StacTable(path)
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
    label_path = io.raster.geotiff.write_cog(label, tmp_path / "label.tif")
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
    assert all(node.gs.to_dataset().gs.geobox == grid for node in tree.leaves)

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
def test_pipeline_runs(inputs_lesson, tmp_path: Path, data: Path) -> None:
    """Package the trained forest model as a pipeline and publish its maps."""
    import json

    import yaml

    import geosave as gs

    code = _run("04_pipeline.ipynb", tmp_path, data)
    assert all(cell.execution_count is not None for cell in code)
    roundtrip = next(
        cell for cell in code if "reloaded.predict(rasters).equals(cover)" in cell.source
    )
    assert roundtrip.outputs[-1].data["text/plain"] == "True"

    config = json.loads((data / "landcover_pipeline" / "config.json").read_text())
    assert config["predictor"]["class_path"] == "__main__.LandCoverPredictor"
    maps = gs.open_vector(data / "khdtk_maps" / "catalog.parquet")
    assert sorted(maps["datetime"].dt.strftime("%Y-%m")) == [f"2019-{month:02}" for month in range(5, 11)]

    target = yaml.safe_load((data / "dynamic_world/labels.yaml").read_text())["dynamic_world"]
    with gs.open_raster(maps.iloc[0]["assets"]["class_id"]["href"]) as cover:
        flags = cover["class_id"].gs.flags
        assert flags is not None
        assert flags.classes == dict(enumerate(target["classes"]))
        assert flags.colors == {
            code: color.upper() for code, color in enumerate(target["colors"])
        }


@pytest.mark.parametrize("name", ["00_basics.ipynb", "01_model.ipynb"])
def test_training_dataset_scales_images_and_remaps_label_codes(name, tmp_path):
    """Original annotation codes become training classes; reflectance is scaled."""
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

    image = build_dataset(bands, grid).gs.write_packing(
        attrs.CFPacking(scale_factor=1e-4, fill_value=0)
    )
    image["B03"].values[1, 0] = 0
    # Original codes: 0 unlabelled, 1 water, 2 trees, 10 cloud.
    labels = build_dataset(
        {"band_1": (("y", "x"), np.array([[1, 0], [2, 10]], dtype="uint8"))}, grid
    )
    # The `dynamic_world` scheme: codes 1-9 shift down, 0 and 10 are ignored.
    lookup = np.array([255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype="uint8")
    image_path = gs.geodata.io.raster.geotiff.write_cog(image, tmp_path / "image.tif")
    label_path = gs.geodata.io.raster.geotiff.write_cog(labels, tmp_path / "label.tif")

    notebook = nbformat.read(EXAMPLES / name, as_version=4)
    namespace = {
        "Dataset": Dataset,
        "gs": gs,
        "BANDS": ["B02", "B03", "B04", "B08"],
        "lookup": lookup,
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
    pixels, target = dataset[0]

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
    image_path = write_cog(image, tmp_path / "image.tif")
    sample = gs.open_raster(image_path, mask_and_scale=True)
    class_names = {0: "water", 1: "trees"}

    model = torch.nn.Conv2d(4, 2, 1)
    model.device = torch.device("cpu")
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
        cell for cell in notebook.cells if "predicted = xr.DataArray(" in cell.source
    )
    exec(compile(cell.source, name, "exec"), namespace)

    predicted = namespace["predicted"]
    assert predicted.gs.geobox == grid
    assert predicted.gs.flags.classes == class_names
    assert predicted.gs.nodata == 255
    assert predicted.dtype == np.uint8
    assert set(np.unique(predicted.values)) <= {0, 1}
