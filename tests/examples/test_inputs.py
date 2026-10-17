"""Exercise the notebook's Dataset, DataModule and postprocessing offline."""

import ast
from pathlib import Path

import lightning as L
import nbformat
import numpy as np
from odc.geo.geobox import GeoBox
import pytest
from torch.utils.data import DataLoader, Dataset

import geosave as gs


def _definitions():
    notebook = nbformat.read(
        Path(__file__).parents[2] / "examples/03_inputs.ipynb", as_version=4
    )
    namespace = {
        "gs": gs,
        "L": L,
        "Dataset": Dataset,
        "DataLoader": DataLoader,
        "Path": Path,
        "IGNORE_INDEX": 99,
        "CLASS_MAP": {7: "trees", 2: "water"},
        "COLOR_MAP": {2: "#419bdf", 7: "#397d49"},
    }
    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.source.startswith("%"):
            continue
        module = ast.parse(cell.source)
        for node in module.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in (
                "preprocess",
                "DynamicWorld",
                "DynamicWorldDataModule",
                "postprocess",
            ):
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), "03_inputs", "exec"
                    ),
                    namespace,
                )
    return namespace


@pytest.fixture
def root(tmp_path):
    for split, epsg in [("train", 32633), ("val", 32634), ("test", 32635)]:
        grid = GeoBox.from_bbox(
            (300000, 5000000, 301280, 5001280), f"EPSG:{epsg}", resolution=10
        )
        image = gs.raster(
            {"B02": (("y", "x"), np.full(grid.shape, 0.2, dtype="float32"))}, grid
        ).expand_dims(time=[np.datetime64("2024-01-10")])
        pixels = np.full(grid.shape, 7, dtype="uint8")
        pixels[0, 0] = 99
        label = gs.raster({"label": (("y", "x"), pixels)}, grid, nodata=99)
        label = label.expand_dims(time=[np.datetime64("2024-01-10")])
        image.gs.to_cog(tmp_path / split / "images/tile.tif")
        label.gs.to_cog(tmp_path / split / "labels/tile.tif")
    return tmp_path


def test_dataset_builds_catalog_from_paths_and_preserves_each_native_grid(root):
    namespace = _definitions()
    for split, epsg in [("train", 32633), ("val", 32634), ("test", 32635)]:
        dataset = namespace["DynamicWorld"](
            sorted((root / split / "images").glob("*.tif")),
            sorted((root / split / "labels").glob("*.tif")),
            transform=namespace["preprocess"],
        )
        assert len(dataset.catalog) == 2
        assert set(dataset.catalog["collection"]) == {"sentinel-2-l2a", "label"}
        assert len(dataset) == 1
        assert dataset.chips.iloc[0].gs.geobox.crs.to_epsg() == epsg
        image, label = dataset[0]
        assert image.shape == (1, 128, 128)
        assert label.shape == (128, 128)
        np.testing.assert_allclose(image.numpy(), 0.2)
        assert label[0, 0].item() == 99
        assert label[1, 1].item() == 7


def test_datamodule_constructs_datasets_for_each_stage(root):
    namespace = _definitions()
    data = namespace["DynamicWorldDataModule"](root=root, batch_size=1)
    assert data.train_dataset is None
    assert data.val_dataset is None
    assert data.test_dataset is None

    data.setup("fit")
    image, label = next(iter(data.train_dataloader()))
    assert image.shape == (1, 1, 128, 128)
    assert label[0, 0, 0].item() == 99
    assert data.val_dataset.chips.iloc[0].gs.geobox.crs.to_epsg() == 32634
    assert data.test_dataset is None

    data.setup("test")
    image, label = next(iter(data.test_dataloader()))
    assert image.shape == (1, 1, 128, 128)
    assert data.test_dataset.chips.iloc[0].gs.geobox.crs.to_epsg() == 32635


def test_postprocess_preserves_class_ids_missing_pixels_and_georeferencing(tmp_path):
    namespace = _definitions()
    grid = GeoBox.from_bbox(
        (300000, 5000000, 302560, 5001280), "EPSG:32633", resolution=10
    )
    image = gs.raster({"red": (("y", "x"), np.ones(grid.shape, dtype="float32"))}, grid)
    image = image.expand_dims(time=[np.datetime64("2024-01-10")])
    image.gs.to_cog(tmp_path / "image.tif")
    dataset = namespace["DynamicWorld"](
        [tmp_path / "image.tif"], transform=namespace["preprocess"]
    )
    assert dataset[0].shape == (1, 128, 128)
    logits = np.empty((2, 2, 128, 128), dtype="float32")
    logits[0, 0], logits[0, 1] = 1, 3
    logits[1, 0], logits[1, 1] = 4, 2
    logits[:, :, 0, 0] = np.nan

    cover = namespace["postprocess"](logits, dataset.frames, dataset.chips)

    assert cover.name == "label"
    assert cover.dtype == np.dtype("uint8")
    assert cover.gs.geobox == grid
    assert cover.gs.nodata == 99
    assert cover.gs.legend.classes == {2: "water", 7: "trees"}
    assert cover.gs.legend.colors == {2: "#419BDF", 7: "#397D49"}
    np.testing.assert_array_equal(cover.time.values, [np.datetime64("2024-01-10")])
    assert cover.values[0, 1, 1] == 7
    assert cover.values[0, 1, 129] == 2
    assert cover.values[0, 0, 0] == cover.values[0, 0, 128] == 99

    cover.gs.to_cog(tmp_path / "cover.tif")
    restored = gs.open_raster(tmp_path / "cover.tif")["label"]
    assert restored.gs.nodata == 99
    assert restored.gs.legend == cover.gs.legend
    np.testing.assert_array_equal(restored.values, cover.isel(time=0).values)
