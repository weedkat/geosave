"""Exercise the notebook's Dataset, DataModule and postprocessing offline."""

import ast
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import kornia.augmentation as K
import nbformat
import numpy as np
from odc.geo.geobox import GeoBox
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

import geosave as gs
from geosave.model import build_model


@pytest.fixture(autouse=True)
def _limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _definitions():
    notebook = nbformat.read(
        Path(__file__).parents[2] / "examples/03_inputs.ipynb", as_version=4
    )
    namespace = {
        "gs": gs,
        "L": L,
        "K": K,
        "torch": torch,
        "nn": nn,
        "build_model": build_model,
        "MetricCollection": MetricCollection,
        "MulticlassAccuracy": MulticlassAccuracy,
        "MulticlassJaccardIndex": MulticlassJaccardIndex,
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
                "Segmenter",
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
            (300000, 5000000, 301800, 5001600), f"EPSG:{epsg}", resolution=10
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
        )
        assert len(dataset.catalog) == 2
        assert set(dataset.catalog["collection"]) == {"sentinel-2-l2a", "label"}
        assert len(dataset) == 1
        assert (
            dataset.frames.iloc[0]
            .gs.to_xarray(resolution=10)["label"]
            .dataset.gs.geobox.crs.epsg
            == epsg
        )
        image, label = dataset[0]
        assert image.shape == (1, 160, 180)
        assert label.shape == (160, 180)
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
    assert image.shape == (1, 1, 160, 180)
    assert label[0, 0, 0].item() == 99
    image, label = next(iter(data.val_dataloader()))
    assert image.shape == (1, 1, 128, 128)
    assert label.shape == (1, 128, 128)
    assert data.val_dataset.rows.iloc[0].gs.geobox.crs.epsg == 32634
    assert data.test_dataset is None

    data.setup("test")
    image, label = next(iter(data.test_dataloader()))
    assert image.shape == (1, 1, 128, 128)
    assert label.shape == (1, 128, 128)
    assert data.test_dataset.rows.iloc[0].gs.geobox.crs.epsg == 32635


def test_training_hook_crops_image_and_mask_together_and_skips_validation():
    namespace = _definitions()
    data = namespace["DynamicWorldDataModule"]()
    image = torch.arange(160 * 180).reshape(1, 1, 160, 180).float()
    label = image[:, 0].long() % 2
    data.trainer = SimpleNamespace(training=True)

    torch.manual_seed(0)
    cropped, target = data.on_after_batch_transfer((image, label), 0)

    assert cropped.shape == (1, 1, 128, 128)
    assert target.shape == (1, 128, 128)
    torch.testing.assert_close(cropped[:, 0].long() % 2, target)
    torch.manual_seed(7)
    other, _ = data.on_after_batch_transfer((image, label), 0)
    assert not torch.equal(cropped, other)

    data.trainer = SimpleNamespace(training=False)
    batch = (image, label)
    assert data.on_after_batch_transfer(batch, 0) is batch


def test_training_crop_preserves_label_ids_and_ignored_pixels():
    data = _definitions()["DynamicWorldDataModule"]()
    data.trainer = SimpleNamespace(training=True)
    image = torch.ones(1, 1, 128, 128)
    label = torch.full((1, 128, 128), 2, dtype=torch.long)
    label[:, 0, 0] = 99

    _, target = data.on_after_batch_transfer((image, label), 0)

    assert set(target.unique().tolist()) == {2, 99}
    assert target.eq(2).sum() == 128 * 128 - 1


@pytest.mark.parametrize("size", [64, 128])
@pytest.mark.parametrize("shape", [(160, 180), (31, 45)])
def test_prediction_merges_chips_before_scoring_every_valid_pixel(
    tmp_path, size, shape
):
    namespace = _definitions()
    grid = GeoBox.from_bbox(
        (300000, 5000000, 300000 + shape[1] * 10, 5000000 + shape[0] * 10),
        "EPSG:32633",
        resolution=10,
    )
    pixels = np.linspace(-1, 1, shape[0] * shape[1], dtype="float32").reshape(shape)
    image = gs.raster({"red": (("y", "x"), pixels)}, grid, nodata=np.nan)
    targets = np.zeros(shape, dtype="uint8")
    targets[0, 0] = 99
    label = gs.raster({"label": (("y", "x"), targets)}, grid, nodata=99)
    for name, raster in [("image", image), ("label", label)]:
        raster.expand_dims(time=[np.datetime64("2024-01-10")]).gs.to_cog(
            tmp_path / f"{name}.tif"
        )
    dataset = namespace["DynamicWorld"](
        [tmp_path / "image.tif"],
        [tmp_path / "label.tif"],
        chip_size=size,
        overlap=size // 2,
    )
    module = namespace["Segmenter"](
        num_classes=2, ignore_index=99, in_channels=1, input_size=size, width=2
    )

    class PixelClassifier(nn.Module):
        def forward(self, *, image):
            return torch.cat([image, -image], dim=1)

    module.model = PixelClassifier()
    module.eval()
    predictions = []
    for index, batch in enumerate(DataLoader(dataset, batch_size=8)):
        image, label = batch
        assert image.shape[-2:] == (size, size)
        assert label.shape[-2:] == (size, size)
        with torch.no_grad():
            predictions.append(module(image))
            module.validation_step(batch, index)
            module.test_step(batch, index)
    logits = torch.cat(predictions).numpy()
    scores = next(iter(dataset.merge(logits).values()))
    expected = torch.from_numpy(np.stack([pixels, -pixels])).unsqueeze(0)
    assert scores.gs.geobox == grid
    np.testing.assert_allclose(scores.to_array().values, expected[0].numpy())

    module.evaluate(dataset, module.val_outputs, module.val_metrics, "val")
    module.evaluate(dataset, module.test_outputs, module.test_metrics, "test")
    assert module.val_outputs == []
    assert module.test_outputs == []
    target = torch.from_numpy(targets.astype("int64")).unsqueeze(0)
    expected_iou = MulticlassJaccardIndex(num_classes=2, ignore_index=99)(
        expected, target
    )
    for metrics in [module.val_metrics, module.test_metrics]:
        assert metrics["miou"].confmat.sum() == shape[0] * shape[1] - 1
        torch.testing.assert_close(metrics["miou"].compute(), expected_iou)


def test_short_lightning_run_uses_random_crops_and_complete_region_metrics(root):
    namespace = _definitions()
    data = namespace["DynamicWorldDataModule"](root=root, batch_size=1, crop_size=64)
    module = namespace["Segmenter"](
        num_classes=8, ignore_index=99, in_channels=1, input_size=64, width=2
    )
    trainer = L.Trainer(
        max_epochs=2,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    shapes = []

    def record_training_shape(model, args, kwargs):
        if module.training:
            shapes.append(tuple(kwargs["image"].shape[-2:]))

    module.model.register_forward_pre_hook(record_training_shape, with_kwargs=True)

    trainer.fit(module, datamodule=data)
    assert "val_miou" in trainer.callback_metrics
    assert shapes == [(64, 64), (64, 64)]
    assert module.val_outputs == []
    result = trainer.test(module, datamodule=data, verbose=False)[0]

    assert len(data.train_dataset) == 1
    assert np.isfinite(result["test_miou"])
    assert module.test_outputs == []


def test_empty_training_crop_has_finite_differentiable_zero_loss():
    module = _definitions()["Segmenter"](
        num_classes=2, ignore_index=99, in_channels=1, input_size=64, width=2
    )
    logits = torch.randn(1, 2, 64, 64, requires_grad=True)
    label = torch.full((1, 64, 64), 99, dtype=torch.long)

    loss = module.compute_loss(logits, label)
    loss.backward()

    assert loss.item() == 0
    assert torch.isfinite(logits.grad).all()


def test_postprocess_preserves_class_ids_missing_pixels_and_georeferencing(tmp_path):
    namespace = _definitions()
    grid = GeoBox.from_bbox(
        (300000, 5000000, 302560, 5001280), "EPSG:32633", resolution=10
    )
    image = gs.raster({"red": (("y", "x"), np.ones(grid.shape, dtype="float32"))}, grid)
    image = image.expand_dims(time=[np.datetime64("2024-01-10")])
    image.gs.to_cog(tmp_path / "image.tif")
    dataset = namespace["DynamicWorld"]([tmp_path / "image.tif"], chip_size=128)
    assert dataset[0].shape == (1, 128, 128)
    logits = np.empty((2, 2, 128, 128), dtype="float32")
    logits[0, 0], logits[0, 1] = 1, 3
    logits[1, 0], logits[1, 1] = 4, 2
    logits[:, :, 0, 0] = np.nan

    merged = dataset.merge(logits)
    cover = namespace["postprocess"](merged, dataset.frames)

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


def test_dataset_rejects_partial_predictions_before_reconstructing_maps(root):
    namespace = _definitions()
    dataset = namespace["DynamicWorld"](
        sorted((root / "val/images").glob("*.tif")),
        chip_size=64,
    )
    with pytest.raises(ValueError):
        dataset.merge(np.zeros((1, 2, 64, 64), dtype="float32"))
