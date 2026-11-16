from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from modules.segmentation import supervised
from geosave.ml.processing.raster.preprocessing.base import RasterPreprocessor

FLIP = [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]


class Standardize(RasterPreprocessor):
    """A preprocessor whose `normalize` standardizes each band."""

    def __init__(self, mean: list[float] | None = None, std: list[float] | None = None):
        self.mean = mean
        self.std = std

    def __call__(self, raster):
        return raster

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return image
        mean = torch.tensor(self.mean).reshape(-1, 1, 1)
        std = torch.tensor(self.std).reshape(-1, 1, 1)
        return (image - mean) / std


def _preprocessor(*, standardize: bool = False) -> dict[str, Any]:
    init_args = {"mean": [1.0, 2.0], "std": [2.0, 4.0]} if standardize else {}
    return {"class_path": f"{__name__}.Standardize", "init_args": init_args}


# A 1 m grid that places every test sample.
TRANSFORM = torch.tensor(
    [1.0, 0.0, 500_000.0, 0.0, -1.0, 9_000_000.0], dtype=torch.float64
)
CRS = torch.tensor(32749)


class Chips(Dataset[supervised.Sample]):
    """A dataset returning the segmentation `Sample` contract."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> supervised.Sample:
        return {
            "image": torch.full((2, 8, 8), float(index)),
            "target": torch.full((8, 8), index % 2),
            "transform": TRANSFORM,
            "crs": CRS,
        }


def _data(
    tmp_path: Path, preprocessor: dict[str, Any] | None = None, **options: object
) -> supervised.DataModule:
    return supervised.DataModule(
        preprocessor=preprocessor or _preprocessor(),
        train=Chips(),
        val=Chips(),
        **{"num_workers": 0, **options},
    )


def _batch(times: int = 0) -> supervised.Sample:
    shape = (2, times, 2, 8, 8) if times else (2, 2, 8, 8)
    image = torch.arange(float(np.prod(shape))).reshape(shape)
    return {
        "image": image,
        "target": torch.arange(2 * 8 * 8).reshape(2, 8, 8) % 3,
        "transform": TRANSFORM.expand(2, 6),
        "crs": CRS.expand(2),
        "centroid": torch.tensor([[51.0, 11.0], [51.0, 11.0]]),
    }


def _training(data: supervised.DataModule, training: bool) -> supervised.DataModule:
    data.trainer = SimpleNamespace(training=training)  # type: ignore[assignment]
    return data


def test_augmentations_move_image_and_target_together_while_training(
    tmp_path: Path,
) -> None:
    data = _training(_data(tmp_path, augmentations=FLIP), True)
    original = _batch()

    batch = data.on_after_batch_transfer(_batch(), 0)

    image = original["image"]
    torch.testing.assert_close(batch["image"], image.flip(-1))
    assert torch.equal(batch["target"], original["target"].flip(-1))
    assert batch["target"].dtype == torch.int64


def test_cutmix_pastes_the_same_box_into_pixels_and_labels(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, cutmix=1.0), True)
    original = _batch(times=3)

    batch = data.on_after_batch_transfer(_batch(times=3), 0)

    # Every pixel value is unique, so the box is wherever pixels changed.
    changed = batch["image"] != original["image"]
    box = changed.flatten(1, 2).all(dim=1)
    target = original["target"]
    assert box.any()
    assert torch.equal(batch["target"], torch.where(box, target.roll(1, 0), target))


def test_augmentations_do_not_run_outside_training(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, augmentations=FLIP), False)
    original = _batch()

    batch = data.on_after_batch_transfer(_batch(), 0)

    torch.testing.assert_close(batch["image"], original["image"])
    assert torch.equal(batch["target"], original["target"])


@pytest.mark.parametrize("training", [True, False])
def test_transforms_run_in_every_split_after_augmentation(
    tmp_path: Path, training: bool
) -> None:
    preprocessor = _preprocessor(standardize=True)
    data = _training(_data(tmp_path, preprocessor, augmentations=FLIP), training)
    image = _batch()["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 2, 1, 1)

    result = data.on_after_batch_transfer(_batch(), 0)["image"]

    expected = (image.flip(-1) if training else image) - mean
    torch.testing.assert_close(result, expected / std)


def test_a_time_axis_survives_augmentation_and_transforms(tmp_path: Path) -> None:
    preprocessor = _preprocessor(standardize=True)
    data = _training(_data(tmp_path, preprocessor, augmentations=FLIP), True)
    image = _batch(times=3)["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 1, 2, 1, 1)

    result = data.on_after_batch_transfer(_batch(times=3), 0)["image"]

    assert result.shape == (2, 3, 2, 8, 8)
    torch.testing.assert_close(result, (image.flip(-1) - mean) / std)


def test_a_pixel_an_augmentation_invents_is_ignored(tmp_path: Path) -> None:
    shift = [
        {
            "name": "RandomAffine",
            "init_args": {"degrees": 0.0, "translate": [0.5, 0.5], "p": 1.0},
        }
    ]
    data = _training(_data(tmp_path, augmentations=shift, ignore_index=9), True)
    batch = _batch()
    batch["target"] = torch.zeros(2, 8, 8, dtype=torch.int64)

    moved = data.on_after_batch_transfer(batch, 0)["target"]

    assert set(moved.unique().tolist()) == {0, 9}


def test_every_input_shares_the_geometry_and_the_grid_passes_through(
    tmp_path: Path,
) -> None:
    crop = [{"name": "CenterCrop", "init_args": {"size": (4, 4)}}]
    data = _training(_data(tmp_path, augmentations=crop, layers=["image", "dem"]), True)
    batch = _batch(times=2)
    batch["dem"] = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)

    result = data.on_after_batch_transfer(batch, 0)

    for name in ("image", "dem"):
        torch.testing.assert_close(result[name], batch[name][..., 2:6, 2:6])
    torch.testing.assert_close(result["target"], batch["target"][..., 2:6, 2:6])
    assert result["transform"] is batch["transform"]
    assert result["crs"] is batch["crs"]
    assert result["centroid"] is batch["centroid"]


def test_lightning_cli_builds_each_split_dataset(tmp_path: Path) -> None:
    from geosave.ml.cli import GeosaveCLI

    chips = {"class_path": f"{__name__}.Chips"}
    cli = GeosaveCLI(
        supervised.Module,
        supervised.DataModule,
        run=False,
        save_config_callback=None,
        seed_everything_default=False,
        auto_configure_optimizers=False,
        args={
            "model": {
                "model_chain": {"model": {"class_path": "torch.nn.Identity"}},
                "ignore_index": 255,
                "criterion": {
                    "name": "cross_entropy",
                    "init_args": {"ignore_index": 255},
                },
                "optimizer": {"name": "adamw", "init_args": {"lr": 0.001}},
                "lr_scheduler": None,
                "metrics": ["iou.macro"],
                "predictor": {"name": "dense", "init_args": {"size": 2}},
            },
            "data": {
                "preprocessor": _preprocessor(),
                "train": chips,
                "val": chips,
                "num_workers": 0,
            },
            "trainer": {"logger": False, "default_root_dir": str(tmp_path)},
        },
    )

    assert isinstance(cli.datamodule.train_dataset, Chips)
    assert cli.datamodule.test_dataset is None


def test_validation_and_test_batch_whole_scenes_apart_from_training(
    tmp_path: Path,
) -> None:
    data = _data(tmp_path, batch_size=4, eval_batch_size=1)

    assert data.train_dataloader().batch_size == 4
    assert data.val_dataloader().batch_size == 1
