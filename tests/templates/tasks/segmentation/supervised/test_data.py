from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from generated_kits.segmentation.supervised import training as supervised

FLIP = [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]


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
            "inputs": {"image": torch.full((2, 8, 8), float(index))},
            "target": torch.full((8, 8), index % 2),
        }


def _data(tmp_path: Path, **options: object) -> supervised.DataModule:
    return supervised.DataModule(
        train=Chips(),
        val=Chips(),
        **{"num_workers": 0, **options},
    )


def _batch(times: int = 0) -> supervised.Sample:
    shape = (2, times, 2, 8, 8) if times else (2, 2, 8, 8)
    image = torch.arange(float(np.prod(shape))).reshape(shape)
    return {
        "inputs": {
            "image": image,
            "transform": TRANSFORM.expand(2, 6),
            "crs": CRS.expand(2),
            "centroid": torch.tensor([[51.0, 11.0], [51.0, 11.0]]),
        },
        "target": torch.arange(2 * 8 * 8).reshape(2, 8, 8) % 3,
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

    image = original["inputs"]["image"]
    torch.testing.assert_close(batch["inputs"]["image"], image.flip(-1))
    assert torch.equal(batch["target"], original["target"].flip(-1))
    assert batch["target"].dtype == torch.int64


def test_cutmix_pastes_the_same_box_into_pixels_and_labels(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, cutmix=1.0), True)
    original = _batch(times=3)

    batch = data.on_after_batch_transfer(_batch(times=3), 0)

    # Every pixel value is unique, so the box is wherever pixels changed.
    changed = batch["inputs"]["image"] != original["inputs"]["image"]
    box = changed.flatten(1, 2).all(dim=1)
    target = original["target"]
    assert box.any()
    assert torch.equal(batch["target"], torch.where(box, target.roll(1, 0), target))


def test_augmentations_do_not_run_outside_training(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, augmentations=FLIP), False)
    original = _batch()

    batch = data.on_after_batch_transfer(_batch(), 0)

    torch.testing.assert_close(batch["inputs"]["image"], original["inputs"]["image"])
    assert torch.equal(batch["target"], original["target"])


def test_a_time_axis_survives_augmentation(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, augmentations=FLIP), True)
    image = _batch(times=3)["inputs"]["image"]

    result = data.on_after_batch_transfer(_batch(times=3), 0)["inputs"]["image"]

    torch.testing.assert_close(result, image.flip(-1))


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
    batch["inputs"]["dem"] = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(
        2, 1, 8, 8
    )

    result = data.on_after_batch_transfer(batch, 0)

    for name in ("image", "dem"):
        torch.testing.assert_close(
            result["inputs"][name], batch["inputs"][name][..., 2:6, 2:6]
        )
    torch.testing.assert_close(result["target"], batch["target"][..., 2:6, 2:6])
    for name in ("transform", "crs", "centroid"):
        assert result["inputs"][name] is batch["inputs"][name]


def test_lightning_cli_builds_each_split_dataset(tmp_path: Path) -> None:
    from geosave.ml.torch.training.cli import GeosaveCLI

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
                    "class_path": "torch.nn.CrossEntropyLoss",
                    "init_args": {"ignore_index": 255},
                },
                "optimizer": {
                    "class_path": "torch.optim.AdamW",
                    "init_args": {"lr": 0.001},
                },
                "param_groups": None,
                "lr_scheduler": None,
                "lr_scheduler_config": None,
                "metrics": ["iou_macro"],
            },
            "data": {
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
