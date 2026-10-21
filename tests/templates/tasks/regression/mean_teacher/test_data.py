from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from modules.regression import mean_teacher
from tests.templates.tasks.segmentation.supervised.test_data import FLIP, _batch, _spec
from tests.templates.tasks.regression.supervised.test_data import Chips

NOISE = [{"name": "RandomGaussianNoise", "init_args": {"std": 0.5, "p": 1.0}}]


class Scenes(Dataset[mean_teacher.Unlabeled]):
    """Unlabeled source returning only model inputs."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> mean_teacher.Unlabeled:
        return {"inputs": {"image": torch.full((2, 8, 8), float(index))}}


def test_views_need_only_model_inputs(tmp_path):
    data = _training(tmp_path, augmentations=FLIP, strong_augmentations=NOISE)
    labeled = {"inputs": _batch()["inputs"], "target": torch.ones(2, 1, 8, 8)}

    result = data.on_after_batch_transfer(
        {"labeled": labeled, "unlabeled": {"inputs": _batch(times=2)["inputs"]}}, 0
    )

    assert set(result["unlabeled"]) == {"weak", "strong"}
    assert set(result["labeled"]) == {"inputs", "target"}


def test_strong_augmentations_cannot_move_geometry(tmp_path):
    with pytest.raises(ValueError, match="geometry"):
        _training(tmp_path, strong_augmentations=FLIP)


def _training(tmp_path: Path, **options: object) -> mean_teacher.DataModule:
    """Build only the batch pipelines of a module that is fitting."""
    path = _spec().save(tmp_path / "model_spec.yaml")
    data = mean_teacher.DataModule(
        spec=path, train=Chips(), val=Chips(), unlabeled=Scenes(), **options
    )
    data.trainer = SimpleNamespace(training=True)  # type: ignore[assignment]
    return data


def test_labeled_values_and_unlabeled_views_are_finished_together(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP, strong_augmentations=NOISE)
    image = _batch()["inputs"]["image"]
    values = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    unlabeled = _batch()

    batch = data.on_after_batch_transfer(
        {
            "labeled": {**_batch(), "target": values},
            "unlabeled": {"inputs": unlabeled["inputs"]},
        },
        0,
    )

    views = batch["unlabeled"]
    assert torch.equal(batch["labeled"]["target"], values.flip(-1))
    torch.testing.assert_close(views["weak"]["image"], image.flip(-1))
    assert (views["strong"]["image"] - views["weak"]["image"]).abs().gt(0).any()
    assert set(views) == {"weak", "strong"}


def test_evaluation_batches_are_supervised_ones(tmp_path) -> None:
    data = _training(tmp_path)
    data.trainer = SimpleNamespace(training=False)  # type: ignore[assignment]
    values = torch.ones(2, 1, 8, 8)

    batch = data.on_after_batch_transfer({**_batch(), "target": values}, 0)

    assert torch.equal(batch["target"], values)
