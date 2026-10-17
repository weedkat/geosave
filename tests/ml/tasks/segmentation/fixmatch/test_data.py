from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from geosave.ml.tasks.segmentation import fixmatch
from tests.ml.tasks.segmentation.supervised.test_data import FLIP, Chips, _batch, _spec

NOISE = [{"name": "RandomGaussianNoise", "init_args": {"std": 0.5, "p": 1.0}}]


class Scenes(Dataset[fixmatch.Unlabeled]):
    """Unlabeled source returning only model inputs."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> fixmatch.Unlabeled:
        return {"inputs": {"image": torch.full((2, 8, 8), float(index))}}


def test_views_need_only_model_inputs(tmp_path):
    data = _training(tmp_path, augmentations=FLIP, strong_augmentations=NOISE)
    labeled = _batch()

    result = data.on_after_batch_transfer(
        {"labeled": labeled, "unlabeled": {"inputs": _batch(times=2)["inputs"]}}, 0
    )

    assert set(result["unlabeled"]) == {"weak", "strong"}
    assert set(result["labeled"]) == {"inputs", "target"}


def test_strong_augmentations_cannot_move_geometry(tmp_path):
    with pytest.raises(ValueError, match="geometry"):
        _training(tmp_path, strong_augmentations=FLIP)


def _training(tmp_path: Path, **options: object) -> fixmatch.DataModule:
    """Build only the batch pipelines of a module that is fitting."""
    path = _spec().save(tmp_path / "model_spec.yaml")
    data = fixmatch.DataModule(
        spec=path, train=Chips(), val=Chips(), unlabeled=Scenes(), **options
    )
    data.trainer = SimpleNamespace(training=True)  # type: ignore[assignment]
    return data


def _unlabeled(times: int = 0) -> fixmatch.Unlabeled:
    batch = _batch(times)
    return {"inputs": batch["inputs"]}


def test_strong_views_perturb_the_weak_view_without_moving_it(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP, strong_augmentations=NOISE)
    inputs = _unlabeled()["inputs"]

    batch = data.on_after_batch_transfer(
        {"labeled": _batch(), "unlabeled": _unlabeled()}, 0
    )

    views = batch["unlabeled"]
    weak, strong = views["weak"], views["strong"]
    torch.testing.assert_close(weak["image"], inputs["image"].flip(-1))
    noise = strong["image"] - weak["image"]
    assert noise.abs().gt(0).any()
    assert noise.abs().mean() < 2.0
    assert torch.equal(weak["location_coords"], inputs["location_coords"])
    assert set(views) == {"weak", "strong"}


def test_labeled_chips_are_finished_as_in_supervised_training(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP)
    original = _batch()

    batch = data.on_after_batch_transfer(
        {"labeled": _batch(), "unlabeled": _unlabeled()}, 0
    )

    labeled = batch["labeled"]
    image = original["inputs"]["image"]
    torch.testing.assert_close(labeled["inputs"]["image"], image.flip(-1))
    assert torch.equal(labeled["target"], original["target"].flip(-1))


def test_shared_geometry_is_applied_before_splitting_views(tmp_path) -> None:
    crop = [{"name": "CenterCrop", "init_args": {"size": (4, 4)}}]
    data = _training(tmp_path, augmentations=crop)
    original = _unlabeled(times=2)

    result = data.on_after_batch_transfer(
        {"labeled": _batch(), "unlabeled": original}, 0
    )

    for view in ("weak", "strong"):
        torch.testing.assert_close(
            result["unlabeled"][view]["image"],
            original["inputs"]["image"][..., 2:6, 2:6],
        )


def test_evaluation_batches_pass_as_supervised(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP)
    data.trainer = SimpleNamespace(training=False)  # type: ignore[assignment]
    original = _batch()

    batch = data.on_after_batch_transfer(_batch(), 0)

    torch.testing.assert_close(batch["inputs"]["image"], original["inputs"]["image"])
    assert torch.equal(batch["target"], original["target"])


def test_fitting_reads_labeled_and_unlabeled_loaders_side_by_side(tmp_path) -> None:
    loaders = _training(tmp_path, num_workers=0).train_dataloader()

    assert set(loaders) == {"labeled", "unlabeled"}
