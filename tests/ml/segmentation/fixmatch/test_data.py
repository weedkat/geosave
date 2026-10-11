from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from geosave_engine.ml.segmentation import fixmatch
from tests.ml.segmentation.supervised.test_data import FLIP, _batch, _spec

NOISE = [{"name": "RandomGaussianNoise", "init_args": {"std": 0.5, "p": 1.0}}]


def _training(tmp_path: Path, **options: object) -> fixmatch.DataModule:
    """Build only the batch pipelines of a module that is fitting."""
    path = _spec().save(tmp_path / "model_spec.yaml")
    data = fixmatch.DataModule(
        spec=path, train=tmp_path, val=tmp_path, unlabeled=tmp_path, **options
    )
    data.setup("predict")
    data.trainer = SimpleNamespace(training=True)  # type: ignore[assignment]
    return data


def _unlabeled(times: int = 0):
    model_inputs, _, ids, valid = _batch(times)
    return model_inputs, ids, valid


def test_strong_views_perturb_the_weak_view_without_moving_it(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP, strong_augmentations=NOISE)
    model_inputs, _, _ = _unlabeled()

    batch = data.on_after_batch_transfer(
        {"labeled": _batch(), "unlabeled": _unlabeled()}, 0
    )

    weak, strong, valid = batch["unlabeled"]
    torch.testing.assert_close(weak["image"], model_inputs["image"].flip(-1))
    noise = strong["image"] - weak["image"]
    assert noise.abs().gt(0).any()
    assert noise.abs().mean() < 2.0
    assert torch.equal(weak["location_coords"], model_inputs["location_coords"])
    assert valid.all()


def test_labeled_chips_are_finished_as_in_supervised_training(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP)
    model_inputs, target, _, _ = _batch()

    batch = data.on_after_batch_transfer(
        {"labeled": _batch(), "unlabeled": _unlabeled()}, 0
    )

    image, labels, _, _ = batch["labeled"]
    torch.testing.assert_close(image["image"], model_inputs["image"].flip(-1))
    assert torch.equal(labels, target.flip(-1))


def test_pixels_an_augmentation_invents_are_not_valid(tmp_path) -> None:
    shift = [
        {
            "name": "RandomAffine",
            "init_args": {"degrees": 0.0, "translate": [0.5, 0.5], "p": 1.0},
        }
    ]
    data = _training(tmp_path, augmentations=shift)

    batch = data.on_after_batch_transfer(
        {"labeled": _batch(), "unlabeled": _unlabeled(times=2)}, 0
    )

    _, strong, valid = batch["unlabeled"]
    assert valid.any() and not valid.all()
    assert strong["image"].shape == (2, 2, 2, 8, 8)


def test_evaluation_batches_pass_as_supervised(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP)
    data.trainer = SimpleNamespace(training=False)  # type: ignore[assignment]
    model_inputs, target, _, _ = _batch()

    image, labels, _, _ = data.on_after_batch_transfer(_batch(), 0)

    torch.testing.assert_close(image["image"], model_inputs["image"])
    assert torch.equal(labels, target)
