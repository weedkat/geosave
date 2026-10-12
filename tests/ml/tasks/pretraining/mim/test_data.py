from __future__ import annotations

from types import SimpleNamespace

import torch

from geosave_engine.ml.tasks.pretraining import mim
from tests.ml.tasks.segmentation.supervised.test_data import (
    FLIP,
    NORMALIZE,
    Chips,
    _batch,
    _spec,
)


def _data(tmp_path, training: bool, **options: object) -> mim.DataModule:
    """Build only the batch pipelines."""
    path = _spec(transforms=NORMALIZE).save(tmp_path / "model_spec.yaml")
    data = mim.DataModule(spec=path, train=Chips(), val=Chips(), **options)
    data.trainer = SimpleNamespace(training=training)  # type: ignore[assignment]
    return data


def _unlabeled(times: int = 0) -> mim.Sample:
    batch = _batch(times)
    return {"inputs": batch["inputs"], "valid": batch["valid"]}


def test_training_batches_are_augmented_then_transformed(tmp_path) -> None:
    data = _data(tmp_path, True, augmentations=FLIP)
    image = _unlabeled(times=3)["inputs"]["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 1, 2, 1, 1)

    batch = data.on_after_batch_transfer(_unlabeled(times=3), 0)

    torch.testing.assert_close(batch["inputs"]["image"], (image.flip(-1) - mean) / std)
    assert batch["valid"].all()


def test_non_finite_pixels_become_zero_and_invalid(tmp_path) -> None:
    data = _data(tmp_path, False, augmentations=FLIP)
    unlabeled = _unlabeled()
    unlabeled["inputs"]["image"][0, 1, 3, 4] = torch.nan

    batch = data.on_after_batch_transfer(unlabeled, 0)

    valid = batch["valid"]
    assert batch["inputs"]["image"][0, 1, 3, 4] == 0
    assert not valid[0, 3, 4]
    assert valid.sum() == valid.numel() - 1
