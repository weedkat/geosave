from __future__ import annotations

from types import SimpleNamespace

import torch

from geosave_engine.ml.pretraining import mim
from tests.ml.segmentation.supervised.test_data import FLIP, NORMALIZE, _batch, _spec


def _data(tmp_path, training: bool, **options: object) -> mim.DataModule:
    """Build only the batch pipelines."""
    path = _spec(transforms=NORMALIZE).save(tmp_path / "model_spec.yaml")
    data = mim.DataModule(spec=path, train=tmp_path, val=tmp_path, **options)
    data.setup("predict")
    data.trainer = SimpleNamespace(training=training)  # type: ignore[assignment]
    return data


def _unlabeled(times: int = 0):
    model_inputs, _, ids, valid = _batch(times)
    return model_inputs, ids, valid


def test_training_batches_are_augmented_then_transformed(tmp_path) -> None:
    data = _data(tmp_path, True, augmentations=FLIP)
    image = _unlabeled(times=3)[0]["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 1, 2, 1, 1)

    finished, ids, valid = data.on_after_batch_transfer(_unlabeled(times=3), 0)

    torch.testing.assert_close(finished["image"], (image.flip(-1) - mean) / std)
    assert valid.all()
    assert torch.equal(ids, torch.tensor([4, 9]))


def test_non_finite_pixels_become_zero_and_invalid(tmp_path) -> None:
    data = _data(tmp_path, False, augmentations=FLIP)
    model_inputs, ids, valid = _unlabeled()
    model_inputs["image"][0, 1, 3, 4] = torch.nan

    finished, _, valid = data.on_after_batch_transfer((model_inputs, ids, valid), 0)

    assert finished["image"][0, 1, 3, 4] == 0
    assert not valid[0, 3, 4]
    assert valid.sum() == valid.numel() - 1
