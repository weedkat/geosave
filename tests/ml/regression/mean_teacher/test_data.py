from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from geosave_engine.ml.regression import mean_teacher
from tests.ml.segmentation.supervised.test_data import FLIP, _batch, _spec

NOISE = [{"name": "RandomGaussianNoise", "init_args": {"std": 0.5, "p": 1.0}}]


def _training(tmp_path: Path, **options: object) -> mean_teacher.DataModule:
    """Build only the batch pipelines of a module that is fitting."""
    path = _spec().save(tmp_path / "model_spec.yaml")
    data = mean_teacher.DataModule(
        spec=path, train=tmp_path, val=tmp_path, unlabeled=tmp_path, **options
    )
    data.setup("predict")
    data.trainer = SimpleNamespace(training=True)  # type: ignore[assignment]
    return data


def test_labeled_values_and_unlabeled_views_are_finished_together(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP, strong_augmentations=NOISE)
    model_inputs, _, ids, valid = _batch()
    values = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)

    batch = data.on_after_batch_transfer(
        {
            "labeled": (model_inputs, values, ids, valid),
            "unlabeled": (model_inputs, ids, valid),
        },
        0,
    )

    _, moved, _, _ = batch["labeled"]
    weak, strong, unlabeled_valid = batch["unlabeled"]
    assert torch.equal(moved, values.flip(-1))
    torch.testing.assert_close(weak["image"], model_inputs["image"].flip(-1))
    assert (strong["image"] - weak["image"]).abs().gt(0).any()
    assert unlabeled_valid.all()


def test_evaluation_batches_are_supervised_ones(tmp_path) -> None:
    data = _training(tmp_path)
    data.trainer = SimpleNamespace(training=False)  # type: ignore[assignment]
    model_inputs, _, ids, valid = _batch()
    values = torch.ones(2, 1, 8, 8)

    _, finished, _, _ = data.on_after_batch_transfer(
        (model_inputs, values, ids, valid), 0
    )

    assert torch.equal(finished, values)
