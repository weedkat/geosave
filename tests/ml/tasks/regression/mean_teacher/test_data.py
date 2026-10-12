from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from geosave_engine.ml.tasks.regression import mean_teacher
from tests.ml.tasks.segmentation.supervised.test_data import FLIP, Chips, _batch, _spec

NOISE = [{"name": "RandomGaussianNoise", "init_args": {"std": 0.5, "p": 1.0}}]


def _training(tmp_path: Path, **options: object) -> mean_teacher.DataModule:
    """Build only the batch pipelines of a module that is fitting."""
    path = _spec().save(tmp_path / "model_spec.yaml")
    data = mean_teacher.DataModule(
        spec=path, train=Chips(), val=Chips(), unlabeled=Chips(), **options
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
            "unlabeled": {"inputs": unlabeled["inputs"], "valid": unlabeled["valid"]},
        },
        0,
    )

    views = batch["unlabeled"]
    assert torch.equal(batch["labeled"]["target"], values.flip(-1))
    torch.testing.assert_close(views["weak"]["image"], image.flip(-1))
    assert (views["strong"]["image"] - views["weak"]["image"]).abs().gt(0).any()
    assert views["valid"].all()


def test_evaluation_batches_are_supervised_ones(tmp_path) -> None:
    data = _training(tmp_path)
    data.trainer = SimpleNamespace(training=False)  # type: ignore[assignment]
    values = torch.ones(2, 1, 8, 8)

    batch = data.on_after_batch_transfer({**_batch(), "target": values}, 0)

    assert torch.equal(batch["target"], values)
