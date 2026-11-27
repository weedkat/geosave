from __future__ import annotations

import pytest
import torch

from geosave.ml.torch.training.metrics import ClassificationMetrics


def test_scores_one_class_per_sample() -> None:
    metrics = ClassificationMetrics(
        3,
        labels=["water", "trees", "urban"],
        metrics=["accuracy_micro", "f1_per_class", "kappa"],
    )
    predictions = torch.tensor([0, 1, 2, 2])
    targets = torch.tensor([0, 1, 1, 2])
    metrics.update(predictions, targets)

    scores = {name: value.item() for name, value in metrics.compute().items()}

    assert scores == pytest.approx(
        {
            "accuracy_micro": 3 / 4,
            "f1_water": 1.0,
            "f1_trees": 2 / 3,
            "f1_urban": 2 / 3,
            "kappa": 0.6363636,
        }
    )


def test_auroc_scores_logits() -> None:
    metrics = ClassificationMetrics(3, metrics=["auroc_macro"])
    logits = torch.eye(3)
    metrics.update(logits, torch.tensor([0, 1, 2]))

    assert metrics.compute()["auroc_macro"].item() == pytest.approx(1.0)
