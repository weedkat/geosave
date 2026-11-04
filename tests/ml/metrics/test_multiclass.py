from __future__ import annotations

import pytest
import torch

from geosave.ml.metrics import MulticlassMetrics


@pytest.fixture
def predictions() -> torch.Tensor:
    return torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 2]]])


@pytest.fixture
def targets() -> torch.Tensor:
    return torch.tensor([[[0, 2], [2, 255]], [[1, 0], [0, 2]]])


def test_aggregations_and_class_labels_exclude_ignored_pixels(
    predictions: torch.Tensor, targets: torch.Tensor
) -> None:
    metrics = MulticlassMetrics(
        3,
        ignore_index=255,
        labels={"0": "water", "1": "trees", "2": "urban"},
        metrics=[
            "f1.macro.micro.per_class",
            "iou.per_class",
            "accuracy.micro",
            "kappa",
        ],
    )
    metrics.update(predictions, targets)

    scores = {name: value.item() for name, value in metrics.compute().items()}

    assert scores == pytest.approx(
        {
            "accuracy_micro": 5 / 7,
            "f1_macro": 32 / 45,
            "f1_micro": 5 / 7,
            "f1_water": 4 / 5,
            "f1_trees": 2 / 3,
            "f1_urban": 2 / 3,
            "iou_water": 2 / 3,
            "iou_trees": 1 / 2,
            "iou_urban": 1 / 2,
            "kappa": 9 / 16,
        }
    )


def test_clones_prefix_classwise_results_and_keep_independent_state(
    predictions: torch.Tensor, targets: torch.Tensor
) -> None:
    metrics = MulticlassMetrics(3, ignore_index=255, metrics=["f1.per_class"])
    train = metrics.clone(prefix="train_")
    validation = metrics.clone(prefix="val_")
    train.update(predictions, targets)
    validation.update(targets, targets)

    assert {
        name: value.item() for name, value in train.compute().items()
    } == pytest.approx({"train_f1_0": 4 / 5, "train_f1_1": 2 / 3, "train_f1_2": 2 / 3})
    assert {name: value.item() for name, value in validation.compute().items()} == {
        "val_f1_0": 1.0,
        "val_f1_1": 1.0,
        "val_f1_2": 1.0,
    }

    train.reset()
    train.update(targets, targets)
    assert all(value.item() == 1.0 for value in train.compute().values())
