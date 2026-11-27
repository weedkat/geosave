from __future__ import annotations

import pytest
import torch

from geosave.ml.torch.training.metrics import SemanticSegmentationMetrics


@pytest.fixture
def predictions() -> torch.Tensor:
    return torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 2]]])


@pytest.fixture
def targets() -> torch.Tensor:
    return torch.tensor([[[0, 2], [2, 255]], [[1, 0], [0, 2]]])


def test_scores_exclude_ignored_pixels_and_name_classes(
    predictions: torch.Tensor, targets: torch.Tensor
) -> None:
    metrics = SemanticSegmentationMetrics(
        3,
        ignore_index=255,
        labels=["water", "trees", "urban"],
        metrics=["f1_macro", "f1_per_class", "iou_per_class", "accuracy_micro"],
    )
    metrics.update(predictions, targets)

    scores = {name: value.item() for name, value in metrics.compute().items()}

    assert scores == pytest.approx(
        {
            "accuracy_micro": 5 / 7,
            "f1_macro": 32 / 45,
            "f1_water": 4 / 5,
            "f1_trees": 2 / 3,
            "f1_urban": 2 / 3,
            "iou_water": 2 / 3,
            "iou_trees": 1 / 2,
            "iou_urban": 1 / 2,
        }
    )


def test_defaults_score_miou_per_class_iou_pixel_accuracy_and_dice(
    predictions: torch.Tensor, targets: torch.Tensor
) -> None:
    metrics = SemanticSegmentationMetrics(3, ignore_index=255)
    metrics.update(predictions, targets)

    assert set(metrics.compute()) == {
        "iou_macro",
        "iou_0",
        "iou_1",
        "iou_2",
        "accuracy_micro",
        "f1_macro",
    }


def test_unknown_names_list_the_available_metrics() -> None:
    with pytest.raises(ValueError, match="iou_macro"):
        SemanticSegmentationMetrics(3, ignore_index=255, metrics=["iou.macro"])


def test_clones_prefix_classwise_results_and_keep_independent_state(
    predictions: torch.Tensor, targets: torch.Tensor
) -> None:
    metrics = SemanticSegmentationMetrics(3, ignore_index=255, metrics=["f1_per_class"])
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


@pytest.mark.filterwarnings(
    "error:The .*compute.* method of metric .* was called before"
)
def test_macro_and_classwise_iou_reset_between_epochs(predictions, targets):
    metrics = SemanticSegmentationMetrics(
        3, ignore_index=255, metrics=["iou_macro", "iou_per_class", "f1_macro"]
    )
    metrics.update(predictions, targets)
    assert metrics.compute()["iou_macro"].item() == pytest.approx(5 / 9)

    metrics.reset()
    metrics.update(targets, targets)
    assert all(value.item() == 1.0 for value in metrics.compute().values())
