from __future__ import annotations

from collections.abc import Sequence

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.wrappers import ClasswiseWrapper

from .base import TaskMetrics


class SemanticSegmentationMetrics(TaskMetrics):
    """Score per-pixel class scores against per-pixel class IDs.

    Updates take logits or class IDs shaped `(B, C, H, W)` or `(B, H, W)`,
    and targets shaped `(B, H, W)`. Pixels holding `ignore_index` are skipped.

    Args:
        num_classes: Number of output classes.
        ignore_index: Target class ID excluded from every metric.
        labels: Class names in class-ID order, naming per-class results.
            None names them by class ID.
        metrics: Names among `iou_macro`, `iou_per_class`, `accuracy_micro`
            (pixel accuracy), `accuracy_macro`, `f1_macro` (Dice), `f1_per_class`,
            `precision_macro` and `recall_macro`. None or an empty list selects
            `defaults`.

    Raises:
        ValueError: A metric name is unknown.

    Examples:
        >>> metrics = SemanticSegmentationMetrics(
        ...     9, ignore_index=255, metrics=["iou_macro", "iou_per_class"]
        ... )
        >>> metrics.update(logits, targets)
        >>> metrics.compute()
        {'iou_macro': tensor(0.41), 'iou_0': tensor(0.83), ...}
    """

    defaults = ("iou_macro", "iou_per_class", "accuracy_micro", "f1_macro")

    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        labels: Sequence[str] | None = None,
        metrics: Sequence[str] | None = None,
    ) -> None:
        names = None if labels is None else list(labels)
        setup = {"num_classes": num_classes, "ignore_index": ignore_index}
        available = {
            "iou_macro": MulticlassJaccardIndex(**setup, average="macro"),
            "iou_per_class": ClasswiseWrapper(
                MulticlassJaccardIndex(**setup, average=None),
                labels=names,
                prefix="iou_",
            ),
            "accuracy_micro": MulticlassAccuracy(**setup, average="micro"),
            "accuracy_macro": MulticlassAccuracy(**setup, average="macro"),
            "f1_macro": MulticlassF1Score(**setup, average="macro"),
            "f1_per_class": ClasswiseWrapper(
                MulticlassF1Score(**setup, average=None), labels=names, prefix="f1_"
            ),
            "precision_macro": MulticlassPrecision(**setup, average="macro"),
            "recall_macro": MulticlassRecall(**setup, average="macro"),
        }
        super().__init__(available, metrics)
