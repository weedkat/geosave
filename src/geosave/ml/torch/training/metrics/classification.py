from __future__ import annotations

from collections.abc import Sequence

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.wrappers import ClasswiseWrapper

from .base import TaskMetrics


class ClassificationMetrics(TaskMetrics):
    """Score one class per sample, such as one land-cover class per chip.

    Updates take logits shaped `(B, C)` or class IDs shaped `(B,)`, and
    targets shaped `(B,)`. `auroc_macro` needs logits.

    Args:
        num_classes: Number of output classes.
        labels: Class names in class-ID order, naming per-class results.
            None names them by class ID.
        metrics: Names among `accuracy_micro`, `accuracy_macro`, `f1_macro`,
            `f1_per_class`, `precision_macro`, `recall_macro`, `kappa`, `mcc`
            and `auroc_macro`. None or an empty list selects `defaults`.

    Raises:
        ValueError: A metric name is unknown.

    Examples:
        >>> metrics = ClassificationMetrics(10, metrics=["accuracy_macro", "kappa"])
        >>> metrics.update(logits, targets)
        >>> metrics.compute()
        {'accuracy_macro': tensor(0.72), 'kappa': tensor(0.66)}
    """

    defaults = ("accuracy_micro", "accuracy_macro", "f1_macro", "kappa")

    def __init__(
        self,
        num_classes: int,
        labels: Sequence[str] | None = None,
        metrics: Sequence[str] | None = None,
    ) -> None:
        names = None if labels is None else list(labels)
        available = {
            "accuracy_micro": MulticlassAccuracy(num_classes, average="micro"),
            "accuracy_macro": MulticlassAccuracy(num_classes, average="macro"),
            "f1_macro": MulticlassF1Score(num_classes, average="macro"),
            "f1_per_class": ClasswiseWrapper(
                MulticlassF1Score(num_classes, average=None), labels=names, prefix="f1_"
            ),
            "precision_macro": MulticlassPrecision(num_classes, average="macro"),
            "recall_macro": MulticlassRecall(num_classes, average="macro"),
            "kappa": MulticlassCohenKappa(num_classes),
            "mcc": MulticlassMatthewsCorrCoef(num_classes),
            "auroc_macro": MulticlassAUROC(num_classes, average="macro"),
        }
        super().__init__(available, metrics)
