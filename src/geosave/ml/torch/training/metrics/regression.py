from __future__ import annotations

from collections.abc import Sequence

from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from .base import TaskMetrics


class RegressionMetrics(TaskMetrics):
    """Score predicted values against target values.

    Updates take flat predictions and targets; a caller selects the pixels or
    chips that hold a target first.

    Args:
        metrics: Names among `mae`, `rmse` and `r2`. None or an empty list
            selects `defaults`.

    Raises:
        ValueError: A metric name is unknown.

    Examples:
        >>> metrics = RegressionMetrics(["mae", "rmse"])
        >>> metrics.update(predictions[known], target[known])
        >>> metrics.compute()
        {'mae': tensor(1.2), 'rmse': tensor(2.3)}
    """

    defaults = ("mae", "rmse", "r2")

    def __init__(self, metrics: Sequence[str] | None = None) -> None:
        available = {
            "mae": MeanAbsoluteError(),
            "rmse": MeanSquaredError(squared=False),
            "r2": R2Score(),
        }
        super().__init__(available, metrics)
