from __future__ import annotations

from torchmetrics import Metric, MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


class RegressionMetrics(MetricCollection):
    """Collect regression metrics selected by name.

    Scores flat predictions against flat targets; a caller selects the
    pixels or chips that hold a target first.

    Args:
        metrics: Names among ``mae``, ``rmse`` and ``r2``. None or an empty
            list selects all three.

    Raises:
        ValueError: A metric name is unknown.

    Examples:
        >>> metrics = RegressionMetrics(["mae", "rmse"])
        >>> metrics.update(predictions[known], target[known])
        >>> metrics.compute()
        {'mae': tensor(1.2), 'rmse': tensor(2.3)}
    """

    def __init__(self, metrics: list[str] | None = None) -> None:
        available: dict[str, Metric] = {
            "mae": MeanAbsoluteError(),
            "rmse": MeanSquaredError(squared=False),
            "r2": R2Score(),
        }
        names = metrics or list(available)
        unknown = [name for name in names if name not in available]
        if unknown:
            raise ValueError(
                f"Unknown metrics {unknown}; must be among {list(available)}"
            )
        super().__init__({name: available[name] for name in names})
