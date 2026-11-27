from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from torchmetrics import Metric, MetricCollection


class TaskMetrics(MetricCollection):
    """Collect the metrics of one task, selected by name.

    A task builds every metric it offers; the names it is given select which
    ones to score. A selected name is also the key its result is logged under.

    Args:
        available: Every metric the task offers, keyed by name.
        metrics: Names to score. None or an empty list selects `defaults`.

    Raises:
        ValueError: A name is not among `available`.
    """

    defaults: ClassVar[tuple[str, ...]]

    def __init__(
        self, available: dict[str, Metric], metrics: Sequence[str] | None
    ) -> None:
        selected = {}
        for name in metrics or self.defaults:
            if name not in available:
                raise ValueError(
                    f"Unknown metric {name!r}; must be one of {list(available)}"
                )
            selected[name] = available[name]

        # ClasswiseWrapper owns nested update state that compute groups do not reset.
        super().__init__(selected, compute_groups=False)
