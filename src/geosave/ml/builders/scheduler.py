from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

from geosave.ml.factory import BuildSpec, Registry


class SchedulerSpec(BuildSpec):
    """Configure a scheduler and its Lightning metadata.

    Attributes:
        interval: Update the scheduler after optimizer steps (`step`) or
            epochs (`epoch`). None leaves Lightning's default interval.
        frequency: Number of configured intervals between scheduler updates.
            None leaves Lightning's default frequency.
        monitor: Logged metric used by a metric-driven scheduler, such as
            `val_loss` for ReduceLROnPlateau.
        strict: Whether Lightning raises if the monitored metric is absent.
            None leaves Lightning's default behavior.
        scheduler_name: Display name forwarded as Lightning's scheduler
            `name`, for learning-rate monitoring. None omits the override.
    """

    interval: Literal["step", "epoch"] | None = None
    frequency: int | None = None
    monitor: str | None = None
    strict: bool | None = None
    scheduler_name: str | None = None


SCHEDULERS = Registry(LRScheduler)
SCHEDULERS.register("cosine_annealing")(CosineAnnealingLR)
SCHEDULERS.register("reduce_on_plateau")(ReduceLROnPlateau)
SCHEDULERS.register("step")(StepLR)


def build_scheduler(
    spec: SchedulerSpec | Mapping[str, Any],
    optimizer: Optimizer,
) -> dict[str, Any]:
    """Construct a scheduler and its Lightning configuration mapping.

    Args:
        spec: Scheduler selector, arguments, and Lightning metadata.
        optimizer: Optimizer controlled by the scheduler.

    Returns:
        Lightning scheduler configuration.
    """
    configured = SchedulerSpec.model_validate(spec)
    scheduler = SCHEDULERS.build(configured, optimizer)
    metadata = configured.model_dump(
        include={"interval", "frequency", "monitor", "strict"},
        exclude_none=True,
    )
    if configured.scheduler_name is not None:
        metadata["name"] = configured.scheduler_name
    return {"scheduler": scheduler, **metadata}
