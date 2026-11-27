"""Configure Lightning optimization for a model with named modules."""

from collections.abc import Mapping
from typing import Any

from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn
from torch.optim import Optimizer


def configure_optimizers(
    model: nn.Module,
    build_optimizer: OptimizerCallable,
    param_groups: Mapping[str, Mapping[str, Any]] | None,
    build_scheduler: LRSchedulerCallable | None,
    lr_scheduler_config: Mapping[str, Any] | None,
) -> Optimizer | dict[str, Any]:
    """Build an optimizer over the model's trainable parameters and an optional schedule.

    Args:
        model: The trained network. Parameter groups select its named children.
        build_optimizer: Builds the optimizer from the parameter groups, as
            LightningCLI gives it from `class_path` and `init_args`.
        param_groups: Options per named child, such as `{"encoder": {"lr": 1e-4}}`;
            the remaining trainable parameters form one more group.
        build_scheduler: Builds the scheduler from the optimizer. None keeps
            the learning rate constant.
        lr_scheduler_config: Lightning's scheduler options, such as
            `{"interval": "step"}`, beside the built scheduler.

    Returns:
        A native optimizer or Lightning's optimizer and scheduler configuration.

    Raises:
        ValueError: A group sets `params` or shares parameters with an earlier group.

    Examples:
        >>> configure_optimizers(
        ...     model, partial(torch.optim.AdamW, lr=1e-3), {"encoder": {"lr": 1e-4}}, None, None
        ... )
        AdamW (...)
    """
    modules = dict(model.named_children())
    groups = []
    grouped = set()
    for name, options in (param_groups or {}).items():
        parameters = [p for p in modules[name].parameters() if p.requires_grad]
        if "params" in options or grouped.intersection(parameters):
            raise ValueError(f"optimizer group {name!r} overrides or shares parameters")
        if parameters:
            groups.append({"params": parameters, **options})
            grouped.update(parameters)
    remaining = [p for p in model.parameters() if p.requires_grad and p not in grouped]
    if remaining:
        groups.append({"params": remaining})

    optimizer = build_optimizer(groups)
    if build_scheduler is None:
        return optimizer
    schedule = {"scheduler": build_scheduler(optimizer), **(lr_scheduler_config or {})}
    return {"optimizer": optimizer, "lr_scheduler": schedule}
