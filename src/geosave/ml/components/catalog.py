"""List the registered components and which of them connect."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping

import pandas as pd
import torch.nn as nn

from geosave.ml.chain.routing import declared_steps
from geosave.ml.components.published import published_attrs
from geosave.ml.components.registry import COMPONENTS


def list_models(name: str | None = None) -> pd.DataFrame:
    """List the registered models, or what connects to one of them.

    Args:
        name: A model's name, as a recipe spells it. None lists every model.

    Returns:
        With no name, one row per stage and name: `forward`, the step methods
        a chain can call and what they read; `publishes`, what later stages
        can take; and `requires`, the arguments a caller must give. With a
        name, one row per stage holding the models that can run `before` it,
        feeding what it reads and needs, and `after` it, reading what it
        returns.

    Raises:
        KeyError: No model has the name.

    Examples:
        >>> list_models().loc[("head", "segmentation"), "requires"]
        'classes'
        >>> list_models("unet")
                                            before                     after
        encoder  clay, dinov3, prithvi, prithvi_tl
        head                                        regression, segmentation
    """
    # Stages read in chain order, the order COMPONENTS declares them.
    models: dict[tuple[str, str], Callable[..., nn.Module]] = {}
    for stage, registry in COMPONENTS.items():
        for key, factory in registry.items():
            models[(stage, key)] = factory

    published: set[str] = set()
    for factory in models.values():
        published.update(published_attrs(factory))

    if name is None:
        rows = {}
        for model, factory in models.items():
            rows[model] = {
                "forward": _forward(factory),
                "publishes": ", ".join(published_attrs(factory)),
                "requires": ", ".join(_requires(factory, published)),
            }
        table = pd.DataFrame.from_dict(
            rows, orient="index", columns=["forward", "publishes", "requires"]
        )
        table.index = pd.MultiIndex.from_tuples(table.index, names=["stage", "name"])
        return table

    found = [factory for (_, key), factory in models.items() if key == name.casefold()]
    if not found:
        raise KeyError(f"no model named {name!r}; list_models() lists them")
    chosen = found[0]

    connections: dict[str, dict[str, list[str]]] = {}
    for (stage, key), factory in models.items():
        sides = connections.setdefault(stage, {"before": [], "after": []})
        if _feeds(factory, chosen, models, published):
            sides["before"].append(key)
        if _feeds(chosen, factory, models, published):
            sides["after"].append(key)

    rows = {}
    for stage, sides in connections.items():
        if sides["before"] or sides["after"]:
            rows[stage] = {side: ", ".join(keys) for side, keys in sides.items()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["before", "after"])


def _forward(factory: Callable[..., nn.Module]) -> str:
    """Spell each step a model declares as its call, as `forward_pyramid(image)`."""
    calls = []
    for step in declared_steps(factory):
        method = getattr(factory, step.name)
        arguments = []
        for parameter in list(inspect.signature(method).parameters.values())[1:]:
            if parameter.default is inspect.Parameter.empty:
                arguments.append(parameter.name)
            else:
                arguments.append(f"{parameter.name}={parameter.default!r}")
        calls.append(f"{step.name}({', '.join(arguments)})")
    # A lone end-to-end model runs its forward.
    return ", ".join(calls) or "forward(image)"


def _requires(factory: Callable[..., nn.Module], published: set[str]) -> list[str]:
    """Return the constructor arguments only a caller can give."""
    own = published_attrs(factory)
    required = []
    for argument, parameter in inspect.signature(factory).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        if argument in published and argument not in own:
            continue
        required.append(argument)
    return required


def _reads_from(
    first: Callable[..., nn.Module], then: Callable[..., nn.Module]
) -> bool:
    """Say whether `then` reads something `first` returns."""
    returns: set[str] = set()
    for step in declared_steps(first):
        returns.update(step.outputs)
    reads: set[str] = set()
    for step in declared_steps(then):
        reads.update(step.inputs)
    return bool(returns & reads)


def _feeds(
    first: Callable[..., nn.Module],
    then: Callable[..., nn.Module],
    models: Mapping[tuple[str, str], Callable[..., nn.Module]],
    published: set[str],
) -> bool:
    """Say whether `then` can run right after `first`.

    It reads something `first` returns, and every argument it must take from
    earlier stages is published by `first` or by a stage that can run before it.
    """
    if not _reads_from(first, then):
        return False

    # Published values reach every later stage, so walk back from `first`.
    given: set[str] = set()
    pending, seen = [first], []
    while pending:
        stage = pending.pop()
        if any(stage is visited for visited in seen):
            continue
        seen.append(stage)
        given.update(published_attrs(stage))
        pending.extend(m for m in models.values() if _reads_from(m, stage))

    own = published_attrs(then)
    for argument, parameter in inspect.signature(then).parameters.items():
        taken = argument in published and argument not in own
        required = parameter.default is inspect.Parameter.empty
        if taken and required and argument not in given:
            return False
    return True
