from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import pandas as pd
import torch.nn as nn

from geosave.model.chain import ModelChain
from geosave.model.chain.published import published_attrs, published_kwargs
from geosave.model.chain.routing import declared_steps
from geosave.model.factory import BuildSpec

MODEL_REGISTRY: dict[str, dict[str, Callable[..., nn.Module]]] = {}


def register_model[Factory: Callable[..., nn.Module]](
    stage: str, name: str
) -> Callable[[Factory], Factory]:
    """Register a model factory under a stage and name.

    Args:
        stage: Stage the factory fills, such as encoder, decoder, head, or model.
        name: Case-insensitive name used in construction specifications.

    Returns:
        Decorator that registers and returns the class or factory unchanged.

    Raises:
        ValueError: The stage and name already identify another factory. A
            factory of the same module and qualified name, as a notebook cell
            run again defines, replaces the one registered.
    """

    def decorator(factory: Factory) -> Factory:
        key = name.upper()
        existing = MODEL_REGISTRY.get(stage, {}).get(key)
        # A notebook cell run again defines a new class under the same path.
        redefined = existing is not None and _path(existing) == _path(factory)
        if existing is not None and existing is not factory and not redefined:
            raise ValueError(f"{stage!r}/{name!r} already identifies another factory")
        MODEL_REGISTRY.setdefault(stage, {})[key] = factory
        return factory

    return decorator


def _path(factory: Callable[..., nn.Module]) -> tuple[str, str]:
    """Return where a factory is defined, as its module and qualified name."""
    return factory.__module__, factory.__qualname__


# Stages a chain runs in order; other stages follow alphabetically.
_CHAIN_ORDER = ("encoder", "decoder", "head", "model")


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
    from geosave.model.stages import decoder, encoder, head, monolith  # noqa: F401

    stages = sorted(
        MODEL_REGISTRY,
        key=lambda stage: (
            _CHAIN_ORDER.index(stage) if stage in _CHAIN_ORDER else len(_CHAIN_ORDER),
            stage,
        ),
    )
    models = {
        (stage, key.lower()): factory
        for stage in stages
        for key, factory in MODEL_REGISTRY[stage].items()
    }
    published = set()
    for factory in models.values():
        published.update(published_attrs(factory))

    if name is None:
        rows = {
            model: {
                "forward": _forward(factory),
                "publishes": ", ".join(published_attrs(factory)),
                "requires": ", ".join(_requires(factory, published)),
            }
            for model, factory in models.items()
        }
        table = pd.DataFrame.from_dict(
            rows, orient="index", columns=["forward", "publishes", "requires"]
        )
        table.index = pd.MultiIndex.from_tuples(table.index, names=["stage", "name"])
        return table

    found = [factory for (_, key), factory in models.items() if key == name.lower()]
    if not found:
        raise KeyError(f"no model named {name!r}; list_models() lists them")
    (chosen,) = found[:1]
    connections: dict[str, dict[str, list[str]]] = {}
    for (stage, key), factory in models.items():
        if _feeds(factory, chosen, models, published):
            connections.setdefault(stage, {"before": [], "after": []})["before"].append(
                key
            )
        if _feeds(chosen, factory, models, published):
            connections.setdefault(stage, {"before": [], "after": []})["after"].append(
                key
            )
    return pd.DataFrame.from_dict(
        {
            stage: {side: ", ".join(keys) for side, keys in sides.items()}
            for stage, sides in connections.items()
        },
        orient="index",
        columns=["before", "after"],
    )


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
    returns = {output for step in declared_steps(first) for output in step.outputs}
    reads = {name for step in declared_steps(then) for name in step.inputs}
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


def build_model(
    stages: Mapping[str, BuildSpec | Mapping[str, Any] | type[nn.Module] | nn.Module],
) -> ModelChain:
    """Construct a reproducible model chain from ordered stage specifications.

    Args:
        stages: Each stage as a registered name or class path with its
            arguments, as the module class itself, built with its defaults and
            whatever earlier stages publish, or as a module already built.

    Returns:
        Model chain with an independent resolved construction recipe, or with
        none where a stage arrived already built: its arguments are not known.

    Raises:
        ValueError: No stage is given.
        TypeError: A stage given as a class is not an `nn.Module` subclass.

    Examples:
        >>> model = build_model(
        ...     {"encoder": Encoder(channels=32), "decoder": Decoder, "head": Head}
        ... )
        >>> model(image=torch.randn(1, 3, 64, 64)).shape
        torch.Size([1, 11, 64, 64])
    """
    from geosave.model.stages import decoder, encoder, head, monolith  # noqa: F401

    if not stages:
        raise ValueError("Supply at least one model stage")

    modules: dict[str, nn.Module] = {}
    recipe: dict[str, dict[str, Any]] = {}
    prebuilt: list[str] = []
    for stage, value in stages.items():
        if isinstance(value, nn.Module):
            modules[stage] = value
            prebuilt.append(stage)
            continue
        if isinstance(value, type):
            if not issubclass(value, nn.Module):
                raise TypeError(f"Stage {stage!r} must be an nn.Module subclass")
            # The class is built as given: one defined in a notebook or a
            # function has a path no import reaches.
            factory: Callable[..., nn.Module] = value
            spec = BuildSpec(class_path=f"{value.__module__}.{value.__qualname__}")
        else:
            spec = (
                value
                if isinstance(value, BuildSpec)
                else BuildSpec.model_validate(value)
            )
            factory = spec.resolve(MODEL_REGISTRY.get(stage, {}), nn.Module)
        try:
            kwargs = published_kwargs(factory, modules, spec.init_args)
            signature = inspect.signature(factory)
            bound = signature.bind(**kwargs)
            bound.apply_defaults()
            init_args: dict[str, object] = {}
            for name, argument in bound.arguments.items():
                kind = signature.parameters[name].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    init_args.update(argument)
                elif kind is not inspect.Parameter.VAR_POSITIONAL:
                    init_args[name] = argument
            init_args = deepcopy(init_args)
            modules[stage] = factory(**kwargs)
        except Exception as error:
            error.add_note(f"While building stage {stage!r}")
            raise

        if "pretrained" in signature.parameters:
            init_args["pretrained"] = False
        recipe[stage] = {
            **spec.model_dump(exclude_none=True),
            "init_args": init_args,
        }

    model = ModelChain(**modules)
    model._prebuilt = tuple(prebuilt)
    if not prebuilt:
        model._stage_specs = deepcopy(recipe)
    return model
