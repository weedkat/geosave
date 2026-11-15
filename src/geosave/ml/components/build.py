"""Build a model chain from a recipe of registered components."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import torch.nn as nn

from geosave.ml.chain import ModelChain
from geosave.ml.components.published import published_kwargs
from geosave.ml.components.registry import COMPONENTS
from geosave.ml.factory import BuildSpec, Registry, bind_init_args


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

        # Select the constructor: a class as given, or a spec's name or class path.
        if isinstance(value, type):
            if not issubclass(value, nn.Module):
                raise TypeError(f"Stage {stage!r} must be an nn.Module subclass")
            # The class is built as given: one defined in a function has a
            # path no import reaches.
            factory: Callable[..., nn.Module] = value
            spec = BuildSpec(class_path=f"{value.__module__}.{value.__qualname__}")
        else:
            spec = BuildSpec.model_validate(value)
            registry = COMPONENTS.get(stage, Registry(nn.Module))
            factory = registry.resolve(spec)

        # Build it with what earlier stages publish, recording every argument.
        try:
            kwargs = published_kwargs(factory, modules, spec.init_args)
            init_args = bind_init_args(factory, kwargs)
            modules[stage] = factory(**kwargs)
        except Exception as error:
            error.add_note(f"While building stage {stage!r}")
            raise

        # Weights come from the release, so a rebuild skips the download.
        if "pretrained" in inspect.signature(factory).parameters:
            init_args["pretrained"] = False
        recipe[stage] = {**spec.model_dump(exclude_none=True), "init_args": init_args}

    model = ModelChain(**modules)
    model._prebuilt = tuple(prebuilt)
    if not prebuilt:
        model._stage_specs = deepcopy(recipe)
    return model
