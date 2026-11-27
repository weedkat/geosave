"""Values a module offers the modules built after it."""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Mapping
from typing import Annotated, Any, TypeVar, get_args, get_origin, get_type_hints

import torch.nn as nn

_PUBLISHED = "geosave.published"

_T = TypeVar("_T")

Published = Annotated[_T, _PUBLISHED]
"""Mark a class attribute as readable by later modules during construction.

Annotate at class level; the runtime value stays a plain ``T``. A module
built later whose constructor takes a parameter of the same name and type,
and that the configuration omits, receives it from the nearest earlier module
that publishes it.

Examples:
    >>> class DINOv3(nn.Module):
    ...     pyramid_channels: Published[list[int]]
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.pyramid_channels = [768] * 4
"""


def read_published(cls: type[nn.Module]) -> dict[str, Any]:
    """Return the attributes a class and its bases mark with `Published`.

    Args:
        cls: Module class to inspect.

    Returns:
        Each attribute's name mapped to its declared type, as
        `typing.get_type_hints` maps them, inherited marks included.
    """
    published = {}
    for name, hint in get_type_hints(cls, include_extras=True).items():
        if get_origin(hint) is Annotated and _PUBLISHED in get_args(hint)[1:]:
            published[name] = get_args(hint)[0]
    return published


def accepts(published: Any, parameter: Any) -> bool:
    """Say whether a constructor parameter's annotation takes a published type.

    Args:
        published: Type declared with `Published` on the source attribute.
        parameter: Receiving parameter's annotation, optionally including None.

    Returns:
        Whether the types match after removing None from the parameter's type.
    """
    if get_origin(parameter) not in (types.UnionType, typing.Union):
        return parameter == published

    members = set(get_args(parameter)) - {type(None)}
    if get_origin(published) in (types.UnionType, typing.Union):
        return members == set(get_args(published))
    return members == {published}


def fill_published_arguments(
    cls: type[nn.Module],
    modules: Mapping[str, nn.Module],
    explicit: Mapping[str, object],
) -> dict[str, object]:
    """Fill omitted constructor arguments from the nearest earlier module publishing them.

    In a pipe the nearest earlier module is where a value flowed from: a head
    takes `input_size` from the encoder when the decoder between them
    publishes none.

    Args:
        cls: Class to construct.
        modules: The chain's earlier modules, in calling order.
        explicit: Arguments the configuration gives, which take precedence.

    Returns:
        The explicit arguments plus each omitted one an earlier module
        publishes.

    Raises:
        TypeError: A published type does not match the parameter's.
    """
    arguments = dict(explicit)
    hints = get_type_hints(cls.__init__)

    for parameter in inspect.signature(cls).parameters:
        if parameter in arguments:
            continue

        for name, module in reversed(list(modules.items())):
            published = read_published(type(module))
            if parameter not in published:
                continue

            annotation = hints.get(parameter, published[parameter])
            if not accepts(published[parameter], annotation):
                raise TypeError(
                    f"Argument {parameter!r} is {annotation}, but {name!r} "
                    f"publishes {published[parameter]}; set it in init_args"
                )

            arguments[parameter] = getattr(module, parameter)
            break

    return arguments
