"""List the registered modules and which of them connect."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pandas as pd
import torch.nn as nn

from .chain import check_chain, format_forward, read_forward_inputs, read_returns
from .published import read_published
from .registry import MODULES


@dataclass(frozen=True)
class ModuleRow:
    """One row of the module table."""

    family: str
    name: str
    cls: str
    forward: str
    publishes: str
    requires: str


@dataclass(frozen=True)
class ConnectionRow:
    """One module that can run right before or right after another, and the values between them."""

    position: str
    name: str
    values: str


def list_models(name: str | None = None) -> pd.DataFrame:
    """List the registered models, or what connects to one of them.

    Args:
        name: A model's registered name. None lists every model.

    Returns:
        With no name, one row per registered name, indexed by `family`
        (`encoder`, `decoder`, `head`, `monolith`, or `custom` for a class
        registered outside the library) and `name`: its `class`; `forward`,
        what it reads and returns; `publishes`, what later modules can take;
        and `requires`, the arguments a caller must give. With a name, one
        row per model that can run right `before` or `after` it, with the
        `values` passed between them: the names the later model reads from
        what the earlier one returns, by name and type hint. Names neither
        returns, such as `centroid`, are chain inputs.

    Examples:
        >>> list_models().loc[("encoder", "dinov3"), "forward"]
        'forward(image) -> pyramid, prefix_tokens'
        >>> list_models().loc["head"].index
        Index(['classification', 'regression', 'segmentation'], dtype='object', name='name')
        >>> list_models("unet")
                                  values
        position name
        before   clay             pyramid
                 dinov3           pyramid
                 ...
        after    regression       feature_map
                 segmentation     feature_map
    """
    if name is None:
        return build_module_table()
    return build_connection_table(name)


def build_module_table() -> pd.DataFrame:
    """Return one row per registered name, grouped by family: its class, forward, publications and requirements."""
    published: set[str] = set()
    for cls in MODULES.values():
        published.update(read_published(cls))

    rows = []
    for family in MODULES.families:
        for name, cls in MODULES.items():
            if MODULES.find_family(cls) != family:
                continue
            rows.append(
                ModuleRow(
                    family=family,
                    name=name,
                    cls=cls.__name__,
                    forward=format_forward(cls),
                    publishes=", ".join(read_published(cls)),
                    requires=", ".join(find_required_arguments(cls, published)),
                )
            )

    table = pd.DataFrame(rows).set_index(["family", "name"])
    return table.rename(columns={"cls": "class"})


def build_connection_table(name: str) -> pd.DataFrame:
    """Return the names that can run right before and right after a name, through which values."""
    cls = MODULES[name]
    before = []
    after = []
    for other, other_cls in MODULES.items():
        passed = find_passed_names(other_cls, cls)
        if passed:
            before.append(ConnectionRow("before", other, ", ".join(passed)))

        passed = find_passed_names(cls, other_cls)
        if passed:
            after.append(ConnectionRow("after", other, ", ".join(passed)))

    columns = pd.Index(["position", "name", "values"])
    return pd.DataFrame(before + after, columns=columns).set_index(["position", "name"])


def find_passed_names(previous: type[nn.Module], cls: type[nn.Module]) -> list[str]:
    """Return the names `cls` reads from what `previous` returns, when the two form a chain.

    Returns:
        The names, in `cls`'s parameter order; none when the two do not form a
        chain or `cls` reads nothing `previous` returns.
    """
    try:
        check_chain({"previous": previous, "next": cls})
    except TypeError:
        return []

    returned = read_returns(previous)
    passed = []
    for parameter in read_forward_inputs(cls):
        if parameter.name in returned:
            passed.append(parameter.name)
    return passed


def find_required_arguments(cls: type[nn.Module], published: set[str]) -> list[str]:
    """Return the constructor arguments only a caller can give."""
    own = set(read_published(cls))
    required = []
    for argument, parameter in inspect.signature(cls).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        if argument in published and argument not in own:
            continue
        required.append(argument)
    return required
