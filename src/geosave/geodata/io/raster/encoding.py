"""The `encoding=` a caller passes to the Zarr and netCDF writers."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import xarray as xr


def check_encoding(
    names: Iterable[Hashable], encoding: Mapping[Any, Any] | None
) -> None:
    """Refuse an encoding naming a variable the raster does not hold.

    Args:
        names: Variable names of the raster, or of every group of a tree.
        encoding: Storage settings keyed by variable name.

    Raises:
        KeyError: An encoding names no variable.
    """
    unknown = set(encoding or {}) - set(names)
    if unknown:
        raise KeyError(f"encoding names no variable: {sorted(unknown, key=str)}")


def split_encoding(
    tree: xr.DataTree, encoding: Mapping[Any, Any] | None
) -> dict[str, dict[Hashable, Any]]:
    """Hand each group the storage settings of the variables it holds.

    xarray's tree writers take encoding per group path; callers state it once
    per variable name.

    Args:
        tree: Tree about to be written.
        encoding: Storage settings keyed by variable name.

    Returns:
        {
            "<group path>": {"<variable name>": its storage settings},
        }

    Raises:
        KeyError: An encoding names a variable no group holds.

    Examples:
        >>> split_encoding(tree, {"red": {"dtype": "uint16"}})
        {'/': {}, '/s2': {'red': {'dtype': 'uint16'}}}
    """
    shared = dict(encoding or {})
    names: set[Hashable] = set()
    grouped: dict[str, dict[Hashable, Any]] = {}
    for node in tree.subtree:
        names.update(node.variables)
        grouped[node.path] = {}
        for name, settings in shared.items():
            if name in node.variables:
                grouped[node.path][name] = settings
    check_encoding(names, shared)
    return grouped
