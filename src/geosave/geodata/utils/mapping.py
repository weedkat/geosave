"""Merge several mappings into one, key by key."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr


def intersect_mappings[K: Hashable](
    mappings: Sequence[Mapping[K, Any]],
) -> dict[K, Any]:
    """Return the keys every mapping states with one value.

    Args:
        mappings: Nonempty mappings to compare.

    Returns:
        New mapping of the keys every input states alike, in the first
        input's order. A key missing from any input or stated differently is
        left out. Arrays compare element-wise, NaN equals NaN, and xarray
        objects compare with `equals`.

    Raises:
        ValueError: No mapping is given.

    Examples:
        >>> intersect_mappings([{"platform": "s2a", "gsd": 10}, {"platform": "s2b", "gsd": 10}])
        {'gsd': 10}
    """
    if not mappings:
        raise ValueError("intersecting needs at least one mapping")
    shared = dict(mappings[0])
    for mapping in mappings[1:]:
        for key in list(shared):
            if key not in mapping or not _equal(mapping[key], shared[key]):
                del shared[key]
    return shared


def stack_mappings[K: Hashable](
    mappings: Sequence[Mapping[K, Any]],
) -> dict[K, list[Any]]:
    """Return every key mapped to each mapping's value, in mapping order.

    Args:
        mappings: Mappings to stack.

    Returns:
        New mapping with one list per key, holding one entry per input; None
        where an input lacks the key. Keys keep the order they first appear in.

    Examples:
        >>> stack_mappings([{"platform": "s2a", "eo:cloud_cover": 10.0}, {"platform": "s2b"}])
        {'platform': ['s2a', 's2b'], 'eo:cloud_cover': [10.0, None]}
    """
    keys: list[K] = []
    for mapping in mappings:
        for key in mapping:
            if key not in keys:
                keys.append(key)
    stacked = {}
    for key in keys:
        values = []
        for mapping in mappings:
            values.append(mapping.get(key))
        stacked[key] = values
    return stacked


def _equal(left: Any, right: Any) -> bool:
    """Compare two values, arrays element-wise and NaN as equal to NaN."""
    if isinstance(left, xr.DataArray):
        return isinstance(right, xr.DataArray) and left.equals(right)
    if isinstance(left, xr.Variable):
        return isinstance(right, xr.Variable) and left.equals(right)
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    if isinstance(left, (np.ndarray, np.generic)) or isinstance(
        right, (np.ndarray, np.generic)
    ):
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            # Text and object arrays hold no NaN to compare.
            return bool(np.array_equal(left, right))
    return bool(left == right)
