"""Join the attrs of several xarray objects inside xarray's own concat and merge.

Pass `combine_attrs=attrs.combine_attrs`; xarray calls it once per variable,
coordinate and root with the attrs of every input at that place. Each call
joins one level, so a group is read off variables, coordinates or roots alike.

An input's attrs hold attr groups, whose keys mean one value together, and
plain keys, which belong to no group. A group is joined whole by the group
itself; a plain key is kept when every input states it alike.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, cast

from xarray.structure.merge import merge_attrs

from geosave import __path__ as _package_paths
from geosave.geodata.warnings import DroppedAttrsWarning

from .groups.base import AttrsGroup

type Attrs = Mapping[Any, Any]


def combine_attrs(
    input_attrs: Sequence[Attrs], context: object = None
) -> dict[str, Any]:
    """Join attrs as `intersect` does, refusing groups whose conflict is an error.

    Args:
        input_attrs: Attrs of the inputs joined, in order.
        context: Where in the tree xarray is joining; xarray passes it by keyword,
            so the callback must accept it, but a join does not depend on it.

    Returns:
        Attrs of the joined object.

    Raises:
        ValueError: Every input states a refusing group, not alike; align
            it before joining, as with `gs.write_nodata`.

    Warns:
        DroppedAttrsWarning: Inputs state other attrs differently, so the
            join leaves them out.

    Examples:
        >>> xr.concat(rasters, dim="time", combine_attrs=attrs.combine_attrs)
    """
    shared_attrs = _intersect_keys(input_attrs)
    written_keys: set[str] = set()
    for attrs in input_attrs:
        written_keys.update(str(key) for key in attrs)

    for group in AttrsGroup.__subclasses__():
        values = [group.from_attrs(attrs) for attrs in input_attrs]
        joined = group.combine_values(values)
        if joined is not None:
            shared_attrs.update(joined.to_attrs())
        written_keys.difference_update(group.KEYS)
        for value in values:
            if value is not None:
                written_keys.update(value.to_attrs())

    dropped_keys = written_keys - shared_attrs.keys()
    if dropped_keys:
        warnings.warn(
            f"joining drops attrs the inputs state differently: {sorted(dropped_keys)}",
            DroppedAttrsWarning,
            skip_file_prefixes=tuple(_package_paths),
        )
    return shared_attrs


def intersect(input_attrs: Sequence[Attrs]) -> dict[str, Any]:
    """Return the attrs every input states alike, each group kept whole or not at all.

    A group is compared as the value it validates to, so its aliases read
    alike. Unlike `combine_attrs`, nothing is refused or warned about, as when
    bands of one raster are stacked.

    Args:
        input_attrs: Nonempty attrs of the inputs to compare.

    Returns:
        New mapping of shared attrs, each group written once. Groups and keys
        missing from any input or stated differently are omitted.

    Raises:
        ValueError: The sequence is empty or an attr group is invalid.

    Examples:
        >>> intersect([ds[name].attrs for name in ds.data_vars])
        {'units': '1'}
    """
    shared_attrs = _intersect_keys(input_attrs)
    for group in AttrsGroup.__subclasses__():
        values = [group.from_attrs(attrs) for attrs in input_attrs]
        joined = group.intersect_values(values)
        if joined is not None:
            shared_attrs.update(joined.to_attrs())
    return shared_attrs


def _intersect_keys(input_attrs: Sequence[Attrs]) -> dict[str, Any]:
    """Keep equal keys present in every input, leaving groups to their models."""
    if not input_attrs:
        raise ValueError("joining needs the attrs of at least one input")

    shared_keys = set(input_attrs[0])
    for attrs in input_attrs[1:]:
        shared_keys.intersection_update(attrs)
    for group in AttrsGroup.__subclasses__():
        shared_keys.difference_update(group.KEYS)
    merged_attrs = cast("dict[Any, Any]", merge_attrs(input_attrs, "drop_conflicts"))
    return {str(key): merged_attrs[key] for key in shared_keys & merged_attrs.keys()}
