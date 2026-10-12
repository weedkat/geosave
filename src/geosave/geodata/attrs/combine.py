"""Join the attrs of several xarray objects inside xarray's own concat and merge.

Pass `combine_attrs=attrs.combine`; xarray calls it once per variable,
coordinate and root with every object's attrs.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from xarray.core.utils import equivalent

from geosave import __path__ as _package_paths
from geosave.geodata.warnings import DroppedAttrsWarning

from .groups.base import AttrsGroup

# CF keys outside any group whose disagreement changes what the pixels mean.
AGREE = frozenset({"units", "standard_name", "cell_methods"})


def combine(
    attrs_list: Sequence[Mapping[Hashable, Any]], context: object = None
) -> dict[str, Any]:
    """Join attrs as each group's `JOIN` says, other keys by what every object states alike.

    Groups and plain keys every object states alike are kept, a group spelled
    once. Where objects differ, an `"agree"` group or an `AGREE` key refuses
    the join, and anything else is left out.

    Args:
        attrs_list: Attrs of the objects joined, in order.
        context: Passed by xarray; unused.

    Returns:
        Attrs of the joined object.

    Raises:
        ValueError: Objects state an `"agree"` group or an `AGREE` key
            differently; align it before joining, as with `gs.write_nodata`.

    Warns:
        DroppedAttrsWarning: Objects state other attrs differently, so the
            join leaves them out.

    Examples:
        >>> xr.concat(rasters, dim="time", combine_attrs=attrs.combine)
    """
    joined = _join(attrs_list, refuse=True)
    # A kept group is spelled once, so its other spellings were not dropped.
    respelled = {
        key
        for group in AttrsGroup.__subclasses__()
        if group.JOIN != "each" and group.stated_in(joined)
        for key in group.KEYS
    }
    stated = {str(key) for attrs in attrs_list for key in attrs}
    dropped = sorted(stated - set(joined) - respelled)
    if dropped:
        warnings.warn(
            f"joining drops attrs the objects state differently: {dropped}",
            DroppedAttrsWarning,
            skip_file_prefixes=tuple(_package_paths),
        )
    return joined


def shared(attrs_list: Sequence[Mapping[Hashable, Any]]) -> dict[str, Any]:
    """Return the attrs every object states alike, each group kept whole or not at all.

    Unlike `combine`, a group the objects state differently is left out
    rather than refused, as when bands of one raster are stacked.

    Examples:
        >>> shared([ds[name].attrs for name in ds.data_vars])
        {'units': '1'}
    """
    return _join(attrs_list, refuse=False)


def _join(
    attrs_list: Sequence[Mapping[Hashable, Any]], *, refuse: bool
) -> dict[str, Any]:
    """Join groups whole and plain keys one by one, refusing `"agree"` groups that differ."""
    joined: dict[str, Any] = {}
    owned: set[str] = set()
    for group in AttrsGroup.__subclasses__():
        # An "each" group's keys are facts on their own, joined below as plain keys.
        if group.JOIN == "each":
            continue
        owned.update(group.KEYS)
        joined |= _join_group(group, attrs_list, refuse=refuse)
    for attrs in attrs_list:
        for key, value in attrs.items():
            if key in owned or key in joined:
                continue
            # A key one object lacks would speak for pixels it never described.
            if all(
                key in other and equivalent(other[key], value) for other in attrs_list
            ):
                joined[str(key)] = value
            elif refuse and key in AGREE and all(key in other for other in attrs_list):
                stated = sorted({repr(other[key]) for other in attrs_list})
                raise ValueError(
                    f"the objects joined state {key!r} differently: {stated}; "
                    f"align it before joining them"
                )
    return joined


def _join_group(
    group: type[AttrsGroup],
    attrs_list: Sequence[Mapping[Hashable, Any]],
    *,
    refuse: bool,
) -> dict[str, Any]:
    """Return the group's attrs where every object states it alike, else none."""
    values = [group.from_attrs(attrs) for attrs in attrs_list]
    first, *rest = values
    if first is not None and all(value == first for value in rest):
        return first.to_attrs()
    if refuse and group.JOIN == "agree" and None not in values:
        stated = sorted({repr(value) for value in values})
        raise ValueError(
            f"the objects joined state {group.__name__} differently: {stated}; "
            f"align it before joining them"
        )
    return {}
