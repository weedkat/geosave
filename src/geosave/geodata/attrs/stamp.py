"""Write or remove one attr group on an object with attrs."""

from __future__ import annotations

from typing import Any, Protocol, Self

from .groups.base import AttrsGroup


class AttrsObject(Protocol):
    """Object whose shallow copy has an independent attrs dictionary."""

    @property
    def attrs(self) -> dict[str, Any]:
        """Return the object's mutable attrs dictionary."""
        ...

    def copy(self, *, deep: bool = False) -> Self:
        """Copy the object, sharing data when deep is False."""
        ...


def stamp_attrs[X: AttrsObject](obj: X, value: AttrsGroup) -> X:
    """Replace one group on a copy's own attrs without traversing children.

    Every key of the group is removed first, so no stale alias survives.

    Args:
        obj: Object with attrs and a shallow copy that owns its attrs dictionary.
        value: Group value to state.

    Returns:
        Shallow copy of the object with replaced attrs; pixels are not copied.

    Examples:
        >>> stamp_attrs(ds["B04"], Packing(fill_value=0)).attrs["_FillValue"]
        0
        >>> stamp_attrs(ds, ACDD(title="Forest")).attrs["title"]
        'Forest'
    """
    stamped = clear_attrs(obj, type(value))
    stamped.attrs.update(value.to_attrs())
    return stamped


def clear_attrs[X: AttrsObject](obj: X, group: type[AttrsGroup]) -> X:
    """Remove one group from a copy's own attrs without traversing children.

    Args:
        obj: Object with attrs and a shallow copy that owns its attrs dictionary.
        group: Group whose every key, aliases included, is removed.

    Returns:
        Shallow copy of the object with replaced attrs; pixels are not copied.

    Examples:
        >>> clear_attrs(ds["B04"], Packing).attrs
        {'units': '1'}
    """
    # A shallow copy still gets its own attrs dict, so the original keeps its keys.
    cleared = obj.copy(deep=False)
    for key in group.KEYS:
        cleared.attrs.pop(key, None)
    return cleared
