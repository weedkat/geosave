"""Write or remove one attr group on an xarray object."""

from __future__ import annotations

import xarray as xr

from .groups.base import AttrsGroup


def stamp_attrs[X: xr.DataArray | xr.Dataset](obj: X, value: AttrsGroup) -> X:
    """Return a copy of the object stating `value` for its group.

    Every key of the group is removed first, so no stale alias survives.

    Args:
        obj: Variable or coordinate, or a Dataset whose root states the group.
        value: Group value to state.

    Returns:
        Shallow copy of the object with replaced attrs; pixels are not copied.

    Examples:
        >>> stamp_attrs(ds["B04"], Nodata(value=0)).attrs["_FillValue"]
        0
        >>> stamp_attrs(ds, ACDD(title="Forest")).attrs["title"]
        'Forest'
    """
    stamped = clear_attrs(obj, type(value))
    stamped.attrs.update(value.to_attrs())
    return stamped


def clear_attrs[X: xr.DataArray | xr.Dataset](obj: X, group: type[AttrsGroup]) -> X:
    """Return a copy of the object stating none of the group.

    Args:
        obj: Variable or coordinate, or a Dataset whose root states the group.
        group: Group whose every key, aliases included, is removed.

    Returns:
        Shallow copy of the object with replaced attrs; pixels are not copied.

    Examples:
        >>> clear_attrs(ds["B04"], Nodata).attrs
        {'units': '1'}
    """
    # A shallow copy still gets its own attrs dict, so the original keeps its keys.
    cleared = obj.copy(deep=False)
    for key in group.KEYS:
        cleared.attrs.pop(key, None)
    return cleared
