"""Write one attr group onto an xarray object."""

from __future__ import annotations

import xarray as xr

from .groups.base import AttrsGroup


def stamp[G: AttrsGroup, X: xr.DataArray | xr.Dataset](
    obj: X, group: type[G], value: G | None
) -> X:
    """Return a copy of the object stating `value` for one group.

    Every key of the group is removed first, so no stale spelling survives.

    Args:
        obj: Variable or coordinate, or a Dataset whose root states the group.
        group: Group the value belongs to.
        value: Value to state. None removes the group.

    Returns:
        Shallow copy of the object with replaced attrs; pixels are not copied.

    Raises:
        ValueError: The variable's dtype cannot hold the value.

    Examples:
        >>> stamp(ds["B04"], Nodata, Nodata(value=0)).attrs["_FillValue"]
        0
        >>> stamp(ds, ACDD, ACDD(title="Forest")).attrs["title"]
        'Forest'
    """
    attrs = {key: v for key, v in obj.attrs.items() if key not in group.KEYS}
    if value is not None:
        # Only a variable has a dtype a value must fit.
        held = value.held_by(obj) if isinstance(obj, xr.DataArray) else value
        attrs |= held.to_attrs()
    stamped = obj.copy(deep=False)
    stamped.attrs = attrs
    return stamped
