"""Read one attr group off an xarray object."""

from __future__ import annotations

import xarray as xr

from .groups.base import AttrsGroup


def parse[G: AttrsGroup](obj: xr.DataArray | xr.Dataset, group: type[G]) -> G | None:
    """Read one group off an object's attrs, validated against a variable's dtype.

    Args:
        obj: Variable or coordinate, or a Dataset whose root states the group.
        group: Group to read.

    Returns:
        The group's value, or None where the attrs state none of its keys.

    Raises:
        ValueError: The attrs state the group invalidly or the variable's
            dtype cannot hold it.

    Examples:
        >>> parse(ds["B04"], Nodata)
        Nodata(value=0)
        >>> parse(ds, ACDD).title
        'Sentinel-2 Level-2A'
    """
    value = group.from_attrs(obj.attrs)
    # Only a variable has a dtype a value must fit.
    if value is None or not isinstance(obj, xr.DataArray):
        return value
    return value.held_by(obj)
