"""What a variable's values are, as CF chapter 3 names it."""

from __future__ import annotations


from .base import AttrsGroup


class Description(AttrsGroup):
    """Say what one variable's values are.

    Find variables by these names with `.cf.standard_names` or
    `ds.cf["<standard name>"]`; this group reads them for one variable.

    Attributes:
        long_name: Human-readable name, as `"Red reflectance"`.
        standard_name: CF standard name, as `"surface_bidirectional_reflectance"`.
        units: UDUNITS unit, as `"1"` or `"m"`.

    Examples:
        >>> ds["B04"].gs.description.units
        '1'
    """

    long_name: str | None = None
    standard_name: str | None = None
    units: str | None = None
