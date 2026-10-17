"""What physical quantity a variable's values measure, as CF names it."""

from __future__ import annotations


from .base import AttrsGroup


class Quantity(AttrsGroup):
    """Name the quantity one variable or coordinate measures.

    Two objects stating it differently hold values that mean different
    things, so a join refuses them rather than dropping it.

    Attributes:
        standard_name: CF standard name, as `"surface_reflectance"`.
        units: UDUNITS unit, as `"1"` or `"m"`.
        cell_methods: How values were collapsed, as `"time: mean"`.

    Examples:
        >>> attrs.Quantity.from_attrs(ds["B04"].attrs).units
        '1'
    """

    REFUSE_CONFLICT = True

    standard_name: str | None = None
    units: str | None = None
    cell_methods: str | None = None
