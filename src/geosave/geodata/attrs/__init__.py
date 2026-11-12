"""Attr groups: a few attr keys that mean one value together.

xarray's attrs stay plain mappings, kept through operations by xarray. A
group validates the keys GeoSave decides on or translates at a format
boundary; `gs` reads each one as a property and writes it with `write_*`.
Joins pass `combine_attrs=attrs.combine_attrs`.
"""

from .groups.base import AttrsGroup, Decoded, decode
from .combine import combine_attrs
from .stamp import AttrsObject, clear_attrs, stamp_attrs
from .groups.acdd import ACDD, ACDDExtent
from .groups.cf import CFDescription, CFFlags, CFPacking
from .groups.display import Display
from .groups.eo import EOBand
from .groups.stac import StacCollection

__all__ = [
    "ACDD",
    "ACDDExtent",
    "AttrsGroup",
    "AttrsObject",
    "CFDescription",
    "CFFlags",
    "CFPacking",
    "Decoded",
    "Display",
    "EOBand",
    "StacCollection",
    "clear_attrs",
    "combine_attrs",
    "decode",
    "stamp_attrs",
]
