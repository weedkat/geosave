"""Attr groups: a few attr keys that mean one value together.

xarray's attrs stay plain mappings, kept through operations by xarray. A
group validates the keys GeoSave decides on or translates at a format
boundary; `gs` reads each one as a property and writes it with `write_*`.
Joins use xarray's own `combine_attrs` policies.
"""

from .groups.base import AttrsGroup, Decoded, decode
from .stamp import AttrsObject, clear_attrs, stamp_attrs
from .groups.acdd import ACDD
from .groups.cf import CFDescription, CFFlags, CFPacking
from .groups.display import Display
from .groups.eo import EOBand
from .groups.extent import Extent
from .groups.stac import StacCollection, StacProperty

__all__ = [
    "ACDD",
    "AttrsGroup",
    "AttrsObject",
    "CFDescription",
    "CFFlags",
    "CFPacking",
    "Decoded",
    "Display",
    "EOBand",
    "Extent",
    "StacCollection",
    "StacProperty",
    "clear_attrs",
    "decode",
    "stamp_attrs",
]
