"""Attr groups: a few attr keys that mean one value together.

xarray's attrs stay plain mappings, kept through operations by xarray. A
group validates the keys GeoSave decides on or translates at a format
boundary; `gs` reads each one as a property and writes it with `write_*`.
Joins pass `combine_attrs=attrs.combine_attrs`.
"""

from .groups.base import AttrsGroup, Decoded, decode
from .combine import combine_attrs, intersect
from .stamp import clear_attrs, stamp_attrs
from .groups.acdd import ACDD
from .groups.gdal import GDALBand
from .groups.legend import Legend
from .groups.nodata import Nodata
from .groups.packing import Packing
from .groups.quantity import Quantity
from .groups.spectral import Spectral
from .groups.timespec import TimeSpec

__all__ = [
    "ACDD",
    "AttrsGroup",
    "Decoded",
    "GDALBand",
    "Legend",
    "Nodata",
    "Packing",
    "Quantity",
    "Spectral",
    "TimeSpec",
    "clear_attrs",
    "combine_attrs",
    "decode",
    "intersect",
    "stamp_attrs",
]
