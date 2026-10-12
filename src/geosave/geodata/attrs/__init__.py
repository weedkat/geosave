"""Attr groups: a few attr keys that mean one value together.

xarray's attrs stay plain mappings, kept through operations by xarray. A
group validates the keys GeoSave decides on or translates at a format
boundary; `gs` reads each one as a property and writes it with `write_*`.
Joins pass `combine_attrs=attrs.combine`.
"""

from .groups.base import AttrsGroup, Decoded, Join, decode
from .combine import combine, shared
from .parse import parse
from .stamp import stamp
from .groups.acdd import ACDD
from .groups.gdal import GDALBand
from .groups.legend import Legend, is_flag
from .groups.nodata import Nodata
from .groups.packing import Packing
from .groups.spectral import Spectral
from .groups.timespec import TimeSpec

__all__ = [
    "ACDD",
    "AttrsGroup",
    "Decoded",
    "GDALBand",
    "Join",
    "Legend",
    "Nodata",
    "Packing",
    "Spectral",
    "TimeSpec",
    "combine",
    "decode",
    "is_flag",
    "parse",
    "shared",
    "stamp",
]
