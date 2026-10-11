"""Attr groups: a few attr keys that mean one value together.

xarray's attrs stay plain mappings, kept through operations by xarray. A
group validates the keys GeoSave decides on or translates at a format
boundary; `gs` reads each one as a property and writes it with `write_*`.
Joins pass `combine_attrs=attrs.combine`.
"""

from .base import AttrsGroup, Decoded, Join, decode, parse, stamp
from .combine import combine, shared
from .gdal import GDALBand
from .legend import Legend, is_flag
from .nodata import Nodata
from .packing import Packing
from .spectral import Spectral
from .timespec import TimeSpec

__all__ = [
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
