"""Attr groups: a few attr keys that mean one value together.

xarray's attrs stay plain mappings, kept through operations by xarray. A
group validates the keys GeoSave decides on or translates at a format
boundary; `gs` reads each one as a property and writes it with `write_*`.
Joins pass `combine_attrs=attrs.combine_attrs`.
"""

from .groups.base import AttrsGroup, Decoded, decode
from .combine import combine_attrs
from .stamp import AttrsObject, clear_attrs, stamp_attrs
from .groups.acdd import ACDD
from .groups.extent import Extent
from .groups.gdal import GDALBand
from .groups.legend import Legend
from .groups.packing import Packing
from .groups.quantity import Quantity
from .groups.spectral import Spectral
from .groups.stac import StacMetadata

__all__ = [
    "ACDD",
    "AttrsGroup",
    "AttrsObject",
    "Decoded",
    "Extent",
    "GDALBand",
    "Legend",
    "Packing",
    "Quantity",
    "Spectral",
    "StacMetadata",
    "clear_attrs",
    "combine_attrs",
    "decode",
    "stamp_attrs",
]
