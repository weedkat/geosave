"""Public geodata core: the `gs` accessors of rasters and vectors, and GeoRegion."""

from .region import GeoRegion
from .raster.dataarray import GeoDataArray
from .raster.dataset import GeoDataset
from .raster.datatree import GeoDataTree
from .vector.item import GeoCatalogItem
from .vector.catalog import GeoCatalog

# isort: split
from . import register as _register  # noqa: F401 registers native .gs accessors

__all__ = [
    "GeoRegion",
    "GeoDataArray",
    "GeoDataset",
    "GeoDataTree",
    "GeoCatalog",
    "GeoCatalogItem",
]
