"""Public geodata core: the `gs` accessors of rasters and vectors, and GeoAnchor."""


from .anchor import GeoAnchor
from .raster.chips import Chips
from .raster.dataarray import GeoDataArray
from .raster.dataset import GeoDataset
from .raster.datatree import GeoDataTree
from .vector.item import GeoCatalogItem
from .vector.catalog import GeoCatalog
from .vector.samples import GeoSamples

# isort: split
from . import register as _register  # noqa: F401 registers native .gs accessors

__all__ = [
    "Chips",
    "GeoAnchor",
    "GeoDataArray",
    "GeoDataset",
    "GeoDataTree",
    "GeoCatalog",
    "GeoCatalogItem",
    "GeoSamples",
]
