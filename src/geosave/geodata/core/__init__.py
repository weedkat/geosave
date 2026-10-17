"""Public geodata core: the `gs` accessors of rasters and vectors, and GeoAnchor."""

from .anchor import GeoAnchor
from .raster.dataarray import GeoDataArray
from .raster.dataset import GeoDataset
from .raster.datatree import GeoDataTree
from .vector.catalog import GeoCatalog
from .vector.accessors import FrameAccessor, RowAccessor  # noqa: F401  registers `gs`
from .vector.item import GeoItem
from .vector.sample import GeoSample
from .vector.samples import GeoSamples
from .vector.vector import GeoVector

# After the accessors, which import the `raster` package this name would shadow.
from .raster.create import RasterVariable, array, raster, tree  # noqa: E402

__all__ = [
    "GeoAnchor",
    "GeoDataArray",
    "GeoDataset",
    "GeoDataTree",
    "RasterVariable",
    "GeoVector",
    "GeoCatalog",
    "GeoSamples",
    "GeoItem",
    "GeoSample",
    "array",
    "raster",
    "tree",
]
