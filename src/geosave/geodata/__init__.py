"""Public geodata surface."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import xarray as xr

from . import attrs, transform
from .core import (
    GeoAnchor,
    GeoDataArray,
    GeoDataset,
    GeoDataTree,
    GeoVector,
    GeoCatalog,
    GeoItem,
    GeoSample,
    GeoSamples,
    array,
    raster,
    tree,
)
from . import io
from .utils.summary import class_percent, statistics
from .io.raster.gdal import configure_gdal
from .io import open_raster, open_tree, open_vector
from .io.raster.file import RasterFile, Scene


if TYPE_CHECKING:
    from typing import Generic
    from typing_extensions import TypeVar

    from . import features, stac
    from .stac import StacClient, StacQuery

    class DataArray(xr.DataArray):
        """Xarray DataArray carrying GeoSave's `gs` accessor.

        The runtime value is an `xarray.DataArray`; this declaration exists
        only so type checkers can resolve `.gs`.
        """

        gs: GeoDataArray

    class Dataset(xr.Dataset):
        """Xarray Dataset carrying GeoSave's `gs` accessor.

        The runtime value is an `xarray.Dataset`; this declaration exists
        only so type checkers can resolve `.gs`.
        """

        gs: GeoDataset

    class DataTree(xr.DataTree):
        """Xarray DataTree carrying GeoSave's `gs` accessor.

        The runtime value is an `xarray.DataTree`; this declaration exists
        only so type checkers can resolve `.gs`.
        """

        gs: GeoDataTree

    _Vector = TypeVar("_Vector", bound=GeoVector, default=Any)

    class GeoDataFrame(gpd.GeoDataFrame, Generic[_Vector]):
        """GeoPandas GeoDataFrame carrying GeoSave's `gs` accessor.

        The runtime value is a `geopandas.GeoDataFrame`; this declaration
        exists only so type checkers can resolve `.gs`. Known schemas specify
        the accessor type; file readers leave it dynamic until the columns
        are read at runtime.
        """

        gs: _Vector

else:
    DataArray = xr.DataArray
    Dataset = xr.Dataset
    DataTree = xr.DataTree
    GeoDataFrame = gpd.GeoDataFrame


# STAC brings its own heavy dependencies, so these load on first use.
_LAZY = ("features", "stac")
_LAZY_MEMBERS = {"StacClient": "stac", "StacQuery": "stac"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return import_module(f"{__name__}.{name}")
    if name in _LAZY_MEMBERS:
        return getattr(import_module(f"{__name__}.{_LAZY_MEMBERS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY, *_LAZY_MEMBERS})


__all__ = [
    "attrs",
    "GeoAnchor",
    "GeoDataArray",
    "DataArray",
    "DataTree",
    "Dataset",
    "GeoDataFrame",
    "GeoDataset",
    "GeoDataTree",
    "GeoVector",
    "GeoCatalog",
    "GeoItem",
    "GeoSample",
    "GeoSamples",
    "RasterFile",
    "Scene",
    "StacClient",
    "StacQuery",
    "array",
    "configure_gdal",
    "features",
    "io",
    "raster",
    "open_raster",
    "open_tree",
    "open_vector",
    "stac",
    "tree",
    "class_percent",
    "statistics",
    "transform",
]
