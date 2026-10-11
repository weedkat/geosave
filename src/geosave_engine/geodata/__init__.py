"""Public geodata surface."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import xarray as xr

from . import transform
from .core import (
    GeoAnchor,
    GeoArray,
    GeoRaster,
    GeoStack,
    GeoVector,
    array,
    raster,
    stack,
)
from . import io
from .io.raster.gdal import configure_gdal
from .io import open_raster, open_stack, open_vector


if TYPE_CHECKING:
    from . import cuts, features, stac

    class DataArray(xr.DataArray):
        """Xarray DataArray carrying GeoSave's `gs` accessor.

        The runtime value is an `xarray.DataArray`; this declaration exists
        only so type checkers can resolve `.gs`.
        """

        gs: GeoArray

    class Dataset(xr.Dataset):
        """Xarray Dataset carrying GeoSave's `gs` accessor.

        The runtime value is an `xarray.Dataset`; this declaration exists
        only so type checkers can resolve `.gs`.
        """

        gs: GeoRaster

    class DataTree(xr.DataTree):
        """Xarray DataTree carrying GeoSave's `gs` accessor.

        The runtime value is an `xarray.DataTree`; this declaration exists
        only so type checkers can resolve `.gs`.
        """

        gs: GeoStack

    class GeoDataFrame(gpd.GeoDataFrame):
        """GeoPandas GeoDataFrame carrying GeoSave's `gs` accessor.

        The runtime value is a `geopandas.GeoDataFrame`; this declaration
        exists only so type checkers can resolve `.gs`.
        """

        gs: GeoVector

else:
    DataArray = xr.DataArray
    Dataset = xr.Dataset
    DataTree = xr.DataTree
    GeoDataFrame = gpd.GeoDataFrame


# STAC brings its own heavy dependencies, so these load on first use.
_LAZY = ("cuts", "features", "stac")


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "GeoAnchor",
    "GeoArray",
    "DataArray",
    "DataTree",
    "Dataset",
    "GeoDataFrame",
    "GeoRaster",
    "GeoStack",
    "GeoVector",
    "array",
    "configure_gdal",
    "cuts",
    "features",
    "io",
    "raster",
    "open_raster",
    "open_stack",
    "open_vector",
    "stac",
    "stack",
    "transform",
]
