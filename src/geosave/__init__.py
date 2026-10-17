"""GeoSave's geodata surface, reached as `gs.open_raster`, `gs.raster`, `gs.tree`.

Importing the package imports `geodata`, which registers the `gs` accessor on
xarray objects and GeoDataFrames, so `.gs` works right after
`import geosave as gs`. torch and the viz stack stay unloaded until a
member that needs them is called; `features` and `stac` load on first
use.

Examples:
    >>> import geosave as gs
    >>> gs.open_raster("scene.tif").gs.variables
    ('B04', 'B08')
"""

from typing import TYPE_CHECKING

from geosave.geodata import (
    DataArray,
    DataTree,
    Dataset,
    GeoDataFrame,
    GeoAnchor,
    GeoDataArray,
    GeoDataset,
    GeoDataTree,
    GeoVector,
    GeoCatalog,
    GeoSamples,
    array,
    attrs,
    configure_gdal,
    io,
    raster,
    open_raster,
    open_tree,
    open_vector,
    tree,
    class_percent,
    statistics,
    transform,
)

__all__ = [
    "DataArray",
    "DataTree",
    "Dataset",
    "GeoDataFrame",
    "GeoAnchor",
    "GeoDataArray",
    "GeoDataset",
    "GeoDataTree",
    "GeoVector",
    "GeoCatalog",
    "GeoSamples",
    "StacClient",
    "StacQuery",
    "array",
    "attrs",
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

# These bring their own heavy dependencies, so they load on first use.
_LAZY = ("StacClient", "StacQuery", "features", "stac")

if TYPE_CHECKING:
    from geosave.geodata import StacClient, StacQuery, features, stac


def __getattr__(name: str) -> object:
    """Fetch `StacClient`, `StacQuery`, `features` or `stac` from `geodata` on first use.

    Raises:
        AttributeError: `name` is not part of the public surface.
    """
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from geosave import geodata

    return getattr(geodata, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
