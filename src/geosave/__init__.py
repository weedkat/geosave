"""GeoSave's geodata surface, reached as `gs.open_raster`, `gs.raster`, `gs.stack`.

Importing the package imports `geodata`, which registers the `gs` accessor on
xarray objects, GeoDataFrames and cuts rows, so `.gs` works right after
`import geosave as gs`. torch and the viz stack stay unloaded until a
member that needs them is called; `cuts`, `features` and `stac` load on
first use.

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
    GeoArray,
    GeoRaster,
    GeoStack,
    GeoVector,
    array,
    attrs,
    configure_gdal,
    io,
    raster,
    open_raster,
    open_stack,
    open_vector,
    stack,
    transform,
)

__all__ = [
    "DataArray",
    "DataTree",
    "Dataset",
    "GeoDataFrame",
    "GeoAnchor",
    "GeoArray",
    "GeoRaster",
    "GeoStack",
    "GeoVector",
    "array",
    "attrs",
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

# These bring their own heavy dependencies, so they load on first use.
_LAZY = ("cuts", "features", "stac")

if TYPE_CHECKING:
    from geosave.geodata import cuts, features, stac


def __getattr__(name: str) -> object:
    """Fetch `cuts`, `features` or `stac` from `geodata` on first use.

    Raises:
        AttributeError: `name` is not part of the public surface.
    """
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from geosave import geodata

    return getattr(geodata, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
