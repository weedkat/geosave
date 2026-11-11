"""GeoSave's geodata surface, reached as `gs.open_raster` and `gs.open_vector`.

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
    Chips,
    GeoSamples,
    DataArray,
    Dataset,
    DataTree,
    GeoRegion,
    GeoCatalog,
    GeoDataArray,
    GeoDataFrame,
    GeoDataset,
    GeoDataTree,
    GeoCatalogItem,
    attrs,
    class_percent,
    configure_gdal,
    io,
    open_raster,
    open_vector,
    statistics,
)

__all__ = [
    "DataArray",
    "DataTree",
    "Dataset",
    "GeoDataFrame",
    "GeoRegion",
    "GeoDataArray",
    "GeoDataset",
    "GeoDataTree",
    "GeoCatalog",
    "Chips",
    "GeoSamples",
    "GeoCatalogItem",
    "StacClient",
    "StacQuery",
    "create_catalog",
    "create_stac_item",
    "attrs",
    "configure_gdal",
    "features",
    "io",
    "open_raster",
    "open_vector",
    "stac",
    "class_percent",
    "statistics",
]

# These bring their own heavy dependencies, so they load on first use.
_LAZY = (
    "StacClient",
    "StacQuery",
    "features",
    "stac",
    "create_catalog",
    "create_stac_item",
)

if TYPE_CHECKING:
    from geosave.geodata import StacClient, StacQuery, features, stac
    from geosave.geodata.stac import create_catalog, create_stac_item


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
