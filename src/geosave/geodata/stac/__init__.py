"""Discover, load and publish STAC Items as native xarray and STAC objects."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import stamps as stamps
    from .catalog import concat_catalogs, create_catalog
    from .load import (
        StacLoadOptions,
        group_bands_by_grid,
        load_dataset,
        load_native_datasets,
    )
    from .publish import StacItem, publish_item
    from .search.api import StacClient
    from .search.query import StacQuery
    from .search.table import StacTable

_LAZY = {
    "StacClient": ".search.api",
    "StacQuery": ".search.query",
    "StacTable": ".search.table",
    "create_catalog": ".catalog",
    "concat_catalogs": ".catalog",
    "StacLoadOptions": ".load",
    "load_dataset": ".load",
    "group_bands_by_grid": ".load",
    "load_native_datasets": ".load",
    "StacItem": ".publish",
    "publish_item": ".publish",
}


def __getattr__(name: str) -> Any:
    if name == "stamps":
        return import_module(".stamps", __name__)
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "stamps",
    "StacClient",
    "StacQuery",
    "StacTable",
    "create_catalog",
    "concat_catalogs",
    "StacLoadOptions",
    "load_dataset",
    "group_bands_by_grid",
    "load_native_datasets",
    "StacItem",
    "publish_item",
]
