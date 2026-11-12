"""Discover, load and publish STAC Items as native xarray and STAC objects."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .catalog import create_catalog
    from .load import StacLoadOptions, load_dataset, load_datatree
    from .publish import create_stac_item
    from .search.api import StacClient, StacProvider
    from .search.query import StacQuery
    from .search.table import StacTable

_LAZY = {
    "StacClient": ".search.api",
    "StacProvider": ".search.api",
    "StacQuery": ".search.query",
    "StacTable": ".search.table",
    "create_catalog": ".catalog",
    "StacLoadOptions": ".load",
    "load_dataset": ".load",
    "load_datatree": ".load",
    "create_stac_item": ".publish",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "StacClient",
    "StacProvider",
    "StacQuery",
    "StacTable",
    "create_catalog",
    "StacLoadOptions",
    "load_dataset",
    "load_datatree",
    "create_stac_item",
]
