"""Build native STAC metadata and discover Items through APIs or tables."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .catalog import from_items
    from .item import create_item
    from .search.api import StacClient, StacProvider
    from .search.query import StacQuery
    from .search.table import StacTable

_LAZY = {
    "StacClient": ".search.api",
    "StacProvider": ".search.api",
    "StacQuery": ".search.query",
    "StacTable": ".search.table",
    "from_items": ".catalog",
    "create_item": ".item",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "StacClient",
    "StacProvider",
    "StacQuery",
    "StacTable",
    "from_items",
    "create_item",
]
