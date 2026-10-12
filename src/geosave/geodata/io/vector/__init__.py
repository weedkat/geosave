"""Vector formats: GeoJSON, GeoPackage, GeoParquet and stac-geoparquet."""

from importlib import import_module
from typing import Any

from . import geojson, geopackage, geoparquet

# STAC brings its own heavy dependencies, so stac-geoparquet loads on first use.
_LAZY = ("stac_geoparquet",)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = ["geojson", "geopackage", "geoparquet", "stac_geoparquet"]
