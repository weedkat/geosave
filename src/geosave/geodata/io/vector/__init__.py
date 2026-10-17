"""Vector formats: GeoJSON, GeoPackage, GeoParquet, sample GeoParquet, stac-geoparquet and STAC JSON."""

from importlib import import_module
from typing import Any

from . import geojson, geopackage, geoparquet, sample_geoparquet

# STAC brings its own heavy dependencies, so its formats load on first use.
_LAZY = ("stac_geoparquet", "stac_json")


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "geojson",
    "geopackage",
    "geoparquet",
    "sample_geoparquet",
    "stac_geoparquet",
    "stac_json",
]
