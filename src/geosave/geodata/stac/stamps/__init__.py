"""Stamps: named steps that write what STAC documents state onto a loaded Dataset.

A stamp takes the loaded Dataset and the Items behind it and returns the
Dataset with metadata written, reading no pixels. `load_dataset` runs the
stamps named in `stamps=`, in order; a later stamp overrides attrs an earlier
one wrote.

Examples:
    >>> scene = stac.load_dataset(items, grid, stamps=[*DEFAULT_STAMPS, "sentinel_2"])
"""

from collections.abc import Sequence

import xarray as xr

from .bands import (
    stamp_classification,
    stamp_eo,
    stamp_raster,
    stamp_rgb,
    stamp_title,
)
from .collection import stamp_collection
from .properties import stamp_properties
from .source import LoadSource, Stamp

STAMPS: dict[str, Stamp] = {
    "raster": stamp_raster,
    "eo": stamp_eo,
    "classification": stamp_classification,
    "rgb": stamp_rgb,
    "title": stamp_title,
    "properties": stamp_properties,
    "collection": stamp_collection,
}

DEFAULT_STAMPS = (
    "raster",
    "eo",
    "classification",
    "rgb",
    "title",
    "properties",
    "collection",
)


def register_stamp(name: str, stamp: Stamp) -> None:
    """Make a project stamp selectable by name in `load_dataset(stamps=...)`.

    Args:
        name: Name callers pass in `stamps=`.
        stamp: Function writing metadata onto a loaded Dataset.

    Raises:
        ValueError: The name is taken.
    """
    if name in STAMPS:
        raise ValueError(f"stamp {name!r} is already registered")
    STAMPS[name] = stamp


def check_stamps(names: Sequence[str]) -> None:
    """Refuse names no stamp is registered under, before anything is loaded.

    Raises:
        KeyError: A name is unknown; the message lists the registered names.
    """
    unknown = []
    for name in names:
        if name not in STAMPS:
            unknown.append(name)
    if unknown:
        raise KeyError(f"unknown stamps {unknown}; registered: {sorted(STAMPS)}")


def run_stamps(
    loaded: xr.Dataset, source: LoadSource, names: Sequence[str]
) -> xr.Dataset:
    """Apply the named stamps in order.

    Raises:
        KeyError: A name is unknown; no stamp runs.
    """
    check_stamps(names)

    for name in names:
        loaded = STAMPS[name](loaded, source)
    return loaded


__all__ = [
    "DEFAULT_STAMPS",
    "STAMPS",
    "LoadSource",
    "Stamp",
    "check_stamps",
    "register_stamp",
    "run_stamps",
]
