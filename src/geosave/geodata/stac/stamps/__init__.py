"""Stamps: named steps that write what STAC documents state onto a loaded Dataset.

A stamp takes the loaded Dataset and the Items behind it and returns the
Dataset with metadata written, reading no pixels. `load_dataset` runs the
stamps named in `stamps=`, in order; a later stamp overrides attrs an earlier
one wrote. Each stamp registers itself with `@register_stamp` where it is
defined; importing this package imports those modules.

Examples:
    >>> scene = stac.load_dataset(items, geobox=grid, stamps=[*DEFAULT_STAMPS, "sentinel_2"])
"""

from . import bands, collection  # noqa: F401  register their stamps
from .registry import STAMPS, check_stamps, register_stamp, run_stamps
from .source import LoadSource, Stamp

DEFAULT_STAMPS = (
    "raster",
    "eo",
    "classification",
    "rgb",
    "title",
    "collection",
)

__all__ = [
    "DEFAULT_STAMPS",
    "STAMPS",
    "LoadSource",
    "Stamp",
    "check_stamps",
    "register_stamp",
    "run_stamps",
]
