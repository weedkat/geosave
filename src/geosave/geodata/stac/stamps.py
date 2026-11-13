"""Registry of named stamps that write STAC metadata onto a loaded Dataset."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pystac
import xarray as xr
from odc.stac.model import ParsedItem


@dataclass(frozen=True)
class LoadSource:
    """The STAC side of a load: what a stamp may read.

    Args:
        items: Items that were loaded.
        collection: Collection the Items belong to, if known.
        groupby: ODC grouping the load used, a name or a function.
    """

    items: Sequence[pystac.Item]
    collection: pystac.Collection | None
    groupby: str | Callable[[pystac.Item, ParsedItem, int], Any]


type Stamp = Callable[[xr.Dataset, LoadSource], xr.Dataset]

STAMPS: dict[str, Stamp] = {}

DEFAULT_STAMPS = (
    "raster",
    "eo",
    "classification",
    "rgb",
    "title",
    "properties",
    "collection",
)


def register_stamp(name: str) -> Callable[[Stamp], Stamp]:
    """Register a stamp under a name.

    Args:
        name: Name callers pass in ``stamps=``.

    Returns:
        Decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If the name is already registered.
    """

    def decorate(stamp: Stamp) -> Stamp:
        if name in STAMPS:
            raise ValueError(f"Stamp {name!r} is already registered")
        STAMPS[name] = stamp
        return stamp

    return decorate


def check_stamps(names: Sequence[str]) -> None:
    """Check that every name is registered.

    Args:
        names: Stamp names to check.

    Raises:
        KeyError: If a name is unknown; the message lists the registered names.
    """
    unknown = []
    for name in names:
        if name not in STAMPS:
            unknown.append(name)
    if unknown:
        raise KeyError(f"Unknown stamps {unknown}; registered: {sorted(STAMPS)}")


def run_stamps(
    loaded: xr.Dataset,
    source: LoadSource,
    names: Sequence[str],
) -> xr.Dataset:
    """Apply stamps in order; a later stamp overrides an earlier one.

    Args:
        loaded: Dataset returned by the ODC load.
        source: STAC Items and Collection behind the load.
        names: Stamp names, applied first to last.

    Returns:
        Dataset after every stamp.

    Raises:
        KeyError: If a name is unknown.
    """
    check_stamps(names)

    for name in names:
        loaded = STAMPS[name](loaded, source)
    return loaded


# Importing the extension modules registers their stamps. This stays at the end
# because those modules import register_stamp and LoadSource from here.
