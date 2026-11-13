"""Registry of named stamps that write STAC metadata onto a loaded Dataset."""

from collections.abc import Callable, Mapping, Sequence
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


def write_band_attrs(
    loaded: xr.Dataset,
    source: LoadSource,
    read: Callable[[pystac.Item, str], Mapping[str, Any]],
) -> xr.Dataset:
    """Write, per loaded variable, the attrs every Item's band states alike.

    The variable's own attrs win over STAC's; attrs Items disagree on are dropped.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items that were loaded.
        read: Reads the attrs one Item states for the band ODC loads as a name.

    Returns:
        `loaded` with the agreed attrs added to each variable.
    """
    stamped = {}
    for name, variable in loaded.data_vars.items():
        observations = []
        for item in source.items:
            observations.append(xr.Dataset(attrs=dict(read(item, str(name)))))
        stated = xr.merge(observations, combine_attrs="drop_conflicts").attrs

        stamped[name] = variable.assign_attrs({**stated, **variable.attrs})
    return loaded.assign(stamped)


def read_title(item: pystac.Item, name: str) -> str | None:
    """Read the title naming the band ODC loads as `name`.

    Only a one-band asset's title names its band, and only when the band has no
    description of its own.
    """
    from geosave.geodata.stac.band import find_asset, read_band  # cycle

    found = find_asset(item, name)
    if found is None:
        return None
    asset = item.assets[found[0]]

    count = 0
    for listing in ("bands", "eo:bands", "raster:bands"):
        count = max(count, len(asset.extra_fields.get(listing, [])))
    if count > 1 or "description" in read_band(item, name):
        return None
    return asset.title


@register_stamp("title")
def stamp_title(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Name a band after its asset's title, where the asset holds only that band.

    The variable's own attrs win over STAC's. The name is kept only when every
    Item that states one agrees.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items that were loaded.

    Returns:
        `loaded` with `long_name` stated.
    """

    def read(item: pystac.Item, name: str) -> dict[str, str]:
        title = read_title(item, name)
        return {} if title is None else {"long_name": title}

    return write_band_attrs(loaded, source, read)


# Importing the extension modules registers their stamps. This stays at the end
# because those modules import register_stamp and LoadSource from here.
from geosave.geodata.stac.extensions import classification, eo, raster  # noqa: E402,F401
