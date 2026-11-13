"""Registry of named stamps that write STAC metadata onto a loaded Dataset."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pystac
import xarray as xr
from odc.stac.model import ParsedItem

import geosave.geodata.attrs as attrs
from geosave.geodata.stac.item import read_instant, select_item_properties


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


def read_scalar_properties(entry: pystac.Item) -> dict[str, Any]:
    """Read the acquisition properties of an Item that a coordinate can hold."""
    scalars = {}
    for key, value in select_item_properties(entry.properties).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            scalars[key] = value
    return scalars


def order_items_by_step(
    loaded: xr.Dataset, items: Sequence[pystac.Item]
) -> list[pystac.Item] | None:
    """Return the Items in `time` order, or None unless each step is one Item.

    ODC labels each Item at its instant, so a label maps to one Item only when
    the Items' instants are distinct and the labels are exactly those instants.
    """
    if "time" not in loaded.dims or loaded.sizes["time"] != len(items):
        return None

    by_instant = {}
    for entry in items:
        instant = read_instant(entry)
        if instant is None or instant in by_instant:
            return None
        by_instant[instant] = entry

    ordered = []
    for label in pd.DatetimeIndex(loaded.time.values):
        if label not in by_instant:
            return None
        ordered.append(by_instant[label])
    return ordered


def fill_missing(values: list[Any]) -> np.ndarray:
    """Build an array from per-Item values, marking missing ones.

    Missing values (None) become NaN in a numeric array and None otherwise.
    """
    if None not in values:
        return np.array(values)

    present = [value for value in values if value is not None]
    numeric = bool(present)
    for value in present:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            numeric = False
    if numeric:
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    filled = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        filled[index] = value
    return filled


@register_stamp("properties")
def stamp_properties(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """Write each scalar Item property as a coordinate marked `stac_property`.

    When every `time` step is exactly one Item, a property the Items disagree on
    follows `time`, missing where an Item lacks it. Otherwise only properties
    every Item states alike are written, as scalars. Lists and dicts are skipped.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items that were loaded.

    Returns:
        `loaded` with one coordinate per property; pixels are not read.
    """
    stepped = order_items_by_step(loaded, source.items)
    entries = source.items if stepped is None else stepped

    readings = []
    keys = {}
    for entry in entries:
        reading = read_scalar_properties(entry)
        readings.append(reading)
        for key in reading:
            keys[key] = None

    for key in keys:
        values = []
        for reading in readings:
            values.append(reading.get(key))
        shared = all(key in reading for reading in readings) and all(
            value == values[0] for value in values
        )

        if shared:
            loaded = loaded.gs.write_stac_property(key, values[0])
        elif stepped is not None:
            loaded = loaded.gs.write_stac_property(key, fill_missing(values))
    return loaded


def build_stac_collection(
    items: Sequence[pystac.Item], collection: pystac.Collection | None
) -> attrs.StacCollection:
    """Read Collection metadata and the schemas declared by Items.

    Args:
        items: Nonempty sequence of source Items.
        collection: Collection to retain without its catalog links, or None.

    Returns:
        Metadata for a Dataset root or a DataTree's collection node.
        Observation properties are stored separately on their coordinates.
    """
    if not items:
        raise ValueError("source metadata needs at least one Item")

    fields = None
    if collection is not None:
        fields = collection.to_dict(include_self_link=False, transform_hrefs=False)
        fields["links"] = []
    extensions = []
    for entry in items:
        extensions.extend(entry.stac_extensions)
    return attrs.StacCollection(
        collection=fields,
        extensions=list(dict.fromkeys(extensions)),
    )


@register_stamp("collection")
def stamp_collection(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """State the Collection and the STAC extensions the Items declare.

    Args:
        loaded: Dataset returned by the ODC load.
        source: Items and Collection that were loaded.

    Returns:
        `loaded` with the collection stated on its attrs.
    """
    return attrs.stamp_attrs(
        loaded, build_stac_collection(source.items, source.collection)
    )


# Importing the extension modules registers their stamps. This stays at the end
# because those modules import register_stamp and LoadSource from here.
from geosave.geodata.stac.extensions import classification, eo, raster  # noqa: E402,F401
