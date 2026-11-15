"""Stamp that states the Collection and the schemas the Items declare."""

import xarray as xr

import geosave.geodata.attrs as attrs

from .registry import register_stamp
from .source import LoadSource


@register_stamp("collection")
def stamp_collection(ds: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """State the Items' Collection, without its links, and their extension schemas.

    The Collection is the one the Items link to, as pystac's
    `item.get_collection()` resolves it: searched Items resolve it through
    their catalog, catalog rows carry it, and a hand-built Item names it with
    `item.set_collection(collection)`.

    Args:
        ds: Dataset to stamp; an empty Dataset gives a tree's collection node.
        source: Items that were loaded.

    Returns:
        `ds` with `StacCollection` attrs.

    Raises:
        ValueError: The Items name more than one collection.
    """
    collection_ids = set()
    for item in source.items:
        if item.collection_id is not None:
            collection_ids.add(item.collection_id)
    if len(collection_ids) > 1:
        raise ValueError(
            f"the Items name several collections {sorted(collection_ids)}; "
            "load one collection's Items per call"
        )

    # The first Item that links to its Collection states it for all.
    fields = None
    for item in source.items:
        collection = item.get_collection()
        if collection is not None:
            fields = collection.to_dict(include_self_link=False, transform_hrefs=False)
            fields["links"] = []
            break

    extensions = []
    for item in source.items:
        extensions.extend(item.stac_extensions)

    stated = attrs.StacCollection(
        collection=fields, extensions=list(dict.fromkeys(extensions))
    )
    return attrs.stamp_attrs(ds, stated)
