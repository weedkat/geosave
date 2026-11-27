"""Stamp that states the Collection and the schemas the Items declare."""

from collections.abc import Sequence

import pystac
import xarray as xr
from odc.stac import ConversionConfig

import geosave.geodata.attrs as attrs


def stamp_collection(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    stac_cfg: ConversionConfig | None = None,
) -> xr.Dataset:
    """State the Items' Collection, without its links, and their extension schemas.

    The Collection is the one the Items link to, as pystac's
    `item.get_collection()` resolves it: searched Items resolve it through
    their catalog, catalog rows carry it, and a hand-built Item names it with
    `item.set_collection(collection)`.

    Args:
        ds: Dataset to stamp; an empty Dataset gives a tree's collection node.
        items: Items that were loaded.
        stac_cfg: The load's ODC configuration; not needed here.

    Returns:
        `ds` with `StacCollection` attrs.

    Raises:
        ValueError: The Items name more than one collection.
    """
    collection_ids = set()
    for item in items:
        if item.collection_id is not None:
            collection_ids.add(item.collection_id)
    if len(collection_ids) > 1:
        raise ValueError(
            f"the Items name several collections {sorted(collection_ids)}; "
            "load one collection's Items per call"
        )

    # The first Item that links to its Collection states it for all.
    fields = None
    for item in items:
        collection = item.get_collection()
        if collection is not None:
            fields = collection.to_dict(include_self_link=False, transform_hrefs=False)
            fields["links"] = []
            break

    extensions = []
    for item in items:
        extensions.extend(item.stac_extensions)

    stated = attrs.StacCollection(
        collection=fields, extensions=list(dict.fromkeys(extensions))
    )
    return attrs.stamp_attrs(ds, stated)
