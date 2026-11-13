"""Stamp that states the Collection and the schemas the Items declare."""

import xarray as xr

import geosave.geodata.attrs as attrs

from .registry import register_stamp
from .source import LoadSource


@register_stamp("collection")
def stamp_collection(loaded: xr.Dataset, source: LoadSource) -> xr.Dataset:
    """State the Collection, without its links, and the Items' extension schemas.

    Args:
        loaded: Dataset to stamp; an empty Dataset gives a tree's collection node.
        source: Items and Collection that were loaded.

    Returns:
        `loaded` with `StacCollection` attrs.
    """
    fields = None
    if source.collection is not None:
        fields = source.collection.to_dict(
            include_self_link=False, transform_hrefs=False
        )
        fields["links"] = []

    extensions = []
    for item in source.items:
        extensions.extend(item.stac_extensions)

    stated = attrs.StacCollection(
        collection=fields, extensions=list(dict.fromkeys(extensions))
    )
    return attrs.stamp_attrs(loaded, stated)
