"""Shared STAC publication policy for JSON and GeoParquet writers."""

from collections.abc import Iterable

import pystac


def prepare_collections(
    items: list[pystac.Item], collections: Iterable[pystac.Collection] | None
) -> dict[str, pystac.Collection]:
    """Copy referenced Collection definitions with the extent of their Items.

    Args:
        items: Items naming their collection ids.
        collections: Collections to copy fields from, by id. They are not
            changed. Every referenced id needs a definition; unused definitions
            are ignored.

    Returns:
        Collection id mapped to a copy with updated extent and absolute asset
        hrefs. Catalog hierarchy links are removed; descriptive links remain.

    Raises:
        ValueError: A referenced Collection definition is missing.
    """
    by_id = {collection.id: collection for collection in collections or ()}
    grouped: dict[str, list[pystac.Item]] = {}
    for item in items:
        if item.collection_id is not None:
            grouped.setdefault(item.collection_id, []).append(item)

    missing = grouped.keys() - by_id.keys()
    if missing:
        raise ValueError(
            f"missing Collection definitions: {sorted(missing)}; pass collections= when creating or publishing the catalog"
        )

    built = {}
    for collection_id, members in grouped.items():
        # Client Collection subclasses require an API root; publish native values.
        fields = by_id[collection_id].to_dict(transform_hrefs=False)
        collection = pystac.Collection.from_dict(fields)
        collection.extent = pystac.Extent.from_items(members)

        collection.make_asset_hrefs_absolute()
        for link in collection.links:
            if link.rel not in ("root", "parent", "self", "item", "child"):
                link.target = link.get_absolute_href() or link.target
        for rel in ("root", "parent", "self", "item", "child"):
            collection.remove_links(rel)

        built[collection_id] = collection

    return built
