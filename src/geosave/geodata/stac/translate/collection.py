"""A STAC Collection read as the ACDD attrs on a Dataset's root, and the extent its Items cover."""

from __future__ import annotations

from collections.abc import Iterable

import pystac

import geosave.geodata.attrs as attrs


def read(collection: pystac.Collection) -> attrs.ACDD:
    """Read what a Collection says about its data as ACDD attrs.

    Args:
        collection: Collection as a catalog publishes it.

    Returns:
        Its id, title, description as `summary`, keywords joined by commas,
        first provider as `institution`, and license.

    Examples:
        >>> read(source.collection).summary
        'Sentinel-2 Level-2A surface reflectance'
    """
    providers = collection.providers or []
    return attrs.ACDD(
        id=collection.id,
        title=collection.title,
        summary=collection.description,
        keywords=", ".join(collection.keywords) if collection.keywords else None,
        institution=providers[0].name if providers else None,
        license=collection.license,
    )


def describe(
    items: list[pystac.Item], collections: Iterable[pystac.Collection] | None
) -> dict[str, pystac.Collection]:
    """Build the Collection each collection id of the Items names, with their extent.

    Args:
        items: Items whose Collections are described.
        collections: Collections given for the Items. They are not changed.
            An id the Items name and this omits is described by its id.

    Returns:
        {
            "<collection id>": new Collection without links, its extent the
                extent of its Items,
        }

    Raises:
        ValueError: A given Collection has no Item.
    """
    given = {collection.id: collection for collection in collections or ()}
    if unused := sorted(set(given) - {item.collection_id for item in items}):
        raise ValueError(f"no row belongs to the given collections {unused}")

    described = {}
    for collection_id in dict.fromkeys(item.collection_id for item in items):
        if collection_id is None:
            continue
        members = [item for item in items if item.collection_id == collection_id]
        extent = pystac.Extent.from_items(members)
        if collection_id in given:
            # A pystac-client Collection cannot be cloned off its API, so copy its fields.
            fields = given[collection_id].to_dict(transform_hrefs=False)
            collection = pystac.Collection.from_dict(fields)
            collection.extent = extent
        else:
            collection = pystac.Collection(collection_id, collection_id, extent)
        collection.clear_links()
        described[collection_id] = collection
    return described
