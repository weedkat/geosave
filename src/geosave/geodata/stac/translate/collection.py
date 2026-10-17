"""Collection discovery metadata as ordinary raster attributes."""

from __future__ import annotations

import pystac

import geosave.geodata.attrs as attrs


def read_attrs(collection: pystac.Collection) -> dict[str, str]:
    """Translate Collection discovery metadata into plain raster attributes.

    Args:
        collection: Collection as a catalog publishes it.

    Returns:
        Its id, title, description as `summary`, keywords joined by commas,
        first provider as `institution`, and license.

    Examples:
        >>> read_attrs(source.collection)["summary"]
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
    ).to_attrs()
