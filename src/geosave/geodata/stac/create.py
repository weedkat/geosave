"""Build a Collection, the Item of each saved scene, and their item table."""

from __future__ import annotations

from geosave.geodata.stac.translate import table as stac_table

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import pystac

from geosave.geodata.io.raster.file import Scene

from .translate import item as stac_item
from .translate.item import ItemTime

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame, GeoCatalog
    from geosave.geodata.io.storage import Location

# An Item's id by default: its time and the centre of its grid, which together
# name one scene of a collection.
ID_FORMAT = "{start:%Y%m%dT%H%M%S}_{lat:.4f}_{lon:.4f}"


def create_collection(
    id: str,
    *,
    description: str,
    license: str = "other",
    title: str | None = None,
    keywords: Sequence[str] | None = None,
    providers: Sequence[pystac.Provider] | None = None,
) -> pystac.Collection:
    """Build a Collection that Items are created into.

    Args:
        id: Collection identity, which its Items carry.
        description: What the Collection holds.
        license: SPDX license identifier, or `"other"`.
        title: Short human-readable name.
        keywords: Words a catalog search matches.
        providers: Organizations that produced, processed or host the data.

    Returns:
        Collection covering the whole globe over an unbounded time range,
        until a written table states what its rows cover.

    Examples:
        >>> forest = create_collection("forest", description="Forest samples")
        >>> scenes = ds.gs.to_cog("samples/forest_{time:%Y%m%d}.tif")
        >>> next(create_items(scenes, collection=forest)).collection_id
        'forest'
    """
    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
        pystac.TemporalExtent([[None, None]]),
    )
    return pystac.Collection(
        id=id,
        description=description,
        extent=extent,
        title=title,
        license=license,
        keywords=None if keywords is None else list(keywords),
        providers=None if providers is None else list(providers),
    )


def create_items(
    scenes: Mapping[str, Location] | Scene | Iterable[Scene],
    *,
    id_format: str = ID_FORMAT,
    collection: pystac.Collection | str | None = None,
    datetime: ItemTime | None = None,
) -> Iterator[pystac.Item]:
    """Build the Item listing each scene's saved files, reading their headers only.

    Args:
        scenes: Asset names mapped to saved file paths, or the result of a
            raster writer. File paths are opened for headers only.
        id_format: `GeoAnchor.format` template each Item's id is filled from,
            its fields taken from the Item's own grid and time: `lat`, `lon`,
            `start`, `end`, `res`, `width`, `height` and `epsg`. Text without
            fields names every Item alike.
        collection: Collection, or its id, of every Item. None takes each
            scene's.
        datetime: Time for a scene whose files state none, as one instant or
            a start and end.

    Yields:
        One Item per scene, in order, with one data Asset per file.

    Raises:
        RuntimeError: A scene is still pending; compute it first.

    Warns:
        TimelessItemWarning: A scene states no time and `datetime` is None,
            so its Item spans all of time.

    Examples:
        >>> scenes = ds.gs.to_cog("samples/forest_{time:%Y%m%d}.tif")
        >>> [item.id for item in create_items(scenes, id_format="forest_{start:%Y%m%d}")]
        ['forest_20250601', 'forest_20250611']
        >>> catalog = create_catalog(create_items(scenes))
        >>> items = create_items({"image": "image.tif"}, collection="optical")
    """
    if isinstance(collection, pystac.Collection):
        collection = collection.id
    if isinstance(scenes, Mapping):
        scenes = Scene.open(
            cast("Mapping[str, Location]", scenes), collection=collection
        )
    for scene in [scenes] if isinstance(scenes, Scene) else scenes:
        yield stac_item.from_scene(
            scene, id_format=id_format, collection=collection, datetime=datetime
        )


def create_catalog(
    items: Iterable[pystac.Item], *, collections: Iterable[pystac.Collection] = ()
) -> GeoDataFrame[GeoCatalog]:
    """Lay Items out as an item table, one row per Item.

    Args:
        items: Items, as `create_items` builds them.
        collections: Native definitions stored once on the table, keyed by id.

    Returns:
        Item table in WGS84, in the order given.

    Raises:
        ValueError: Two Items of one collection share an id.

    Examples:
        >>> scenes = ds.gs.to_cog("samples/forest_{time:%Y%m%d}.tif")
        >>> create_catalog(create_items(scenes, id_format="forest_{start:%Y%m%d}"))["id"].tolist()
        ['forest_20250601', 'forest_20250611']
    """
    items = list(items)
    stac_table.validate_ids(items)
    return stac_table.from_items(items, collections=collections)
