"""Build a Collection and the item table of saved rasters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import pystac
import xarray as xr

from geosave.geodata.io.raster import cogs
from geosave.geodata.io.readers import open_raster
from geosave.geodata.io.storage import absolute_location
from geosave.geodata.io.vector import stac_geoparquet

from . import translate
from .translate.item import ItemTime, Scene

if TYPE_CHECKING:

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.io.storage import Location, StorageOptions


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
        >>> files = ds.gs.to_cog("samples/forest_{time:%Y%m%d}.tif")
        >>> create_item(files, collection=forest)["collection"].unique().tolist()
        ['forest']
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


def create_item(
    files: Iterable[Location] | Mapping[str, Location],
    *,
    collection: pystac.Collection | str | None = None,
    id: str | None = None,
    datetime: ItemTime | None = None,
    storage_options: StorageOptions | None = None,
) -> GeoDataFrame:
    """Build the item table of saved rasters, reading their headers only.

    Files sharing a collection, an instant and a grid are one Item, one Asset
    each, named by a single-band file's band or `"image"`. Asset keys mapped
    to files make one Item whose assets the caller names.

    Args:
        files: Saved raster files (GeoTIFF, JPEG 2000 or PNG) as paths, URLs
            or the UPaths `to_cog` returned, or asset keys mapped to files, as
            `{"optical": "s0/optical.tif", "label": "s0/label.tif"}`.
        collection: Collection, or its id, of every Item. None takes the ACDD
            `id` each file states, as `to_cog` writes a stack's layer.
        id: Item id. None names each Item after its file, or the stem its
            files share. A `GeoAnchor.format` template, as
            `"{lat:.2f}N_{start:%Y%m%d}"`, is filled from each Item's raster;
            plain text names one Item, or prefixes each of several, as
            `"s0/forest_20250601"`.
        datetime: Time for undated files, as one instant or a start and end.
            None dates named assets by their `time` coordinates, and undated
            files by the span the dated ones cover.
        storage_options: Options for the filesystem a URL names.

    Returns:
        Item table, one row per Item, in the order the files first name
        them. No pixel is read.

    Raises:
        ValueError: No file is given; a file holds several dates; two files
            of one scene share an asset key; ids repeat; or no file is dated
            and `datetime` is None.

    Examples:
        >>> create_item(sorted(Path("labels").glob("*.tif")), collection="label")["id"].tolist()
        ['north', 'south']
        >>> create_item(ds.gs.to_cog("samples/forest_{time:%Y%m%d}.tif"))["id"].tolist()
        ['forest_20250601', 'forest_20250611']
        >>> create_item({"optical": "s0/optical.tif", "label": "s0/label.tif"}, id="s0")["id"].tolist()
        ['s0']
    """
    if isinstance(files, Mapping):
        named = cast("Mapping[str, Location]", files)
        hrefs = {key: absolute_location(path) for key, path in named.items()}
        headers = {
            absolute_location(path): _header(path, storage_options)
            for path in named.values()
        }
        scene = Scene(_identity(collection), None, None, hrefs)
        item_id = scene.stem if id is None else id
        items = [translate.item.write(scene, headers, id=item_id, when=datetime)]
    else:
        headers = {
            absolute_location(path): _header(path, storage_options) for path in files
        }
        if not headers:
            raise ValueError("create_item needs saved raster files")
        found = translate.item.scenes(headers, collection=_identity(collection))
        undated = datetime or translate.item.span(headers.values())
        items = [
            translate.item.write(
                scene,
                headers,
                id=_scene_id(id, scene, headers, several=len(found) > 1),
                when=None if scene.datetime is not None else undated,
            )
            for scene in found
        ]

    # A template that leaves out time would name several Items alike.
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"id {id!r} gives several Items the same id {ids}; add a time field "
            f"such as {{start:%Y%m%d}}"
        )
    return stac_geoparquet.from_item(items)


def _scene_id(
    template: str | None,
    scene: Scene,
    headers: Mapping[str, xr.Dataset],
    *,
    several: bool,
) -> str:
    """Name a scene's Item as `create_item` documents its `id`."""
    if template is None:
        return scene.stem
    # Fields fill per scene; plain text names one Item or prefixes several stems.
    if cogs.fields_named(template):
        first = next(iter(scene.assets.values()))
        return headers[first].gs.anchor.format(template)
    return f"{template}/{scene.stem}" if several else template


def _header(path: Location, storage_options: StorageOptions | None) -> xr.Dataset:
    """Read a saved raster's grid, variables, times and attrs; its pixels stay unread."""
    with open_raster(path, storage_options=storage_options) as raster:
        return raster


def _identity(collection: pystac.Collection | str | None) -> str | None:
    """Return the id an Item of this Collection carries, None for no Collection."""
    if collection is None or isinstance(collection, str):
        return collection
    return collection.id
