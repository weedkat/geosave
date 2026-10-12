"""Static STAC: an item table as a folder of Catalog, Collection and Item JSON."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pystac

from geosave.geodata.io import storage

from .stac_geoparquet import to_item

if TYPE_CHECKING:
    from upath import UPath

    from geosave.geodata.io.storage import Location, StorageOptions


class UPathStacIO(pystac.StacIO):
    """Read and write STAC JSON on any filesystem UPath reaches.

    Args:
        storage_options: Options for the filesystem the hrefs name.

    Examples:
        >>> io = UPathStacIO({"anon": True})
        >>> pystac.Catalog.from_file("s3://bucket/stac/catalog.json", stac_io=io)
    """

    def __init__(self, storage_options: StorageOptions | None = None) -> None:
        """Bind the filesystem options every href is read and written with."""
        super().__init__()
        self._options = storage_options

    def read_text(self, source: pystac.link.HREF, *args: Any, **kwargs: Any) -> str:
        """Read one JSON document."""
        return storage.location(str(source), self._options).read_text()

    def write_text(
        self, dest: pystac.link.HREF, txt: str, *args: Any, **kwargs: Any
    ) -> None:
        """Write one JSON document, creating its folder."""
        target = storage.location(str(dest), self._options)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(txt)


def write(
    rows: gpd.GeoDataFrame,
    path: Location,
    *,
    collections: Iterable[pystac.Collection] | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> UPath:
    """Write an item table as a self-contained static STAC catalog.

    `rows.gs.to_stac` is the accessor spelling. The folder holds
    `catalog.json`, one `<collection>/collection.json` per Collection and one
    `<collection>/<id>/<id>.json` per Item. Asset hrefs are stored relative
    to their Item's file, so the folder and its assets move together. Read it
    back with `pystac.Catalog.from_file` and `stac.from_item`.

    Args:
        rows: Item table, as `stac.from_item` or `open_vector` returns it.
        path: Folder to write into, as a path, URL, or UPath.
        collections: Collections the rows belong to. They are not changed. A
            collection the rows name and this omits is described by its id.
        overwrite: Replace a catalog already in the folder when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        The root `catalog.json`: a `PosixUPath` locally, else the remote UPath
        carrying its storage options.

    Raises:
        FileExistsError: The folder holds a catalog and `overwrite` is false.
        ValueError: An id repeats within one collection, or a given Collection
            has no rows.

    Examples:
        >>> write(rows, "data/stac", collections=[forest])
        PosixUPath('data/stac/catalog.json')
        >>> write(rows, "s3://bucket/stac", storage_options={"anon": False})
        S3Path('s3://bucket/stac/catalog.json')
    """
    # Deferred: the STAC package imports this module while it loads.
    from geosave.geodata.stac.translate.collection import describe

    folder = storage.location(path, storage_options)
    root_file = folder / "catalog.json"
    if root_file.exists() and not overwrite:
        raise FileExistsError(f"{root_file} exists; pass overwrite=True to replace it")

    items = to_item(rows)
    # Each Item is saved as <collection>/<id>/<id>.json, so a repeat would overwrite one.
    counts = Counter((item.collection_id, item.id) for item in items)
    if repeated := sorted(key for key, count in counts.items() if count > 1):
        raise ValueError(
            f"{len(repeated)} ids repeat within one collection, so their files "
            f"would overwrite each other: {repeated[:5]}; drop the duplicates"
        )

    # Collections hang under the root; Items without one hang on the root itself.
    root = pystac.Catalog(id=folder.name, description=folder.name)
    described = describe(items, collections)
    for collection in described.values():
        root.add_child(collection)
    for item in items:
        # Links to a searched API's root, parent or collection would be read
        # while the Item joins this catalog; it gets its own.
        for rel in (pystac.RelType.ROOT, pystac.RelType.PARENT, pystac.RelType.COLLECTION):
            item.remove_links(rel)
        if item.collection_id is None:
            root.add_item(item)
        else:
            described[item.collection_id].add_item(item)

    if root_file.exists():
        folder.fs.rm(folder.path, recursive=True)
    root.normalize_hrefs(storage.absolute_location(folder))
    root.make_all_asset_hrefs_relative()
    root.save(pystac.CatalogType.SELF_CONTAINED, stac_io=UPathStacIO(storage_options))
    return root_file
