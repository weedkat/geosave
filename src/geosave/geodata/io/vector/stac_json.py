"""Persist native Items as a folder of Catalog, Collection and Item JSON."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import pystac
import orjson
from pystac.utils import make_absolute_href

from geosave.geodata.io import storage


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
        """Bind the filesystem options every href is read and written with.

        Args:
            storage_options: Credentials and filesystem options shared by
                reads and writes. None uses each filesystem's defaults.
        """
        super().__init__()
        self._options = storage_options

    def read_text(self, source: pystac.link.HREF, *args: Any, **kwargs: Any) -> str:
        """Read one JSON document.

        Args:
            source: Document path or URL, interpreted with the bound options.
            *args: Extra PySTAC arguments, accepted and unused.
            **kwargs: Extra PySTAC keywords, accepted and unused.

        Returns:
            Document text decoded by UPath's default text encoding.
            JSON is not parsed here.
        """
        return storage.location(str(source), self._options).read_text()

    def write_text(
        self, dest: pystac.link.HREF, txt: str, *args: Any, **kwargs: Any
    ) -> None:
        """Write one JSON document, creating its folder.

        Args:
            dest: Document path or URL, interpreted with the bound options.
            txt: Document text to write without parsing or reformatting.
            *args: Extra PySTAC arguments, accepted and unused.
            **kwargs: Extra PySTAC keywords, accepted and unused.
        """
        target = storage.location(str(dest), self._options)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(txt)


def write(
    items: Iterable[pystac.Item],
    path: Location,
    *,
    collections: Iterable[pystac.Collection] | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
) -> UPath:
    """Write native Items as a folder of static STAC JSON.

    `rows.gs.to_stac_json` is the accessor spelling. The folder holds
    `catalog.json`, one `<collection>/collection.json` per Collection and one
    `<collection>/<id>/<id>.json` per Item. Asset hrefs are stored relative
    to their Item's file. Existing asset files are referenced, not copied;
    move them with the JSON while preserving those relative paths. Read it
    back with `pystac.Catalog.from_file` and `stac.create_catalog`.

    Args:
        items: Native Items; their fields and asset hrefs are not mutated.
        path: Folder to write into, as a path, URL, or UPath.
        collections: Definitions for every named Collection. Unused definitions
            are ignored; input definitions are not changed.
        overwrite: Replace a catalog already in the folder when true.
        storage_options: Options for the filesystem a URL names.

    Returns:
        The root `catalog.json`: a `PosixUPath` locally, else the remote UPath
        carrying its storage options.

    Raises:
        FileExistsError: The folder holds a catalog and `overwrite` is false.
        ValueError: An id repeats within one collection, or a referenced Collection
            definition is missing.

    Examples:
        >>> write(items, "data/stac", collections=[forest])
        PosixUPath('data/stac/catalog.json')
        >>> write(items, "s3://bucket/stac", storage_options={"anon": False})
        S3Path('s3://bucket/stac/catalog.json')
    """
    from ._stac import prepare_collections

    folder = storage.location(path, storage_options)
    root_file = folder / "catalog.json"
    if root_file.exists() and not overwrite:
        raise FileExistsError(f"{root_file} exists; pass overwrite=True to replace it")

    from geosave.geodata.stac.translate.table import validate_ids

    items = list(items)
    validate_ids(items)
    items = [
        pystac.Item.from_dict(item.to_dict(transform_hrefs=False)) for item in items
    ]
    for item in items:
        if item.get_self_href():
            item.make_asset_hrefs_absolute()
            for link in item.links:
                if link.rel not in ("root", "parent", "self", "collection"):
                    link.target = link.get_absolute_href() or link.target

    # Collections hang under the root; Items without one hang on the root itself.
    root = pystac.Catalog(id=folder.name, description=folder.name)
    built = prepare_collections(items, collections)
    for collection in built.values():
        root.add_child(collection)
    for item in items:
        # Links to a searched API's root, parent or collection would be read
        # while the Item joins this catalog; it gets its own.
        for rel in (
            pystac.RelType.ROOT,
            pystac.RelType.PARENT,
            pystac.RelType.COLLECTION,
        ):
            item.remove_links(rel)
        if item.collection_id is None:
            root.add_item(item)
        else:
            built[item.collection_id].add_item(item)

    root.normalize_hrefs(storage.absolute_location(folder))
    root.make_all_asset_hrefs_relative()
    owned = (
        _owned_documents(root_file, storage_options) if root_file.exists() else set()
    )
    for catalog, _, members in root.walk():
        for document in [catalog, *members]:
            target = storage.location(document.get_self_href(), storage_options)
            if target.exists() and storage.absolute_location(target) not in owned:
                raise FileExistsError(f"{target} is not owned by the existing catalog")
    root.save(pystac.CatalogType.SELF_CONTAINED, stac_io=UPathStacIO(storage_options))
    return root_file


def _owned_documents(root_file: UPath, options: StorageOptions | None) -> set[str]:
    """Find existing catalog documents through structural links inside its folder."""
    prefix = storage.absolute_location(root_file.parent).rstrip("/") + "/"
    pending = [storage.absolute_location(root_file)]
    owned: set[str] = set()
    while pending:
        href = pending.pop()
        if href in owned or not href.startswith(prefix):
            continue
        target = storage.location(href, options)
        if not target.exists():
            continue
        fields = orjson.loads(target.read_bytes())
        if "stac_version" not in fields or fields.get("type") not in (
            "Catalog",
            "Collection",
            "Feature",
        ):
            raise FileExistsError(f"{target} is not owned by a valid STAC catalog")
        owned.add(href)
        pending.extend(
            make_absolute_href(link["href"], href)
            for link in fields.get("links", [])
            if link["rel"] in ("child", "item")
        )
    return owned
