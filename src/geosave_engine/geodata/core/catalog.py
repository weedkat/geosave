"""The `gs` accessor of an item table: a vector whose rows are STAC Items."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NoReturn

import pandas as pd

from .vector import GeoVector

if TYPE_CHECKING:
    import pystac
    from upath import UPath

    from geosave_engine.geodata.io.storage import Location, StorageOptions


# Columns every STAC Item row carries; their presence makes a frame an item table.
ITEM_COLUMNS = ("type", "stac_version", "id", "assets")


def is_item_table(data: pd.DataFrame) -> bool:
    """Return whether a frame's rows are STAC Items.

    Args:
        data: Any DataFrame.

    Returns:
        True when the columns every STAC Item row carries are all present.
    """
    return all(column in data.columns for column in ITEM_COLUMNS)


class GeoCatalog(GeoVector):
    """GeoSave operations on an item table, read through `gs`.

    An item table is a GeoDataFrame of STAC Items, as `open_vector` reads a
    stac-geoparquet file and `stac.create_items` builds one. It is a vector,
    so `query` and `rasterize` apply. It is written only as stac-geoparquet,
    the format that keeps its assets.

    Examples:
        >>> rows = gs.open_vector("data/catalog.parquet")
        >>> rows.gs.to_geoparquet("data/copy.parquet")
        PosixUPath('data/copy.parquet')
    """

    def to_geoparquet(
        self,
        path: Location,
        *,
        collections: Iterable[pystac.Collection] | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Any,
    ) -> UPath:
        """Write this item table as one stac-geoparquet file.

        Asset hrefs are stored relative to the file, so a table and its assets
        move together. One file holds every Collection its rows name.

        Args:
            path: Output path or URL ending in `.parquet` or `.geoparquet`.
            collections: Collections the rows belong to, stored with the
                extent of their rows. A collection the rows name and this
                omits is stored with its id as its description.
            overwrite: Replace an existing file when true.
            storage_options: Options for the filesystem a URL names.
            **options: Not accepted; stac-geoparquet fixes its own encoding.

        Returns:
            Local writes return a PosixUPath; URL writes return a UPath
            carrying its storage options.

        Raises:
            FileExistsError: The file exists and `overwrite` is false.
            ValueError: The suffix is wrong, a given Collection has no rows,
                or options are passed.

        Examples:
            >>> rows.gs.to_geoparquet("data/catalog.parquet", collections=[forest])
            PosixUPath('data/catalog.parquet')
        """
        if options:
            raise ValueError(
                f"stac-geoparquet fixes its own encoding; {sorted(options)} do not apply"
            )
        # Deferred: only an item table needs the STAC dependencies.
        from geosave_engine.geodata.io.vector import stac_geoparquet

        return stac_geoparquet.write(
            self._data,
            path,
            collections=collections,
            overwrite=overwrite,
            storage_options=storage_options,
        )

    def to_geopackage(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refuse: GeoPackage cannot hold an item table's assets."""
        raise ValueError(
            "an item table is STAC; GeoPackage cannot hold its assets, write "
            "GeoParquet with to_geoparquet"
        )

    def to_geojson(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refuse: GeoJSON features would lose the stac-geoparquet encoding."""
        raise ValueError(
            "an item table is STAC; write it as GeoParquet with to_geoparquet"
        )
