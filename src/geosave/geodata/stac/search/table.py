"""Search a STAC GeoParquet table, local or remote, with rustac."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pystac
import rustac
from pystac.utils import datetime_to_str

from geosave.geodata.errors import CollectionNotFoundError
from geosave.geodata.io.storage import absolute_location
from geosave.geodata.io.vector import stac_geoparquet
from geosave.geodata.stac.source import StacSource

from .query import StacQuery

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame
    from geosave.geodata.io.storage import Location


class StacTable:
    """Search one STAC GeoParquet table with rustac and build sources on it.

    Item properties are columns of a table, so a sort or filter names them
    without the `properties.` prefix a STAC API takes.

    Args:
        path: Parquet file as `to_geoparquet` stores it, as a path, URL, or UPath.
        duckdb: Configured rustac session, needed for a bucket behind
            credentials or a custom endpoint. None creates a session.

    Examples:
        >>> table = StacTable("samples/catalog.parquet")
        >>> table.collections()
        {'forest'}
        >>> source = table.source("forest")
    """

    def __init__(
        self,
        path: Location,
        *,
        duckdb: rustac.DuckdbClient | None = None,
    ) -> None:
        """Bind the table, reading nothing yet.

        Args:
            path: Parquet file, as a path, URL, or UPath.
            duckdb: Caller-configured rustac session, or None to create one.
        """
        self._location = absolute_location(path)
        self._duckdb = duckdb if duckdb is not None else rustac.DuckdbClient()

    def search(self, query: StacQuery | dict[str, Any]) -> GeoDataFrame:
        """Run one search to completion as an item table.

        Args:
            query: Query object, or raw search parameters.

        Returns:
            One row per matching Item, read through `gs` as a catalog. Empty
            where nothing matches.

        Examples:
            >>> table.search(StacQuery(collections=["forest"]))["id"].tolist()
            ['forest_20250601', 'forest_20250611']
        """
        return stac_geoparquet.from_item(self.search_items(query))

    def search_items(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]:
        """Run one search over the table as pystac Items, for loaders.

        Args:
            query: Query object, or raw `rustac` search parameters.

        Returns:
            Every matching Item, its asset hrefs absolute.

        Examples:
            >>> query = StacQuery(collections=["forest"])
            >>> len(client.search_items(query.set_filter("eo:cloud_cover <= 10")))
            3
        """
        given = query.to_search_params() if isinstance(query, StacQuery) else query
        params = dict(given)

        # rustac takes one RFC 3339 string, a range as "start/end".
        when = params.get("datetime")
        if isinstance(when, tuple):
            start, end = when
            params["datetime"] = f"{datetime_to_str(start)}/{datetime_to_str(end)}"
        elif when is not None and not isinstance(when, str):
            params["datetime"] = datetime_to_str(when)

        items = []
        for fields in self._duckdb.search(self._location, **params):
            item = pystac.Item.from_dict(fields)
            # A table stores asset hrefs relative to itself.
            item.set_self_href(self._location)
            item.make_asset_hrefs_absolute()
            items.append(item)
        return items

    def collections(self) -> set[str]:
        """Name the collections this table's rows belong to.

        Returns:
            Collection IDs.
        """
        return set(self._described())

    def collection(self, collection: str) -> pystac.Collection:
        """Read one collection of this table.

        Args:
            collection: Collection ID.

        Returns:
            The Collection the table stores under that id, or the one rustac
            derives from its rows where the table stores none.

        Raises:
            CollectionNotFoundError: No stored Collection and no row carries
                `collection`.
        """
        described = self._described()
        if collection not in described:
            raise CollectionNotFoundError(
                f"collection {collection!r} not found in {self._location}; "
                f"call collections() to see what it holds"
            )
        return described[collection]

    def source(self, collection: str) -> StacSource:
        """Build a source for one collection of this table.

        Args:
            collection: Collection ID. Discover them with `collections`.

        Returns:
            Source for `collection`, on every default.

        Raises:
            CollectionNotFoundError: `collection` is not in this table.

        Examples:
            >>> source = table.source("forest").set_config(bands=["red", "nir"])
        """
        self.collection(collection)
        return StacSource(self, collection=collection)

    def _described(self) -> dict[str, pystac.Collection]:
        """Return every collection of the table, keyed by id.

        Returns:
            {
                "<collection id>": the Collection the table stores, or the
                    one derived from the rows carrying that id,
            }
        """
        described = {}
        for fields in self._duckdb.get_collections(self._location):
            described[fields["id"]] = pystac.Collection.from_dict(fields)
        # What a writer stored says more than what rows alone can; the same
        # session reads it, so one configuration reaches a remote table.
        metadata = self._duckdb.query_to_table(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'stac-geoparquet'",
            [self._location],
        )
        for row in pa.table(metadata).to_pylist():
            described.update(stac_geoparquet.parse_collections(row["value"]))
        return described


