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

from .api import StacClient
from .query import StacQuery

if TYPE_CHECKING:
    from geosave.geodata.io.storage import Location


class StacTable(StacClient):
    """A STAC GeoParquet table searched with rustac, opened by `StacClient.open`.

    Item properties are columns of a table, so a sort or filter names them
    without the `properties.` prefix a STAC API takes.

    Args:
        path: Parquet file as `to_geoparquet` stores it, as a path, URL, or UPath.
        duckdb: Configured rustac session, needed for a bucket behind
            credentials or a custom endpoint. None creates a session.

    Examples:
        >>> table = StacClient.open("samples/catalog.parquet")
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
        self._patch_url = None
        self._duckdb = duckdb if duckdb is not None else rustac.DuckdbClient()

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
        return set(self._read_collections())

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
        found = self._read_collections()
        if collection not in found:
            raise CollectionNotFoundError(
                f"collection {collection!r} not found in {self._location}; "
                f"call collections() to see what it holds"
            )
        return found[collection]

    def _read_collections(self) -> dict[str, pystac.Collection]:
        """Read the table's collections, the stored ones over the derived.

        Returns:
            Collections keyed by id.
        """
        return self._derived_collections() | self._stored_collections()

    def _derived_collections(self) -> dict[str, pystac.Collection]:
        """Read the Collections rustac derives from the rows' `collection` ids."""
        return {
            fields["id"]: pystac.Collection.from_dict(fields)
            for fields in self._duckdb.get_collections(self._location)
        }

    def _stored_collections(self) -> dict[str, pystac.Collection]:
        """Read the Collections a writer stored in the file metadata.

        They say more than the rows alone can. The same session reads them, so
        one configuration reaches a remote table.
        """
        metadata = self._duckdb.query_to_table(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'stac-geoparquet'",
            [self._location],
        )
        stored = {}
        for row in pa.table(metadata).to_pylist():
            stored.update(stac_geoparquet.parse_collections(row["value"]))
        return stored
