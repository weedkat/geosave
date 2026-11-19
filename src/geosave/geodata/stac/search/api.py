"""Search a STAC API, built in or by URL."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import planetary_computer
import pystac
from pystac_client import Client
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
from urllib3.util import Retry

from geosave.geodata.errors import CollectionNotFoundError

from .query import SearchParams, StacQuery

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

CDSE_URL = "https://stac.dataspace.copernicus.eu/v1/"
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"
ELEMENT84_URL = "https://earth-search.aws.element84.com/v1/"


class StacClient:
    """Search one STAC API and inspect its Collections.

    Use `open(url)` for a STAC API, or a named provider constructor such as
    `planetary_computer()`. Search GeoParquet files with `StacTable(path)`.

    Args:
        client: Open pystac-client session.
        patch_url: Called on every asset href when passed to a loader, for a
            catalog whose files need a short-lived signature. Searched hrefs
            stay as the catalog publishes them.

    Examples:
        >>> client = StacClient.cdse()
        >>> items = client.search_items(StacQuery(collections=["sentinel-2-l2a"]))
    """

    def __init__(
        self, client: Client, *, patch_url: Callable[[str], str] | None = None
    ) -> None:
        """Bind the caller-owned session.

        Args:
            client: Open pystac-client session.
            patch_url: Called on every asset href a loader reads.
        """
        self._client = client
        self._patch_url = patch_url
        self._collections: dict[str, pystac.Collection] = {}

    @classmethod
    def open(cls, url: str) -> StacClient:
        """Connect to a STAC API at its root URL.

        Args:
            url: STAC API root URL. For built-in providers, use
                `planetary_computer()`, `cdse()` or `element84()` instead.
                For a GeoParquet path or URL, use `StacTable(path)`.

        Returns:
            Client for the STAC API, without provider-specific asset signing.

        Examples:
            >>> client = StacClient.open("https://example.org/stac/v1")
            >>> client.collections()
            {'forest'}
        """
        return cls(Client.open(url, stac_io=_default_io()))

    @classmethod
    def cdse(cls) -> StacClient:
        """Connect to the Copernicus Data Space Ecosystem catalog.

        Returns:
            Client for the CDSE STAC API.
        """
        return cls(Client.open(CDSE_URL, stac_io=_default_io()))

    @classmethod
    def planetary_computer(cls) -> StacClient:
        """Connect to the Microsoft Planetary Computer catalog, signing assets as they load.

        Its files need a SAS token that lasts about a day, so searches keep the
        lasting hrefs a saved table can store, and a loader signs each when given patch_url=client.patch_url.

        Returns:
            Client for the Planetary Computer STAC API.
        """
        return cls(
            Client.open(PLANETARY_COMPUTER_URL, stac_io=_default_io()),
            patch_url=planetary_computer.sign,
        )

    @classmethod
    def element84(cls) -> StacClient:
        """Connect to the Element84 Earth Search catalog.

        Returns:
            Client for the Earth Search STAC API.
        """
        return cls(Client.open(ELEMENT84_URL, stac_io=_default_io()))

    def search(self, query: StacQuery | SearchParams) -> GeoDataFrame:
        """Run one search to completion as an item table.

        Args:
            query: Query object, or raw search parameters.

        Returns:
            One row per matching Item, read through `gs` as a catalog,
            carrying the Collections its rows name, fetched from the service.
            Empty where nothing matches.

        Examples:
            >>> client.search(StacQuery(collections=["forest"]))["id"].tolist()
            ['forest_20250601', 'forest_20250611']
        """
        from geosave.geodata.stac.catalog import create_catalog

        items = self.search_items(query)
        collection_ids = dict.fromkeys(
            item.collection_id for item in items if item.collection_id
        )
        collections = []
        for collection_id in collection_ids:
            collections.append(self.collection(collection_id))
        return create_catalog(items, collections=collections)

    def search_items(self, query: StacQuery | SearchParams) -> list[pystac.Item]:
        """Run one search to completion as pystac Items, for loaders.

        Args:
            query: Query object, or raw pystac-client search parameters.

        Returns:
            Every matching item, in the order the API returns them, capped at
            the query's `max_items`.
        """
        params: dict[str, Any] = dict(
            query.to_search_params() if isinstance(query, StacQuery) else query
        )
        return list(self._client.search(**params).items())

    def collections(self) -> set[str]:
        """Name the collections this catalog offers.

        Returns:
            Collection IDs.
        """
        return {collection.id for collection in self._client.get_collections()}

    def collection(self, collection: str) -> pystac.Collection:
        """Fetch one collection, reusing it after the first read.

        Args:
            collection: Collection ID.

        Returns:
            The collection as the catalog publishes it.

        Raises:
            CollectionNotFoundError: `collection` is not on this catalog.
        """
        if collection not in self._collections:
            missing = CollectionNotFoundError(
                f"collection {collection!r} not found on this STAC endpoint; "
                f"call collections() to see what is available"
            )
            try:
                found = self._client.get_collection(collection)
            except KeyError as error:  # a static catalog
                raise missing from error
            except APIError as error:
                # A transport failure carries no status, and is not an answer.
                if getattr(error, "status_code", None) != 404:
                    raise
                raise missing from error
            self._collections[collection] = found
        return self._collections[collection]

    @property
    def patch_url(self) -> Callable[[str], str] | None:
        """Return the provider signer to pass to load_dataset.

        Search Items retain durable hrefs; the loader signs at read time.
        """
        return self._patch_url


def _default_io() -> StacApiIO:
    """Configure retries before a built-in catalog session is opened."""
    return StacApiIO(
        max_retries=Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
    )
