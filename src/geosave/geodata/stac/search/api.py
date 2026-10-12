"""Search a STAC API, built in or by URL."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import planetary_computer
import pystac
from pystac_client import Client
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
from urllib3.util import Retry

from geosave.geodata.errors import CollectionNotFoundError
from geosave.geodata.io.vector import stac_geoparquet

from geosave.geodata.stac.source import StacSource

from .query import StacQuery

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

CDSE_URL = "https://stac.dataspace.copernicus.eu/v1/"
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"
ELEMENT84_URL = "https://earth-search.aws.element84.com/v1/"

type StacProvider = Literal["planetary_computer", "cdse", "element84"]


class StacClient:
    """Search one STAC catalog and build sources on it.

    Args:
        client: Open pystac-client session.
        patch_url: Called on every asset href when a source loads it, for a
            catalog whose files need a short-lived signature. Searched hrefs
            stay as the catalog publishes them.

    Examples:
        >>> client = StacClient.cdse()
        >>> source = client.source("sentinel-2-l2a")
    """

    def __init__(
        self, client: Client, *, patch_url: Callable[[str], str] | None = None
    ) -> None:
        """Bind the caller-owned session.

        Args:
            client: Open pystac-client session.
            patch_url: Called on every asset href a source loads.
        """
        self._client = client
        self._patch_url = patch_url
        self._collections: dict[str, pystac.Collection] = {}

    @classmethod
    def open(cls, endpoint: StacProvider | str) -> StacClient:
        """Open a built-in provider or a STAC API by URL.

        Args:
            endpoint: Built-in provider name or STAC API URL. A STAC
                GeoParquet table opens with `StacTable` instead.

        Returns:
            Client for the catalog.

        Examples:
            >>> StacClient.open("element84").source("sentinel-2-l2a")
        """
        if endpoint == "planetary_computer":
            return cls.planetary_computer()
        if endpoint == "cdse":
            return cls.cdse()
        if endpoint == "element84":
            return cls.element84()
        return cls(Client.open(endpoint, stac_io=_default_io()))

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
        lasting hrefs a saved table can store, and sources sign each when read.

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

    def search(self, query: StacQuery | dict[str, Any]) -> GeoDataFrame:
        """Run one search to completion as an item table.

        Args:
            query: Query object, or raw search parameters.

        Returns:
            One row per matching Item, read through `gs` as a catalog. Empty
            where nothing matches.

        Examples:
            >>> client.search(StacQuery(collections=["forest"]))["id"].tolist()
            ['forest_20250601', 'forest_20250611']
        """
        return stac_geoparquet.from_item(self.search_items(query))

    def search_items(self, query: StacQuery | dict[str, Any]) -> list[pystac.Item]:
        """Run one search to completion as pystac Items, for loaders.

        Args:
            query: Query object, or raw pystac-client search parameters.

        Returns:
            Every matching item.
        """
        params = query.to_search_params() if isinstance(query, StacQuery) else query
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

    def source(self, collection: str) -> StacSource:
        """Build a source for one collection on this catalog.

        The source starts on every default. Narrow it with `set_config` and
        `set_query`.

        Args:
            collection: Collection ID. Discover them with `collections`.

        Returns:
            Source for `collection`.

        Raises:
            CollectionNotFoundError: `collection` is not on this catalog.

        Examples:
            >>> source = client.source("sentinel-2-l1c").set_config(bands=["B02"])
        """
        self.collection(collection)
        source = StacSource(self, collection=collection)
        if self._patch_url is not None:
            source.set_config(patch_url=self._patch_url)
        return source


def _default_io() -> StacApiIO:
    """Configure retries before a built-in catalog session is opened."""
    return StacApiIO(
        max_retries=Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
    )
