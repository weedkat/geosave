"""Search a STAC API, built in or by URL."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any

import planetary_computer
import pystac
from cql2 import Expr
from pystac_client import Client, ConformanceClasses
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
from urllib3.util import Retry

from geosave.geodata.errors import CollectionNotFoundError

from .query import SearchParams, StacQuery

if TYPE_CHECKING:
    from geosave.geodata import GeoDataFrame

CDSE_URL = "https://stac.dataspace.copernicus.eu/v1/"
CDSE_S3_URL = "https://eodata.dataspace.copernicus.eu"
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"
ELEMENT84_URL = "https://earth-search.aws.element84.com/v1/"


class StacClient:
    """Search one STAC API and inspect its Collections.

    Connects on the first request, so a loader can take `patch_url` without
    one. Built-in providers have constructors such as `planetary_computer()`.
    Search GeoParquet files with `StacTable(path)`.

    Args:
        url: STAC API root URL.
        patch_url: Called on every asset href a loader reads, for a catalog
            whose files need a signature. Searched hrefs stay as the catalog
            publishes them.

    Examples:
        >>> client = StacClient("https://example.org/stac/v1")
        >>> client.collections()
        {'forest'}
        >>> items = StacClient.cdse().search_items(StacQuery(collections=["sentinel-2-l2a"]))
    """

    def __init__(
        self, url: str, *, patch_url: Callable[[str], str] | None = None
    ) -> None:
        self.url = url
        self.patch_url = patch_url
        self._collections: dict[str, pystac.Collection] = {}

    @cached_property
    def _session(self) -> Client:
        return Client.open(self.url, stac_io=_default_io())

    @classmethod
    def cdse(cls) -> StacClient:
        """Connect to the Copernicus Data Space Ecosystem catalog.

        Searching needs no account. Reading pixels signs each `s3://eodata`
        href with the S3 keys in `CDSE_S3_ACCESS_KEY` and
        `CDSE_S3_SECRET_KEY`, and needs the `cloud` extra.

        Returns:
            Client for the CDSE STAC API.
        """
        return cls(CDSE_URL, patch_url=_sign_cdse)

    @classmethod
    def planetary_computer(cls) -> StacClient:
        """Connect to the Microsoft Planetary Computer catalog, signing assets as they load.

        Its files need a SAS token that lasts about a day, so searches keep the
        lasting hrefs a saved table can store, and a loader signs each one.

        Returns:
            Client for the Planetary Computer STAC API.
        """
        return cls(PLANETARY_COMPUTER_URL, patch_url=planetary_computer.sign)

    @classmethod
    def element84(cls) -> StacClient:
        """Connect to the Element84 Earth Search catalog.

        Returns:
            Client for the Earth Search STAC API.
        """
        return cls(ELEMENT84_URL)

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

        A filter selects the same Items on every catalog. A server without the
        Filter extension, such as Earth Search, ignores it and returns every
        Item, so the filter is applied to the returned Items instead.

        Args:
            query: Query object, or raw pystac-client search parameters.

        Returns:
            Every matching item, in the order the API returns them, capped at
            the query's `max_items`.
        """
        params: dict[str, Any] = dict(
            query.to_search_params() if isinstance(query, StacQuery) else query
        )
        if "filter" not in params or self._session.conforms_to(
            ConformanceClasses.FILTER
        ):
            return list(self._session.search(**params).items())

        expr = Expr(params.pop("filter"))
        params.pop("filter_lang", None)
        max_items = params.pop("max_items", None)
        items = []
        for item in self._session.search(**params).items():
            if expr.matches(item.to_dict()):
                items.append(item)
            if max_items is not None and len(items) == max_items:
                break
        return items

    def collections(self) -> set[str]:
        """Name the collections this catalog offers.

        Returns:
            Collection IDs.
        """
        return {collection.id for collection in self._session.get_collections()}

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
                found = self._session.get_collection(collection)
            except KeyError as error:  # a static catalog
                raise missing from error
            except APIError as error:
                # A transport failure carries no status, and is not an answer.
                if getattr(error, "status_code", None) != 404:
                    raise
                raise missing from error
            self._collections[collection] = found
        return self._collections[collection]


def _sign_cdse(href: str) -> str:
    """Presign a CDSE `s3://eodata` href for an HTTPS read; other hrefs pass through."""
    if not href.startswith("s3://eodata/"):
        return href
    access_key = os.environ.get("CDSE_S3_ACCESS_KEY")
    secret_key = os.environ.get("CDSE_S3_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "reading CDSE pixels needs S3 keys in CDSE_S3_ACCESS_KEY and "
            "CDSE_S3_SECRET_KEY"
        )
    s3 = _open_cdse_s3(access_key, secret_key)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": "eodata", "Key": href.removeprefix("s3://eodata/")},
        ExpiresIn=12 * 3600,
    )


@cache
def _open_cdse_s3(access_key: str, secret_key: str) -> Any:
    """Open one S3 client per key pair; a load signs every asset href."""
    import botocore.session
    from botocore.config import Config

    return botocore.session.get_session().create_client(
        "s3",
        endpoint_url=CDSE_S3_URL,
        region_name="default",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _default_io() -> StacApiIO:
    """Configure retries before a catalog session is opened."""
    return StacApiIO(
        max_retries=Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
    )
