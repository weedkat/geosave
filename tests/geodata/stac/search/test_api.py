from datetime import datetime
from types import SimpleNamespace

import pystac
from pystac_client import Client
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
import pytest

from geosave.geodata.stac import StacClient, StacQuery
from geosave.geodata.errors import CollectionNotFoundError


def test_wrapping_a_client_preserves_its_transport():
    transport = StacApiIO(headers={"X-Catalog": "private"}, timeout=7)
    client = Client(id="local", description="Custom transport")
    client._stac_io = transport

    StacClient(client)

    assert client._stac_io is transport
    assert transport.session.headers["X-Catalog"] == "private"


def _not_found(status: int | None):
    error = APIError('{"code":"NotFound","description":"Not Found"}')
    error.status_code = status

    def raise_it(_name):
        raise error

    return raise_it


def _missing_key(name):
    raise KeyError(f"Collection {name} not found on catalog")


@pytest.mark.parametrize("lookup", [_not_found(404), _missing_key])
def test_missing_collection_raises_a_lookup_error(monkeypatch, lookup):
    # A STAC API answers 404; a static catalog raises KeyError.
    native = Client(id="local", description="Local catalog")
    monkeypatch.setattr(native, "get_collection", lookup)

    with pytest.raises(CollectionNotFoundError, match="missing"):
        StacClient(native).collection("missing")


def test_another_api_error_is_not_a_missing_collection(monkeypatch):
    native = Client(id="local", description="Local catalog")
    monkeypatch.setattr(native, "get_collection", _not_found(503))

    with pytest.raises(APIError):
        StacClient(native).collection("busy")


@pytest.mark.parametrize("provider", ["planetary_computer", "cdse", "element84"])
def test_named_provider_uses_its_stac_client_constructor(monkeypatch, provider):
    expected = object()
    monkeypatch.setattr(
        StacClient,
        provider,
        classmethod(lambda cls: expected),
    )

    assert StacClient.open(provider) is expected


def _offering(native: Client, collection: str) -> Client:
    found = pystac.Collection(
        collection,
        "Rows",
        pystac.Extent.from_dict(
            {
                "spatial": {"bbox": [[-180, -90, 180, 90]]},
                "temporal": {"interval": [[None, None]]},
            }
        ),
    )
    native.get_collection = lambda _name: found
    return native


def test_a_client_hands_its_url_patch_to_every_source():
    def sign(href: str) -> str:
        return f"{href}?token"

    native = _offering(Client(id="local", description="Local catalog"), "forest")

    source = StacClient(native, patch_url=sign).source("forest")

    assert source.config.patch_url is sign


def test_planetary_computer_signs_when_loading_not_when_searching(monkeypatch):
    import planetary_computer

    opened = {}

    def open_client(url, **options):
        opened.update(options)
        return _offering(Client(id="pc", description="Planetary Computer"), "s2")

    monkeypatch.setattr(Client, "open", open_client)

    client = StacClient.planetary_computer()

    # Searched hrefs stay unsigned, so a saved table outlives the day-long token.
    assert "modifier" not in opened
    assert client.source("s2").config.patch_url is planetary_computer.sign


def _item(name: str, cloud: float | None) -> pystac.Item:
    properties = {} if cloud is None else {"eo:cloud_cover": cloud}
    return pystac.Item(name, None, None, datetime(2024, 1, 1), properties)


def test_a_sorted_search_passes_the_sort_to_the_api(monkeypatch):
    native = Client(id="local", description="Local catalog")
    answered = [_item("c", 0.06), _item("a", 0.5)]
    asked = {}

    def search(**params):
        asked.update(params)
        return SimpleNamespace(items=lambda: iter(answered))

    monkeypatch.setattr(native, "search", search)
    query = StacQuery(collections=["s2"], max_items=2).sort_by("eo:cloud_cover")

    found = StacClient(native).search_items(query)

    assert [item.id for item in found] == ["c", "a"]
    assert asked["sortby"] == [{"field": "eo:cloud_cover", "direction": "asc"}]
    assert asked["max_items"] == 2


def test_search_attaches_only_returned_collections(monkeypatch):
    native = _offering(Client(id="local", description="Local catalog"), "s2")
    entry = _item("scene", None)
    entry.collection_id = "s2"
    entry.geometry = {"type": "Point", "coordinates": [0.0, 0.0]}
    entry.bbox = [0.0, 0.0, 0.0, 0.0]
    client = StacClient(native)
    monkeypatch.setattr(client, "search_items", lambda query: [entry])

    found = client.search(StacQuery(collections=["s2", "absent"]))

    assert list(found.gs.collections) == ["s2"]
    found.attrs["stac:collections"]["s2"]["description"] = "changed"
    assert client.collection("s2").description == "Rows"


def test_empty_search_fetches_no_collections(monkeypatch):
    native = Client(id="local", description="Local catalog")
    client = StacClient(native)
    monkeypatch.setattr(client, "search_items", lambda query: [])

    def refuse(name):
        raise AssertionError("empty search fetched a Collection")

    monkeypatch.setattr(client, "collection", refuse)

    found = client.search(StacQuery(collections=["s2"]))

    assert found.empty
    assert found.gs.collections == {}
