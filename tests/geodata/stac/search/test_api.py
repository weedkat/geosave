import pystac
from pystac_client import Client
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
import pytest

from geosave.geodata.stac import StacClient
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
