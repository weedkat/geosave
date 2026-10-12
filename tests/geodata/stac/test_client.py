from pystac_client import Client
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
import pytest

from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.errors import CollectionNotFoundError


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
