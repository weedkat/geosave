from datetime import datetime
from types import SimpleNamespace

import pystac
import pytest
from pystac_client import Client, ConformanceClasses
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO

from geosave.geodata.errors import CollectionNotFoundError
from geosave.geodata.stac import StacClient, StacQuery


def connect(monkeypatch, native: Client) -> StacClient:
    """Answer the first request with a local catalog instead of the network."""
    monkeypatch.setattr(Client, "open", lambda url, **options: native)
    return StacClient("https://example.org/stac/v1")


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
        connect(monkeypatch, native).collection("missing")


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


def test_a_client_connects_on_its_first_request(monkeypatch):
    native = Client(id="custom", description="Custom STAC API")
    opened = []

    def open_client(url, **options):
        opened.append((url, options))
        return native

    monkeypatch.setattr(Client, "open", open_client)
    client = StacClient("https://example.org/stac/v1")

    assert opened == []
    assert client.patch_url is None

    monkeypatch.setattr(native, "get_collections", lambda: iter([]))
    client.collections()
    client.collections()

    assert len(opened) == 1
    url, options = opened[0]
    assert url == "https://example.org/stac/v1"
    assert isinstance(options["stac_io"], StacApiIO)


def test_planetary_computer_signs_when_loading_not_when_searching(monkeypatch):
    import planetary_computer

    def refuse(url, **options):
        raise AssertionError("taking the signer must not connect")

    monkeypatch.setattr(Client, "open", refuse)

    client = StacClient.planetary_computer()

    # Searched hrefs stay unsigned, so a saved table outlives the day-long token.
    assert client.patch_url is planetary_computer.sign


def _item(name: str, cloud: float | None) -> pystac.Item:
    properties = {} if cloud is None else {"eo:cloud_cover": cloud}
    return pystac.Item(name, None, None, datetime(2024, 1, 1), properties)


def answer(monkeypatch, native: Client, items: list[pystac.Item]) -> dict:
    """Make the local catalog return `items`, recording what a search asked."""
    asked = {}

    def search(**params):
        asked.update(params)
        return SimpleNamespace(items=lambda: iter(items))

    monkeypatch.setattr(native, "search", search)
    return asked


def test_a_sorted_search_passes_the_sort_to_the_api(monkeypatch):
    native = Client(id="local", description="Local catalog")
    asked = answer(monkeypatch, native, [_item("c", 0.06), _item("a", 0.5)])
    query = StacQuery(collections=["s2"], max_items=2).sort_by("eo:cloud_cover")

    found = connect(monkeypatch, native).search_items(query)

    assert [item.id for item in found] == ["c", "a"]
    assert asked["sortby"] == [{"field": "eo:cloud_cover", "direction": "asc"}]
    assert asked["max_items"] == 2


def test_a_server_that_filters_receives_the_filter(monkeypatch):
    native = Client(id="local", description="Filtering catalog")
    native.add_conforms_to(ConformanceClasses.FILTER.name)
    asked = answer(monkeypatch, native, [_item("clear", 5.0)])
    query = StacQuery(collections=["s2"]).set_filter("eo:cloud_cover <= 20")

    found = connect(monkeypatch, native).search_items(query)

    assert [item.id for item in found] == ["clear"]
    assert asked["filter_lang"] == "cql2-json"
    assert "filter" in asked


def test_a_filter_the_server_ignores_is_applied_to_its_items(monkeypatch):
    # Earth Search lacks the Filter extension and returns every Item.
    native = Client(id="local", description="Non-filtering catalog")
    answered = [_item("cloudy", 90.0), _item("clear", 5.0), _item("hazy", 15.0)]
    asked = answer(monkeypatch, native, answered)
    query = StacQuery(collections=["s2"], max_items=1).set_filter(
        "eo:cloud_cover <= 20"
    )

    found = connect(monkeypatch, native).search_items(query)

    assert [item.id for item in found] == ["clear"]
    assert "filter" not in asked
    assert "max_items" not in asked


def test_search_attaches_only_returned_collections(monkeypatch):
    native = _offering(Client(id="local", description="Local catalog"), "s2")
    entry = _item("scene", None)
    entry.collection_id = "s2"
    entry.geometry = {"type": "Point", "coordinates": [0.0, 0.0]}
    entry.bbox = [0.0, 0.0, 0.0, 0.0]
    client = connect(monkeypatch, native)
    monkeypatch.setattr(client, "search_items", lambda query: [entry])

    found = client.search(StacQuery(collections=["s2", "absent"]))

    assert list(found.gs.collections) == ["s2"]
    found.attrs["stac-geoparquet"]["collections"]["s2"]["description"] = "changed"
    assert client.collection("s2").description == "Rows"


def test_cdse_presigns_eodata_hrefs_with_keys_from_the_environment(monkeypatch):
    pytest.importorskip("botocore")
    monkeypatch.setenv("CDSE_S3_ACCESS_KEY", "access")
    monkeypatch.setenv("CDSE_S3_SECRET_KEY", "secret")
    sign = StacClient.cdse().patch_url
    assert sign is not None

    signed = sign("s3://eodata/Sentinel-2/MSI/L2A/B04_10m.jp2")

    assert signed.startswith(
        "https://eodata.dataspace.copernicus.eu/eodata/Sentinel-2/MSI/L2A/B04_10m.jp2?"
    )
    assert "X-Amz-Signature=" in signed
    assert sign("https://example.org/red.tif") == "https://example.org/red.tif"


def test_cdse_pixels_need_s3_keys(monkeypatch):
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_KEY", raising=False)
    sign = StacClient.cdse().patch_url
    assert sign is not None

    with pytest.raises(RuntimeError, match="CDSE_S3_ACCESS_KEY"):
        sign("s3://eodata/Sentinel-2/MSI/L2A/B04_10m.jp2")
