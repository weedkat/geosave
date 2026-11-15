"""The Collection and extension schemas on a loaded Dataset."""

import json

import pystac
import pytest
import xarray as xr

from geosave.geodata.stac.stamps import LoadSource
from geosave.geodata.stac.stamps.collection import stamp_collection
from tests.geodata.stac.stamps.conftest import pixels, scene

EO = "https://stac-extensions.github.io/eo/v1.1.0/schema.json"


def collection() -> pystac.Collection:
    return pystac.Collection(
        id="local",
        description="d",
        extent=pystac.Extent(
            pystac.SpatialExtent([[0, 0, 1, 1]]),
            pystac.TemporalExtent([[None, None]]),
        ),
        license="CC-BY-4.0",
    )


def test_collection_and_extensions_are_stated() -> None:
    item = scene()
    item.stac_extensions.append(EO)
    item.set_collection(collection())

    out = stamp_collection(pixels("red"), LoadSource([item]))

    assert out.gs.stac_collection.id == "local"
    assert out.gs.stac_collection.license == "CC-BY-4.0"
    assert json.loads(out.attrs["stac_extensions"]) == [EO]


def test_extensions_are_listed_once() -> None:
    first, second = scene("a"), scene("b")
    first.stac_extensions.append(EO)
    second.stac_extensions.append(EO)

    out = stamp_collection(pixels("red"), LoadSource([first, second]))

    assert json.loads(out.attrs["stac_extensions"]) == [EO]


def test_an_empty_dataset_becomes_a_collection_node() -> None:
    item = scene()
    item.set_collection(collection())

    out = stamp_collection(xr.Dataset(), LoadSource([item]))

    assert out.gs.stac_collection.id == "local"
    assert not out.data_vars


def test_items_without_a_collection_link_state_none() -> None:
    out = stamp_collection(pixels("red"), LoadSource([scene()]))

    assert out.gs.stac_collection is None


def test_items_of_several_collections_are_refused() -> None:
    first, second = scene("a"), scene("b")
    first.collection_id, second.collection_id = "optical", "radar"

    with pytest.raises(ValueError, match="several collections"):
        stamp_collection(pixels("red"), LoadSource([first, second]))
