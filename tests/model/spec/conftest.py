from importlib import import_module

import numpy as np
import pytest
import xarray as xr

from geosave.geodata.core import GeoAnchor
from geosave.model.spec.stac import StacLoadConfig
from odc.geo.geobox import GeoBox
from tests.geodata.conftest import create_test_collection
from geosave.model.spec import StacRecipe

stac_module = import_module("geosave.model.spec.stac")


@pytest.fixture
def coordinate_anchor():
    return GeoAnchor.from_bbox((12, 45, 12.01, 45.01), timespan="2025-01")


@pytest.fixture
def coordinate_grid(coordinate_anchor):
    return GeoBox.from_geopolygon(
        coordinate_anchor.geometry, crs="EPSG:32633", resolution=10
    )


@pytest.fixture
def stac_recipe():
    return StacRecipe.model_validate(
        {"collection": "optical", "endpoints": ["https://primary.test/stac"]}
    )


class _Client:
    def __init__(self, loaded, patch_url=None):
        self.loaded = loaded
        self.patch_url = patch_url
        self.query = None

    def collection(self, collection):
        return create_test_collection(collection, description="Test collection")

    def search_items(self, query):
        self.query = query
        return [self]


@pytest.fixture
def serve(monkeypatch):
    def connect(served: dict[str, object], *, patch_url=None) -> list[str]:
        opened = []

        def open(endpoint):
            opened.append(endpoint)
            answer = served[endpoint]
            if isinstance(answer, Exception):
                raise answer
            return _Client(answer, patch_url)

        def load(items, *, geobox, collection, **options):
            client = items[0]
            return client.loaded.assign_attrs(
                config=StacLoadConfig(**options), query=client.query
            )

        monkeypatch.setattr(stac_module, "load_dataset", load)
        monkeypatch.setattr(stac_module.StacClient, "open", staticmethod(open))
        return opened

    return connect


@pytest.fixture
def red() -> xr.Dataset:
    return xr.Dataset(
        {"red": (("y", "x"), np.zeros((2, 2))), "nir": (("y", "x"), np.ones((2, 2)))}
    )
