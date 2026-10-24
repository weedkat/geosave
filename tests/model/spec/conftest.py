from importlib import import_module

import numpy as np
import pytest
import xarray as xr

from geosave.geodata.core import GeoAnchor
from geosave.geodata.stac.source import StacSourceConfig
from geosave.model.spec import StacRecipe

stac_module = import_module("geosave.model.spec.stac")


@pytest.fixture
def coordinate_anchor():
    return GeoAnchor.from_coordinates(
        45,
        12,
        4,
        10,
        crs="EPSG:32633",
        timespan="2025-01",
    )


@pytest.fixture
def stac_recipe():
    return StacRecipe.model_validate(
        {"collection": "optical", "endpoints": ["https://primary.test/stac"]}
    )


class _Source:
    def __init__(self, loaded, patch_url=None):
        self.loaded = loaded
        self.config = StacSourceConfig(patch_url=patch_url)

    def load(self, anchor):
        return self.loaded.assign_attrs(config=self.config, query=self.query)


class _Client:
    def __init__(self, loaded, patch_url=None):
        self.loaded = loaded
        self.patch_url = patch_url

    def source(self, collection):
        return _Source(self.loaded, self.patch_url)


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

        monkeypatch.setattr(stac_module.StacClient, "open", staticmethod(open))
        return opened

    return connect


@pytest.fixture
def red() -> xr.Dataset:
    return xr.Dataset(
        {"red": (("y", "x"), np.zeros((2, 2))), "nir": (("y", "x"), np.ones((2, 2)))}
    )


# Raw scenes and monthly composites, for the frame and chip specs.
from tests.geodata.sampling.conftest import (  # noqa: E402, F401
    composites,
    grid,
    items,
    radar,
    sample,
    wide_grid,
)
