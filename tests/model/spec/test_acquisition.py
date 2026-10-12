from importlib import import_module

import numpy as np
from pystac_client.exceptions import APIError
import pytest
import xarray as xr

from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.geodata.errors import CollectionNotFoundError
from geosave_engine.model.spec import ModelSpec, RasterRequirement, StacRecipe


stac_module = import_module("geosave_engine.model.spec.stac")


def coordinate_anchor():
    return GeoAnchor.from_coordinates(
        45,
        12,
        4,
        10,
        crs="EPSG:32633",
        timespan="2025-01",
    )


def recipe():
    return StacRecipe.model_validate(
        {"collection": "optical", "endpoints": ["https://primary.test/stac"]}
    )


class _Source:
    def __init__(self, loaded):
        self.loaded = loaded

    def load(self, anchor):
        return self.loaded.assign_attrs(config=self.config, query=self.query)


class _Client:
    def __init__(self, loaded):
        self.loaded = loaded

    def source(self, collection):
        return _Source(self.loaded)


def _serve(monkeypatch, served: dict[str, object]) -> list[str]:
    """Answer `StacClient.open` per endpoint: a raster, or an error to raise."""
    opened = []

    def open(endpoint):
        opened.append(endpoint)
        answer = served[endpoint]
        if isinstance(answer, Exception):
            raise answer
        return _Client(answer)

    monkeypatch.setattr(stac_module.StacClient, "open", staticmethod(open))
    return opened


def _red() -> xr.Dataset:
    return xr.Dataset(
        {"red": (("y", "x"), np.zeros((2, 2))), "nir": (("y", "x"), np.ones((2, 2)))}
    )


def test_a_recipe_loads_with_its_own_query_and_load_settings(monkeypatch):
    _serve(monkeypatch, {"https://primary.test/stac": _red()})
    recipe = StacRecipe.model_validate(
        {
            "collection": "optical",
            "endpoints": ["https://primary.test/stac"],
            "query": {"max_items": 3},
            "load": {"bands": ["red"]},
        }
    )

    loaded = recipe.load_raster(coordinate_anchor())

    assert loaded.attrs["config"].bands == ["red"]
    assert loaded.attrs["query"].collections == ["optical"]
    assert loaded.attrs["query"].max_items == 3


def test_an_unavailable_endpoint_falls_back_to_the_next(monkeypatch):
    opened = _serve(
        monkeypatch,
        {
            "https://primary.test/stac": APIError("unreachable"),
            "https://backup.test/stac": CollectionNotFoundError("absent"),
            "planetary_computer": _red(),
        },
    )
    recipe = StacRecipe.model_validate(
        {
            "collection": "optical",
            "endpoints": [
                "https://primary.test/stac",
                "https://backup.test/stac",
                "planetary_computer",
            ],
        }
    )

    assert "red" in recipe.load_raster(coordinate_anchor())
    assert opened == [
        "https://primary.test/stac",
        "https://backup.test/stac",
        "planetary_computer",
    ]


def test_no_serving_endpoint_names_every_failure(monkeypatch):
    _serve(monkeypatch, {"https://primary.test/stac": APIError("unreachable")})

    with pytest.raises(ConnectionError, match="primary.test.*unreachable"):
        recipe().load_raster(coordinate_anchor())


def test_load_rasters_selects_what_each_recipe_loads(monkeypatch):
    _serve(monkeypatch, {"https://primary.test/stac": _red()})
    model = ModelSpec(
        schema_version=2,
        rasters={"image": RasterRequirement(variables=("red",), stac=recipe())},
    )

    rasters = model.load_rasters(coordinate_anchor())

    assert list(rasters["image"].data_vars) == ["red"]


def test_load_rasters_refuses_rasters_without_a_recipe():
    model = ModelSpec(
        schema_version=2,
        rasters={
            "image": RasterRequirement(variables=("red",), stac=recipe()),
            "dem": RasterRequirement(variables=("height",)),
        },
    )

    with pytest.raises(ValueError, match=r"no `stac` block: \['dem'\]"):
        model.load_rasters(coordinate_anchor())
