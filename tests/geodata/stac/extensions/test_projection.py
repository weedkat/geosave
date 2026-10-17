"""A raster's grid as Projection fields, and back."""

from __future__ import annotations

import numpy as np
import pystac
import pytest
from odc.geo.geobox import GeoBox

from geosave.geodata import raster
from geosave.geodata.stac.extensions import projection


def test_a_grid_is_spelled_as_projection_fields(scene) -> None:
    stated = projection.from_geobox(scene.gs.geobox)

    assert stated == {
        "proj:code": "EPSG:32633",
        "proj:shape": [64, 64],
        "proj:transform": [10.0, 0.0, 300000.0, 0.0, -10.0, 5000640.0],
    }


def test_a_grid_without_an_epsg_code_states_its_wkt() -> None:
    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    grid = GeoBox.from_bbox((0, 0, 640, 640), crs, resolution=10)
    plain = raster({"height": (("y", "x"), np.zeros((64, 64), "float32"))}, grid)

    stated = projection.from_geobox(plain.gs.geobox)

    assert "proj:code" not in stated
    assert "Sinusoidal" in stated["proj:wkt2"]


def test_a_missing_grid_refuses() -> None:
    with pytest.raises(ValueError, match="no locatable grid"):
        projection.from_geobox(None)


def test_a_grid_round_trips_through_an_asset(scene) -> None:
    asset = pystac.Asset(
        "scene.tif", extra_fields=projection.from_geobox(scene.gs.geobox)
    )

    assert projection.to_geobox(asset) == scene.gs.geobox


def test_an_asset_stating_no_grid_has_none() -> None:
    assert projection.to_geobox(pystac.Asset("scene.tif")) is None
