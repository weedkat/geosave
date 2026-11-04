"""A raster's grid as Projection fields, and back."""

from __future__ import annotations

import numpy as np
import pystac
import pytest
from odc.geo.geobox import GeoBox

from geosave.geodata.stac.extensions import projection
from tests.conftest import build_dataset


def test_a_grid_without_an_epsg_code_states_its_wkt() -> None:
    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    grid = GeoBox.from_bbox((0, 0, 640, 640), crs, resolution=10)
    plain = build_dataset({"height": (("y", "x"), np.zeros((64, 64), "float32"))}, grid)

    stated = projection.build_fields(plain.gs.geobox)

    assert "proj:code" not in stated
    assert "Sinusoidal" in stated["proj:wkt2"]


def test_a_missing_grid_refuses() -> None:
    with pytest.raises(ValueError, match="no locatable grid"):
        projection.build_fields(None)


def test_a_grid_round_trips_through_an_asset(scene) -> None:
    asset = pystac.Asset(
        "scene.tif", extra_fields=projection.build_fields(scene.gs.geobox)
    )

    assert projection.read_geobox(asset) == scene.gs.geobox
