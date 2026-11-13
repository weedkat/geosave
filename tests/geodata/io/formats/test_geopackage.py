"""GeoPackage layers read and write through the shared storage writer."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point
from upath import UPath

from geosave.geodata import (
    open_vector,
)


def _plots() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"class": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
    )


def test_a_layer_is_named_after_the_file_by_default(tmp_path: Path) -> None:
    path = _plots().gs.to_geopackage(tmp_path / "survey.gpkg")

    assert gpd.list_layers(path)["name"].tolist() == ["survey"]
    assert open_vector(path)["class"].tolist() == [1, 2]


def test_a_geopackage_writes_to_a_remote_url(bucket: str) -> None:
    written = _plots().gs.to_geopackage(f"{bucket}/survey.gpkg")

    assert written == UPath(f"{bucket}/survey.gpkg")
    assert written.stat().st_size > 0
