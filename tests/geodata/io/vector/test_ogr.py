"""GeoJSON and GeoPackage read and write through the shared OGR writer."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, box
from upath import UPath

from geosave.geodata import open_vector


def test_geojson_writes_to_a_remote_url(bucket: str) -> None:
    plots = gpd.GeoDataFrame(
        [{"name": "a"}], geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )

    written = plots.gs.to_geojson(f"{bucket}/plots.geojson")

    assert written == UPath(f"{bucket}/plots.geojson")
    stored = json.loads(written.read_text())
    assert [feature["properties"]["name"] for feature in stored["features"]] == ["a"]


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
