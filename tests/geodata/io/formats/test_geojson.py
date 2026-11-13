"""GeoJSON features read and write as plain vectors."""

from __future__ import annotations

import json

import geopandas as gpd
from shapely.geometry import box
from upath import UPath


def test_geojson_writes_to_a_remote_url(bucket: str) -> None:
    plots = gpd.GeoDataFrame(
        [{"name": "a"}], geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )

    written = plots.gs.to_geojson(f"{bucket}/plots.geojson")

    assert written == UPath(f"{bucket}/plots.geojson")
    stored = json.loads(written.read_text())
    assert [feature["properties"]["name"] for feature in stored["features"]] == ["a"]
