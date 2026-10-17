"""GeoJSON features read and write as plain vectors."""

import json

import pytest
from shapely.geometry import box, mapping
from upath import UPath

from geosave.geodata import GeoVector, io


def test_plain_geojson_nested_assets_remain_mappings(tmp_path):
    path = tmp_path / "field.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(box(0, 0, 1, 1)),
                        "properties": {
                            "id": "field",
                            "datetime": "2025-06-01",
                            "assets": {
                                "image": {"href": "image.tif", "roles": ["data"]}
                            },
                        },
                    }
                ],
            }
        )
    )
    source = io.geojson.open_vector(path)
    assert isinstance(source.loc[0, "assets"], dict)
    assert source.loc[0, "assets"]["image"]["roles"] == ["data"]


def test_geojson_writes_to_a_remote_url(bucket: str) -> None:
    plots = GeoVector.from_geometry(box(0, 0, 1, 1), properties={"name": "a"})

    written = plots.gs.to_geojson(f"{bucket}/plots.geojson")

    assert written == UPath(f"{bucket}/plots.geojson")
    stored = json.loads(written.read_text())
    assert [feature["properties"]["name"] for feature in stored["features"]] == ["a"]


def test_a_failed_geojson_write_keeps_the_existing_file(tmp_path, monkeypatch) -> None:
    import geopandas as gpd

    plots = GeoVector.from_geometry(box(0, 0, 1, 1))
    path = plots.gs.to_geojson(tmp_path / "plots.geojson")
    before = path.read_text()

    def fail(*args, **kwargs):
        raise RuntimeError("driver failed")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fail)
    with pytest.raises(RuntimeError, match="driver failed"):
        plots.gs.to_geojson(path, overwrite=True)

    assert path.read_text() == before


def test_an_option_given_twice_refuses(tmp_path) -> None:
    path = GeoVector.from_geometry(box(0, 0, 1, 1)).gs.to_geojson(
        tmp_path / "plots.geojson"
    )

    with pytest.raises(TypeError, match="rows"):
        io.geojson.open_vector(path, rows=1, driver_options={"rows": 2})


def test_a_file_url_reads_as_its_local_path(tmp_path) -> None:
    path = GeoVector.from_geometry(box(0, 0, 1, 1)).gs.to_geojson(
        tmp_path / "plots.geojson"
    )

    assert len(io.geojson.open_vector(UPath(path.as_uri()))) == 1
