"""GeoJSON features read and write as plain vectors."""

from __future__ import annotations

import json

import geopandas as gpd
import pytest
from shapely.geometry import box, mapping
from upath import UPath

from geosave.geodata import io
from tests.geodata.conftest import _size


def test_direct_geojson_writer_refuses_to_discard_catalog_metadata(tmp_path):
    rows = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=4326)
    rows.attrs["stac-geoparquet"] = {"collections": {}}
    path = tmp_path / "catalog.geojson"

    with pytest.raises(ValueError, match="to_geoparquet"):
        io.formats.geojson.write_geojson(rows, path)

    assert not path.exists()


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
    source = io.formats.geojson.open_geojson(path)
    assert isinstance(source.loc[0, "assets"], dict)
    assert source.loc[0, "assets"]["image"]["roles"] == ["data"]


def test_geojson_writes_to_a_remote_url(bucket: str) -> None:
    plots = gpd.GeoDataFrame(
        [{"name": "a"}], geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )

    written = plots.gs.to_geojson(f"{bucket}/plots.geojson")

    assert written == UPath(f"{bucket}/plots.geojson")
    stored = json.loads(written.read_text())
    assert [feature["properties"]["name"] for feature in stored["features"]] == ["a"]


def test_a_failed_geojson_write_keeps_the_existing_file(tmp_path, monkeypatch) -> None:
    import geopandas as gpd

    plots = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    path = plots.gs.to_geojson(tmp_path / "plots.geojson")
    before = path.read_text()

    def fail(*args, **kwargs):
        raise RuntimeError("driver failed")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fail)
    with pytest.raises(RuntimeError, match="driver failed"):
        plots.gs.to_geojson(path, overwrite=True)

    assert path.read_text() == before


def test_a_file_url_reads_as_its_local_path(tmp_path) -> None:
    path = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").gs.to_geojson(
        tmp_path / "plots.geojson"
    )

    assert len(io.formats.geojson.open_geojson(UPath(path.as_uri()))) == 1


@pytest.mark.integration
@pytest.mark.parametrize("suffix", [".geojson"])
def test_a_plain_vector_uploads(remote, suffix) -> None:
    prefix, options = remote
    plots = gpd.GeoDataFrame(
        {"name": ["near", "far"]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )
    write = {
        ".parquet": plots.gs.to_geoparquet,
        ".geojson": plots.gs.to_geojson,
        ".gpkg": plots.gs.to_geopackage,
    }[suffix]
    written = write(f"{prefix}/plots{suffix}", storage_options=options)
    assert written == f"{prefix}/plots{suffix}"
    assert _size(written, options) > 0
