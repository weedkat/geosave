"""GeoPackage layers read and write through the shared storage writer."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, box
from upath import UPath

from geosave.geodata import (
    open_vector,
)
from tests.geodata.conftest import _size


def test_direct_geopackage_writer_refuses_to_discard_catalog_metadata(tmp_path):
    from geosave.geodata.io.formats.geopackage import write_geopackage

    rows = _plots()
    rows.attrs["stac-geoparquet"] = {"collections": {}}
    path = tmp_path / "catalog.gpkg"

    with pytest.raises(ValueError, match="to_geoparquet"):
        write_geopackage(rows, path)

    assert not path.exists()


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


def test_an_option_given_twice_refuses(tmp_path: Path) -> None:
    from geosave.geodata.io.formats import geopackage

    path = _plots().gs.to_geopackage(tmp_path / "survey.gpkg")

    with pytest.raises(TypeError, match="rows"):
        geopackage.open_geopackage(path, rows=1, engine_options={"rows": 2})


def test_a_file_url_reads_as_its_local_path(tmp_path: Path) -> None:
    path = _plots().gs.to_geopackage(tmp_path / "survey.gpkg")

    assert len(open_vector(UPath(path.as_uri()))) == 2


@pytest.mark.integration
@pytest.mark.parametrize("suffix", [".gpkg"])
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
