from pathlib import Path
from uuid import uuid4

import fsspec
import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Point, box

from geosave.geodata import GeoVector, io, open_vector


def test_geoparquet_catalog_supports_bbox_and_column_filtering(
    tmp_path: Path,
) -> None:
    vector = gpd.GeoDataFrame(
        {"name": ["near", "far"]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )
    path = vector.gs.to_geoparquet(
        tmp_path / "catalog.parquet", write_covering_bbox=True
    )

    selected = open_vector(path, bbox=(-1, -1, 2, 2), columns=["name", "geometry"])

    assert list(selected.name) == ["near"]


def test_geoparquet_does_not_write_covering_bbox_by_default(tmp_path: Path) -> None:
    path = GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(
        tmp_path / "catalog.parquet"
    )

    assert "bbox" not in pq.read_schema(path).names


def test_geoparquet_write_replaces_stale_staging_file(tmp_path: Path) -> None:
    target = tmp_path / "catalog.parquet"
    staged = tmp_path / ".catalog.staging.parquet"
    staged.write_text("interrupted")

    GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(target)

    assert target.is_file()
    assert not staged.exists()


def test_empty_geoparquet_catalog_round_trips(tmp_path: Path) -> None:
    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))
    path = empty.gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = open_vector(path)

    assert len(restored) == 0
    assert restored.crs.to_epsg() == 4326


def memory_catalog() -> str:
    return f"memory://geosave-tests/{uuid4()}/catalog.parquet"


def test_a_row_without_assets_stays_without() -> None:
    destination = memory_catalog()
    vector = gpd.GeoDataFrame(
        {"id": ["a"], "assets": [None]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )

    vector.gs.to_geoparquet(destination)

    assert open_vector(destination).assets.isna().all()


def test_remote_write_refuses_existing_catalog() -> None:
    destination = memory_catalog()
    vector = GeoVector.from_geometry(Point(0, 0))
    vector.gs.to_geoparquet(destination)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        vector.gs.to_geoparquet(destination)


def test_file_url_returns_a_usable_local_path(tmp_path: Path) -> None:
    destination = tmp_path / "catalog.parquet"

    written = GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(
        destination.as_uri()
    )

    assert written == destination
    assert len(open_vector(written)) == 1


def test_invalid_remote_suffix_creates_nothing() -> None:
    destination = memory_catalog().replace(".parquet", ".json")
    filesystem, target = fsspec.core.url_to_fs(destination)

    with pytest.raises(ValueError, match="must end"):
        GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(destination)

    assert not filesystem.exists(target)


def test_nullable_assets_on_an_ordinary_vector_survive_geoparquet(tmp_path):
    source = GeoVector.from_geometry(box(0, 0, 1, 1), properties={"assets": None})
    path = source.gs.to_geoparquet(tmp_path / "plain.parquet")
    restored = io.open_vector(path)
    assert sorted(restored.columns) == ["assets", "geometry"]
    assert len(restored) == 1


@pytest.mark.parametrize("existing", [False, True], ids=["new", "overwrite"])
def test_failed_remote_write_keeps_the_destination_unchanged(monkeypatch, existing):
    destination = memory_catalog()
    filesystem, target = fsspec.core.url_to_fs(destination)
    if existing:
        filesystem.pipe(target, b"existing")

    def fail(_self, path: Path, **options: object) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("encode failed")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail)
    with pytest.raises(RuntimeError, match="encode failed"):
        GeoVector.from_geometry(Point(0, 0)).gs.to_geoparquet(
            destination, overwrite=existing
        )

    if existing:
        assert filesystem.cat(target) == b"existing"
    else:
        assert not filesystem.exists(target)
