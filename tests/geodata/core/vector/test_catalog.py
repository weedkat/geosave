"""The `gs` accessor of an item table."""

from __future__ import annotations

from geosave.geodata.stac.translate import table as stac_table

from datetime import UTC, datetime

import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box

from geosave.geodata import open_vector, stac
from geosave.geodata.core.anchor import GeoAnchor
from geosave.geodata.core import GeoCatalog
from geosave.geodata.io.vector import geoparquet
from tests.geodata import conftest as shared_fixtures
from tests.geodata.conftest import build_raster

# `rows` is built on the shared grid, not the window grid this folder's conftest defines.
grid = shared_fixtures.grid


def test_gs_is_a_catalog_on_an_item_table_and_a_vector_otherwise(rows) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    from geosave.geodata import GeoVector

    plots = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:4326")

    assert type(rows.gs) is GeoCatalog
    assert type(plots.gs) is GeoVector
    assert type(rows[["id", "geometry"]].gs) is GeoVector


def test_a_plain_frame_still_has_no_gs_accessor() -> None:

    with pytest.raises(AttributeError):
        pd.DataFrame({"a": [1]}).gs


@pytest.mark.parametrize("reader", [open_vector, geoparquet.open_vector])
@pytest.mark.parametrize("remote", [False, True])
def test_a_catalog_writes_one_file_holding_every_collection(
    rows, tmp_path, bucket, reader, remote
) -> None:
    destination = (
        f"{bucket}/catalog.parquet" if remote else tmp_path / "catalog.parquet"
    )
    path = rows.gs.to_geoparquet(destination)

    assert path.is_file()
    restored = reader(path)
    assert isinstance(restored.gs, GeoCatalog)
    assert len(restored) == len(rows)
    assert sorted(restored.gs.collections) == sorted(rows["collection"].unique())
    for item in restored.gs.to_items():
        for asset in item.assets.values():
            assert asset.href == asset.get_absolute_href()
    assert sorted(stac.read_collections(path)) == sorted(rows["collection"].unique())


def test_a_catalog_writes_stac_geoparquet_that_validates(rows, tmp_path) -> None:
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = stac_table.to_items(open_vector(path))
    assert [item.id for item in restored] == rows["id"].tolist()
    for item in restored:
        item.validate()


def test_collections_rows_name_are_described_when_not_given(rows, tmp_path) -> None:
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    described = stac.read_collections(path)

    assert all(each.extent.spatial.bboxes for each in described.values())


def test_a_catalog_refuses_write_options(rows, tmp_path) -> None:
    with pytest.raises(ValueError, match="compression"):
        rows.gs.to_geoparquet(tmp_path / "catalog.parquet", compression="zstd")


def test_a_catalog_refuses_formats_that_cannot_hold_assets(rows, tmp_path) -> None:
    with pytest.raises(ValueError, match="GeoParquet"):
        rows.gs.to_geopackage(tmp_path / "catalog.gpkg")
    with pytest.raises(ValueError, match="GeoParquet"):
        rows.gs.to_geojson(tmp_path / "catalog.geojson")


def test_open_vector_does_not_read_a_folder(rows, tmp_path) -> None:
    rows.gs.to_geoparquet(tmp_path / "catalog" / "items.parquet")

    with pytest.raises(ValueError, match="no supported vector format"):
        open_vector(tmp_path / "catalog")


def test_a_catalog_converts_its_rows_to_pystac_items_in_order(rows) -> None:
    import pystac

    items = rows.gs.to_items()

    assert all(isinstance(item, pystac.Item) for item in items)
    assert [item.id for item in items] == rows["id"].tolist()
    assert [item.id for item in rows.iloc[[1]].gs.to_items()] == [rows["id"].iloc[1]]


def test_a_catalog_writes_static_stac_json(rows, tmp_path) -> None:
    path = rows.gs.to_stac_json(tmp_path / "stac")

    assert path == tmp_path / "stac" / "catalog.json"
    assert path.is_file()


def _ground(geometry, timespan: str | None = None) -> GeoAnchor:
    """Anchor fitting one longitude/latitude geometry exactly."""
    return GeoAnchor.from_geometry(
        geometry, shape=4, crs="EPSG:4326", anchor="floating", timespan=timespan
    )


def _items(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Give a frame the columns every STAC Item row carries, so it is a catalog."""
    stated = frame.copy()
    stated["type"] = "Feature"
    stated["stac_version"] = "1.1.0"
    stated["assets"] = [{} for _ in range(len(stated))]
    assert isinstance(stated.gs, GeoCatalog)
    return stated


def test_query_reads_time_columns_that_carry_no_timezone() -> None:
    table = _dated()
    for column in ("datetime", "start_datetime", "end_datetime"):
        table[column] = table[column].dt.tz_localize(None)

    assert list(table.gs.query(_ground(box(-1, -1, 2, 2), "2025-01")).id) == ["jan"]


def _dated() -> gpd.GeoDataFrame:
    """Three rows on one ground: two spans and one lone instant."""
    table = gpd.GeoDataFrame(
        {
            "id": ["jan", "jun", "instant"],
            "datetime": [None, None, datetime(2025, 6, 12, tzinfo=UTC)],
            "start_datetime": [
                datetime(2025, 1, 10, tzinfo=UTC),
                datetime(2025, 6, 10, tzinfo=UTC),
                None,
            ],
            "end_datetime": [
                datetime(2025, 1, 15, tzinfo=UTC),
                datetime(2025, 6, 15, tzinfo=UTC),
                None,
            ],
        },
        geometry=[box(0, 0, 1, 1)] * 3,
        crs="EPSG:4326",
    )
    table = _items(table)
    for column in ("datetime", "start_datetime", "end_datetime"):
        table[column] = pd.to_datetime(table[column], utc=True)
    return table


def test_query_compares_time_where_the_anchor_and_the_table_state_one() -> None:
    table = _dated()
    ground = box(-1, -1, 2, 2)

    assert list(table.gs.query(_ground(ground)).id) == ["jan", "jun", "instant"]
    assert list(table.gs.query(_ground(ground, "2025-01")).id) == ["jan"]
    assert list(table.gs.query(_ground(ground, "2025-06-12")).id) == ["jun", "instant"]
    assert list(table.gs.query(_ground(ground, "2025-03")).id) == []


def test_query_ignores_time_on_a_table_that_states_none() -> None:
    undated = _items(
        gpd.GeoDataFrame({"id": ["place"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    )

    assert list(undated.gs.query(_ground(box(-1, -1, 2, 2), "2025-01")).id) == ["place"]


def test_query_filters_time_on_a_table_with_only_datetime() -> None:
    raster = build_raster(times=2)
    footprint = raster.gs.geobox.extent.to_crs("EPSG:4326").geom
    frame = gpd.GeoDataFrame(
        {
            "id": ["first", "second"],
            "datetime": pd.to_datetime(raster.time.values, utc=True),
        },
        geometry=[footprint, footprint],
        crs="EPSG:4326",
    )

    assert _items(frame).gs.query(raster.isel(time=[0]))["id"].tolist() == ["first"]


def test_query_reads_a_span_where_a_row_has_no_datetime() -> None:
    raster = build_raster(times=2)
    footprint = raster.gs.geobox.extent.to_crs("EPSG:4326").geom
    frame = gpd.GeoDataFrame(
        {
            "id": ["store", "later"],
            "datetime": pd.to_datetime([None, "2025-07-01"], utc=True),
            "start_datetime": pd.to_datetime(["2025-06-01", None], utc=True),
            "end_datetime": pd.to_datetime(["2025-06-02", None], utc=True),
        },
        geometry=[footprint, footprint],
        crs="EPSG:4326",
    )

    assert _items(frame).gs.query(raster)["id"].tolist() == ["store"]


def test_undated_records_remain_spatially_selectable_in_a_time_query():
    source = _items(
        gpd.GeoDataFrame(
            {"id": ["undated"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
        )
    )
    source["datetime"] = pd.NaT
    selected = source.gs.query(_ground(box(-1, -1, 2, 2), "2025-01"))
    assert len(selected) == 1
