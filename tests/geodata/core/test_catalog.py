"""The `gs` accessor of an item table."""

from __future__ import annotations

import pytest

from geosave_engine.geodata import open_vector, stac, stack
from geosave_engine.geodata.core.catalog import GeoCatalog
from geosave_engine.geodata.io.vector import stac_geoparquet
from tests.geodata.conftest import stack_templates


@pytest.fixture
def rows(scene, label, tmp_path):
    sample = stack({"optical": scene, "label": label})
    saved = sample.gs.to_cog(stack_templates(sample, tmp_path / "s0"))
    return stac.create_items(saved, id="s0")


def test_gs_is_a_catalog_on_an_item_table_and_a_vector_otherwise(rows) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    from geosave_engine.geodata import GeoVector

    plots = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:4326")

    assert type(rows.gs) is GeoCatalog
    assert type(plots.gs) is GeoVector
    assert type(rows[["id", "geometry"]].gs) is GeoVector


def test_a_plain_frame_still_has_no_gs_accessor() -> None:
    import pandas as pd

    with pytest.raises(AttributeError):
        pd.DataFrame({"a": [1]}).gs


def test_a_catalog_writes_one_file_holding_every_collection(rows, tmp_path) -> None:
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    assert path.is_file()
    assert len(open_vector(path)) == len(rows)
    assert sorted(stac.get_collections(path)) == sorted(rows["collection"].unique())


def test_a_catalog_writes_stac_geoparquet_that_validates(rows, tmp_path) -> None:
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = stac_geoparquet.to_items(open_vector(path))
    assert [item.id for item in restored] == rows["id"].tolist()
    for item in restored:
        item.validate()


def test_collections_rows_name_are_described_when_not_given(rows, tmp_path) -> None:
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    described = stac.get_collections(path)

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
