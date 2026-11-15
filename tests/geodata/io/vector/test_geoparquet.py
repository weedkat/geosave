from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import pytest
from shapely.geometry import box

from geosave.geodata import (
    open_vector,
)
from tests.geodata.conftest import create_test_collection


def make_catalog():
    from geosave.geodata.stac.catalog import create_catalog
    from tests.geodata.stac.conftest import entry

    collection = create_test_collection(
        "optical", description="Reflectance", license="CC-BY-4.0"
    )
    return create_catalog([entry()], collections=[collection])


@pytest.mark.parametrize("options", [{"compression": "zstd"}])
def test_write_selects_stac_encoding_and_preserves_collection_and_codec(
    tmp_path, options
):
    import pyarrow.parquet as pq

    rows = make_catalog()
    saved = rows.gs.to_geoparquet(tmp_path / "items.parquet", **options)
    metadata = pq.read_metadata(saved)
    assert b"stac-geoparquet" in metadata.metadata
    assert metadata.row_group(0).column(0).compression == "ZSTD"
    restored = open_vector(saved)
    assert restored.gs.collections["optical"].license == "CC-BY-4.0"
    assert restored.iloc[0]["id"] == rows.iloc[0]["id"]


@pytest.mark.parametrize("column", ["assets"])
def test_damaged_catalog_cannot_fall_back_to_feature_encoding(tmp_path, column):
    rows = make_catalog().drop(columns=column)
    with pytest.raises(ValueError, match=column):
        rows.gs.to_geoparquet(tmp_path / "damaged.parquet")
    assert not list(tmp_path.iterdir())


def test_item_rows_with_lost_metadata_are_not_written_as_features(tmp_path):
    from geosave.geodata.io.vector import geoparquet

    rows = make_catalog()
    rows.attrs.clear()
    with pytest.raises(ValueError, match="metadata"):
        geoparquet.write_geoparquet(rows, tmp_path / "unmarked.parquet")
    assert not list(tmp_path.iterdir())


def test_selected_catalog_columns_retain_metadata_and_cannot_be_rewritten_as_features(
    tmp_path,
):
    from geosave.geodata.io.vector import stac_geoparquet

    first = stac_geoparquet.write_stac_geoparquet(
        make_catalog(), tmp_path / "original.parquet"
    )
    selected = open_vector(first, columns=["id", "geometry"])
    assert selected.gs.collections["optical"].description == "Reflectance"
    with pytest.raises(ValueError, match="assets"):
        selected.gs.to_geoparquet(tmp_path / "partial.parquet")
    assert not (tmp_path / "partial.parquet").exists()


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


def memory_catalog() -> str:
    return f"memory://geosave-tests/{uuid4()}/catalog.parquet"


@pytest.mark.integration
def test_geoparquet_reads_back_with_a_bbox(remote) -> None:
    prefix, options = remote
    plots = gpd.GeoDataFrame(
        {"name": ["near", "far"]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )
    written = plots.gs.to_geoparquet(
        f"{prefix}/plots.parquet", write_covering_bbox=True, storage_options=options
    )

    selected = open_vector(written, bbox=(-1, -1, 2, 2), storage_options=options)

    assert selected["name"].tolist() == ["near"]
