from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import fsspec
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from geosave.geodata import (
    open_vector,
)
from tests.geodata.conftest import create_test_collection


def make_catalog():
    from geosave.geodata.stac.catalog import from_items
    from tests.geodata.stac.conftest import entry

    collection = create_test_collection(
        "optical", description="Reflectance", license="CC-BY-4.0"
    )
    return from_items([entry()], collections=[collection])


@pytest.mark.parametrize(
    "options", [{"compression": "zstd"}, {"parquet_options": {"compression": "zstd"}}]
)
def test_write_selects_stac_encoding_and_preserves_collection_and_codec(
    tmp_path, options
):
    import pyarrow.parquet as pq

    from geosave.geodata.io.formats import geoparquet

    rows = make_catalog()
    saved = geoparquet.write_geoparquet(rows, tmp_path / "items.parquet", **options)
    metadata = pq.read_metadata(saved)
    assert b"stac-geoparquet" in metadata.metadata
    assert metadata.row_group(0).column(0).compression == "ZSTD"
    restored = open_vector(saved)
    assert restored.gs.collections["optical"].license == "CC-BY-4.0"
    assert restored.iloc[0].gs.to_item().id == rows.iloc[0].id


@pytest.mark.parametrize("column", ["assets", "links", "stac_extensions"])
def test_damaged_catalog_cannot_fall_back_to_feature_encoding(tmp_path, column):
    from geosave.geodata.io.formats import geoparquet

    rows = make_catalog().drop(columns=column)
    with pytest.raises(ValueError, match=column):
        geoparquet.write_geoparquet(rows, tmp_path / "damaged.parquet")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("metadata", [None, [], {"collections": []}])
def test_malformed_catalog_metadata_is_rejected(tmp_path, metadata):
    from geosave.geodata.io.formats import geoparquet

    rows = make_catalog()
    rows.attrs["stac-geoparquet"] = metadata
    with pytest.raises(ValueError, match="metadata|collections"):
        geoparquet.write_geoparquet(rows, tmp_path / "damaged.parquet")
    assert not list(tmp_path.iterdir())


def test_item_rows_with_lost_metadata_are_not_written_as_features(tmp_path):
    from geosave.geodata.io.formats import geoparquet

    rows = make_catalog()
    rows.attrs.clear()
    with pytest.raises(ValueError, match="metadata"):
        geoparquet.write_geoparquet(rows, tmp_path / "unmarked.parquet")
    assert not list(tmp_path.iterdir())


def test_selected_catalog_columns_retain_metadata_and_cannot_be_rewritten_as_features(
    tmp_path,
):
    from geosave.geodata.io.formats import stac_geoparquet

    first = stac_geoparquet.write_stac_geoparquet(
        make_catalog(), tmp_path / "original.parquet"
    )
    selected = open_vector(first, columns=["id", "geometry"])
    assert selected.gs.collections["optical"].description == "Reflectance"
    with pytest.raises(ValueError, match="assets"):
        selected.gs.to_geoparquet(tmp_path / "partial.parquet")
    assert not (tmp_path / "partial.parquet").exists()


def test_empty_metadata_mapping_still_selects_catalog_encoding(tmp_path):
    import pyarrow.parquet as pq

    from geosave.geodata.io.formats import geoparquet

    rows = make_catalog()
    rows.attrs["stac-geoparquet"] = {}
    saved = geoparquet.write_geoparquet(rows, tmp_path / "items.parquet")
    assert b"stac-geoparquet" in pq.read_metadata(saved).metadata
    assert open_vector(saved).iloc[0].gs.to_item().id == rows.iloc[0].id


def test_parquet_options_cannot_override_managed_catalog_metadata(tmp_path):
    from geosave.geodata.io.formats import geoparquet

    with pytest.raises(TypeError, match="collections"):
        geoparquet.write_geoparquet(
            make_catalog(),
            tmp_path / "wrong.parquet",
            parquet_options={"collections": {}},
        )
    assert not list(tmp_path.iterdir())


def test_catalog_cannot_select_an_unsupported_geometry_encoding(tmp_path):
    from geosave.geodata.io.formats import geoparquet

    with pytest.raises(ValueError, match="geometry_encoding"):
        geoparquet.write_geoparquet(
            make_catalog(), tmp_path / "wrong.parquet", geometry_encoding="geoarrow"
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("catalog", [True, False])
def test_options_cannot_remove_required_geoparquet_metadata(tmp_path, catalog):
    rows = (
        make_catalog()
        if catalog
        else gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)
    )
    with pytest.raises(ValueError, match="store_schema"):
        rows.gs.to_geoparquet(
            tmp_path / "broken.parquet", parquet_options={"store_schema": False}
        )
    assert not list(tmp_path.iterdir())


def test_catalog_row_group_size_reaches_actual_encoding(tmp_path):
    import pyarrow.parquet as pq

    from geosave.geodata.stac.catalog import from_items

    source = make_catalog()
    items = []
    for index in range(3):
        item = source.iloc[0].gs.to_item()
        item.id = f"scene-{index}"
        items.append(item)
    rows = from_items(items, collections=source.gs.collections.values())
    path = rows.gs.to_geoparquet(
        tmp_path / "items.parquet", parquet_options={"row_group_size": 1}
    )
    metadata = pq.read_metadata(path)
    assert metadata.num_row_groups == 3
    assert [metadata.row_group(index).num_rows for index in range(3)] == [1, 1, 1]
    assert open_vector(path).gs.collections["optical"].license == "CC-BY-4.0"


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


def test_geoparquet_write_replaces_stale_staging_file(tmp_path: Path) -> None:
    target = tmp_path / "catalog.parquet"
    staged = tmp_path / ".catalog.staging.parquet"
    staged.write_text("interrupted")

    gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326").gs.to_geoparquet(target)

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


def test_file_url_returns_a_usable_local_path(tmp_path: Path) -> None:
    destination = tmp_path / "catalog.parquet"

    written = gpd.GeoDataFrame(
        geometry=[Point(0, 0)], crs="EPSG:4326"
    ).gs.to_geoparquet(destination.as_uri())

    assert written == destination
    assert len(open_vector(written)) == 1


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
        gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326").gs.to_geoparquet(
            destination, overwrite=existing
        )

    if existing:
        assert filesystem.cat(target) == b"existing"
    else:
        assert not filesystem.exists(target)


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
