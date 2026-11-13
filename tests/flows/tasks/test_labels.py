from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from geosave_flows.tasks.labels import find_labels, read_labels


def test_find_labels_preserves_tree_and_removes_only_the_final_suffix(
    tmp_path: Path,
) -> None:
    first = tmp_path / "north" / "a.v1.tif"
    second = tmp_path / "south" / "b.tif"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.touch()

    assert find_labels(tmp_path, "**/*.tif") == {"north/a.v1": first, "south/b": second}


def test_directory_ingestion_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="STAC publication"):
        read_labels(tmp_path / "missing")


def test_a_label_table_keeps_its_caller_columns(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    labels = gpd.GeoDataFrame(
        [
            {
                "id": "a",
                "assets": {"label": {"href": str(root / "a.tif")}},
                "quality": 0.97,
                "surveyor": None,
            }
        ],
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )

    read = read_labels(labels.gs.to_geoparquet(tmp_path / "labels.parquet"))

    assert read.loc[0, "quality"] == 0.97
    assert read.loc[0, "surveyor"] is None


def test_a_label_table_names_its_label_asset(tmp_path: Path) -> None:
    plain = gpd.GeoDataFrame([{"id": "a"}], geometry=[Point(0, 0)], crs="EPSG:4326")
    other = gpd.GeoDataFrame(
        [{"id": "a", "assets": {"image": {"href": "a.tif"}}}],
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match=r"no \['assets'\] column"):
        read_labels(plain.gs.to_geoparquet(tmp_path / "plain.parquet"))
    with pytest.raises(ValueError, match=r"no 'label' asset: \['a'\]"):
        read_labels(other.gs.to_geoparquet(tmp_path / "other.parquet"))
