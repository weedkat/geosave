"""Every vector format reads back the GeoDataFrame it can hold."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from geosave.geodata import io


def build_plots() -> gpd.GeoDataFrame:
    """Build two projected plots with a count, a name and frame attrs."""
    plots = gpd.GeoDataFrame(
        {"count": [1, 2], "name": ["a", None]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs="EPSG:32749",
    )
    plots.attrs = {"title": "Plots", "classes": [1, 2]}
    return plots


def test_a_geopackage_reads_back_the_frame_written(tmp_path) -> None:
    written = build_plots()

    restored = io.open_vector(io.write_geopackage(written, tmp_path / "plots.gpkg"))

    pd.testing.assert_frame_equal(restored, written, check_dtype=False)
    assert restored.crs == written.crs
    assert restored.attrs == written.attrs


def test_caller_layer_metadata_sits_beside_the_frame_attrs(tmp_path) -> None:
    import pyogrio

    path = io.write_geopackage(
        build_plots(), tmp_path / "plots.gpkg", metadata={"source": "survey"}
    )

    assert pyogrio.read_info(path)["layer_metadata"]["source"] == "survey"
    assert io.open_vector(path).attrs == build_plots().attrs


def test_a_geoparquet_reads_back_the_rows_written(tmp_path) -> None:
    written = build_plots()

    restored = io.open_vector(io.write_geoparquet(written, tmp_path / "plots.parquet"))

    pd.testing.assert_frame_equal(restored, written)
    assert restored.crs == written.crs


def test_geojson_reads_back_the_rows_in_wgs84(tmp_path) -> None:
    written = build_plots()

    restored = io.open_vector(io.write_geojson(written, tmp_path / "plots.geojson"))

    # GeoJSON is WGS84 by specification and has no slot for frame attrs.
    assert restored.crs == "EPSG:4326"
    assert restored["name"].tolist() == ["a", None]
    assert (
        restored.to_crs(written.crs)
        .geometry.geom_equals_exact(written.geometry, tolerance=1e-6)
        .all()
    )


@pytest.mark.parametrize("suffix", [".gpkg", ".geojson"])
def test_engine_options_cannot_repeat_a_direct_option(tmp_path, suffix) -> None:
    write = {".gpkg": io.write_geopackage, ".geojson": io.write_geojson}[suffix]

    with pytest.raises(TypeError, match="index"):
        write(
            build_plots(),
            tmp_path / f"plots{suffix}",
            index=False,
            engine_options={"index": True},
        )
