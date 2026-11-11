"""Formatting names outputs without selecting or computing raster data."""

import pytest

from odc.geo.geobox import GeoBox

from geosave.geodata.io.paths import format_path, format_resolution
from tests.geodata.conftest import build_raster


@pytest.mark.parametrize(
    "template",
    [
        "/{variable}.tif",
        "s3://bucket/{variable}.tif",
        "../{variable}.tif",
        "a/../../{variable}.tif",
        "C:/out/{variable}.tif",
        r"..\{variable}.tif",
    ],
)
def test_paths_cannot_escape_output_folder(template):
    with pytest.raises(ValueError, match="relative"):
        format_path(template, build_raster()[["red"]])


def test_rendered_labels_cannot_escape_output_folder():
    with pytest.raises(ValueError, match="relative"):
        format_path("{asset}.tif", build_raster(), asset="../red")


def test_format_requires_available_labels():
    with pytest.raises(ValueError, match="unavailable.*band"):
        format_path("{band}.tif", build_raster()[["red"]])


def test_format_reads_scalar_coords_variable_and_fields_but_not_the_crs():
    chip = build_raster(times=1).isel(time=0)[["red"]]
    path = format_path("{time:%Y%m%d}/{variable}_{asset}.tif", chip, asset="a1")
    assert path == "20250601/red_a1.tif"
    with pytest.raises(ValueError, match="unavailable.*spatial_ref"):
        format_path("{spatial_ref}.tif", chip)


def test_template_does_not_control_raster_partitioning(tmp_path):
    dataset = build_raster(times=2)
    with pytest.raises(ValueError, match="collide"):
        dataset.gs.to_cog(tmp_path, format_paths="single.tif")
    assert not list(tmp_path.iterdir())


def test_duplicate_time_labels_are_rejected_before_writing(tmp_path):
    dataset = build_raster(times=2)
    dataset = dataset.assign_coords(time=[dataset.time.values[0]] * 2)
    with pytest.raises(ValueError, match="collide"):
        dataset.gs.to_cog(tmp_path)
    assert not list(tmp_path.iterdir())


def test_format_resolution_follows_eopf():
    metres = GeoBox.from_bbox(
        (500000, 9000000, 500640, 9000640), crs="EPSG:32749", resolution=10
    )
    degrees = GeoBox.from_bbox(
        (100, 0, 100.01, 0.01), crs="EPSG:4326", resolution=0.0001
    )
    assert format_resolution(metres) == "r10m"
    assert format_resolution(degrees) == "r0.0001deg"
