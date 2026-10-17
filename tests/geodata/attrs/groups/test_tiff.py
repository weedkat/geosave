from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata.attrs.groups.tiff import GeoTIFFTags
from tests.geodata.conftest import build_raster


def test_geosave_datetime_needs_a_valid_instant() -> None:
    with pytest.raises(ValueError, match="GEOSAVE_DATETIME needs an ISO 8601 instant"):
        GeoTIFFTags(GEOSAVE_DATETIME="not-a-datetime")


def test_geotiff_tags_take_description_and_time_from_the_raster() -> None:
    instant = np.datetime64("2024-03-05T06:07:08")
    source = (
        build_raster(times=0)
        .assign_coords(time=instant)
        .assign_attrs(
            summary="Surface reflectance scene",
            TIFFTAG_IMAGEDESCRIPTION="stale",
            TIFFTAG_DATETIME="1999:01:01 00:00:00",
            TIFFTAG_ARTIST="GeoSave",
        )
    )

    tags = GeoTIFFTags.from_xarray(source)

    assert tags.TIFFTAG_IMAGEDESCRIPTION == "Surface reflectance scene"
    assert tags.TIFFTAG_DATETIME == instant.astype("datetime64[s]").astype(object)
    assert tags.TIFFTAG_ARTIST == "GeoSave"


def test_geotiff_tags_keep_a_description_when_acdd_has_no_summary() -> None:
    source = build_raster(times=0).assign_attrs(
        title="A scene", TIFFTAG_IMAGEDESCRIPTION="Hand-authored description"
    )

    tags = GeoTIFFTags.from_xarray(source)

    assert tags.TIFFTAG_IMAGEDESCRIPTION == "Hand-authored description"


@pytest.mark.parametrize(
    ("crs", "expected"),
    [
        ("EPSG:32749", 10.0),
        ("EPSG:2277", 32.808333333333),
    ],
    ids=["metres", "us-survey-feet"],
)
def test_geotiff_tags_calculate_pixels_per_centimetre(crs: str, expected: float):
    tags = GeoTIFFTags.from_xarray(build_raster(crs=crs), map_scale=10_000)

    assert tags.TIFFTAG_XRESOLUTION == pytest.approx(expected)
    assert tags.TIFFTAG_YRESOLUTION == pytest.approx(expected)
    assert tags.TIFFTAG_RESOLUTIONUNIT == 3


@pytest.mark.parametrize("map_scale", [0, -1, float("nan"), float("inf")])
def test_geotiff_tags_refuse_an_invalid_map_scale(map_scale: float) -> None:
    with pytest.raises(ValueError, match="map_scale"):
        GeoTIFFTags.from_xarray(build_raster(), map_scale=map_scale)


def test_geotiff_tags_refuse_physical_resolution_for_a_geographic_grid() -> None:
    with pytest.raises(ValueError, match="projected CRS"):
        GeoTIFFTags.from_xarray(build_raster(crs="EPSG:4326"), map_scale=10_000)


def test_geotiff_tags_refuse_physical_resolution_without_a_grid() -> None:
    source = xr.Dataset({"red": (("y", "x"), np.ones((2, 2), dtype="uint8"))})

    with pytest.raises(ValueError, match="regular grid"):
        GeoTIFFTags.from_xarray(source, map_scale=10_000)
