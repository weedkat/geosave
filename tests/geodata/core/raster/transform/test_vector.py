"""Converting between geolocated rasters and vector features."""

from __future__ import annotations

import numpy as np
import pytest

from odc.geo.geobox import GeoBox
from tests.conftest import build_dataarray


def _flags(values, dtype):
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataarray

    pixels = np.asarray(values, dtype=dtype)
    side = pixels.shape[0] * 10
    grid = GeoBox.from_bbox((0, 0, side, side), crs="EPSG:32633", resolution=10)
    return build_dataarray(pixels, grid, dims=("y", "x"))


def test_vectorize_joins_regions_touching_at_a_corner_into_valid_geometry() -> None:

    diagonal = _flags([[1, 0], [0, 1]], "uint8")

    regions = diagonal.gs.vectorize(connectivity=8)

    assert regions.geometry.is_valid.all()
    ones = regions[regions["value"] == 1]
    assert ones.geometry.area.sum() == 200


@pytest.mark.parametrize("dtype", ["int64"])
def test_vectorize_takes_the_integer_dtypes_a_model_emits(dtype: str) -> None:

    regions = _flags([[1, 0], [0, 1]], dtype).gs.vectorize()

    assert sorted(regions["value"].tolist()) == [0, 0, 1, 1]


def test_vectorize_creates_one_row_per_connected_flag_region(raster) -> None:
    flags = build_dataarray(
        np.array([[1, 0, 1], [1, 0, 0], [2, 2, 0]], dtype="uint8"),
        GeoBox.from_bbox(raster.gs.geobox.boundingbox, shape=(3, 3), crs=raster.gs.crs),
        dims=("y", "x"),
        nodata=0,
    )
    vector = flags.gs.vectorize(value_name="class_id")
    assert list(vector.class_id) == [1, 1, 2]
    assert vector.gs.crs == flags.gs.crs
    assert vector.geometry.is_valid.all()


def test_vectorize_requires_one_spatial_plane(raster) -> None:
    flags = raster.red.expand_dims(time=["2025-01-01"])
    with pytest.raises(ValueError, match="two-dimensional"):
        flags.gs.vectorize()
