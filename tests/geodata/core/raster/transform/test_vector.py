"""Converting between geolocated rasters and vector features."""

from __future__ import annotations

import numpy as np
import pytest
from dask.callbacks import Callback

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


@pytest.mark.parametrize("dtype", ["int64", "int8", "uint32"])
def test_vectorize_takes_the_integer_dtypes_a_model_emits(dtype: str) -> None:

    regions = _flags([[1, 0], [0, 1]], dtype).gs.vectorize()

    assert sorted(regions["value"].tolist()) == [0, 0, 1, 1]


def test_vectorize_refuses_integers_no_supported_dtype_holds() -> None:

    with pytest.raises(ValueError, match="int32"):
        _flags([[2**40, 0], [0, 1]], "int64").gs.vectorize()


def test_geographic_polygonization_and_rasterization_round_trip():
    import xarray as xr
    from affine import Affine
    from odc.geo import GeoBox
    from odc.geo.xr import xr_coords

    dims = ("y", "x")
    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    pixels = np.array([[1, 1, 0], [0, 2, 2]], dtype="uint8")
    band = xr.DataArray(pixels, dims=dims, coords=xr_coords(grid, dims=dims))
    band.encoding["grid_mapping"] = "spatial_ref"

    regions = band.gs.vectorize()
    restored = regions.gs.rasterize(band, column="value", dtype="uint8")

    assert restored.dims == dims
    assert restored.gs.geobox == grid
    np.testing.assert_array_equal(restored, pixels)


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


def test_vectorize_preserves_the_value_schema_when_every_pixel_is_nodata(
    raster,
) -> None:
    flags = build_dataarray(
        np.zeros(raster.gs.geobox.shape, dtype="uint8"),
        raster.gs.geobox,
        dims=("y", "x"),
        nodata=0,
    )
    vector = flags.gs.vectorize(value_name="class_id")
    restored = vector.gs.rasterize(like=flags, column="class_id", fill=0, dtype="uint8")
    assert list(vector) == ["class_id", "geometry"]
    assert vector.class_id.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(restored.values, np.zeros(flags.shape, "uint8"))


def test_vectorize_computes_dask_flags_explicitly(raster) -> None:
    flags = raster.red.chunk({"y": 1, "x": 1})
    started: list[object] = []
    with Callback(pretask=lambda key, *_: started.append(key)):
        flags.gs.vectorize()
    assert started


def test_vectorize_requires_one_spatial_plane(raster) -> None:
    flags = raster.red.expand_dims(time=["2025-01-01"])
    with pytest.raises(ValueError, match="two-dimensional"):
        flags.gs.vectorize()


def test_vectorize_rejects_unsupported_values(raster) -> None:
    flags = raster.red.astype("complex64")
    with pytest.raises(ValueError, match="dtype"):
        flags.gs.vectorize()


def test_vectorize_rejects_same_shape_mask_on_another_grid(raster) -> None:
    shifted = raster.red.assign_coords(x=raster.x + 10)
    with pytest.raises(ValueError, match="exact flags grid"):
        raster.red.gs.vectorize(mask=shifted)
