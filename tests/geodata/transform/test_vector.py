"""Converting between geolocated rasters and vector features."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest

from tests.geodata.transform.conftest import grid_of


def _sliced_geographic_scene():
    """Build a raster whose sliced coordinates drift from its geobox's own."""
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.constructors import raster

    step = 1 / 3600
    grid = GeoBox.from_bbox(
        (13.0, 45.0, 13.0 + 40 * step, 45.0 + 40 * step),
        crs="EPSG:4326",
        resolution=step,
    )
    whole = raster({"red": (("y", "x"), np.ones(grid.shape, "uint16"))}, grid, nodata=0)
    return whole.isel(x=slice(5, 25), y=slice(5, 25))


def test_rasterize_lands_on_the_target_rasters_own_coordinates() -> None:
    import shapely
    import xarray as xr

    from geosave_engine.geodata.transform.nodata import mask

    scene = _sliced_geographic_scene()
    left, bottom, right, top = grid_of(scene).boundingbox.bbox
    plot = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, (left + right) / 2, top)],
        crs="EPSG:4326",
    )

    burned = plot.gs.rasterize(scene)

    xr.align(scene, burned, join="exact")
    masked = mask(scene, burned)
    assert (masked.red.values == 0).any() and (masked.red.values == 1).any()


def _flags(values, dtype):
    from odc.geo.geobox import GeoBox

    from geosave_engine.geodata.core.constructors import array

    pixels = np.asarray(values, dtype=dtype)
    side = pixels.shape[0] * 10
    grid = GeoBox.from_bbox((0, 0, side, side), crs="EPSG:32633", resolution=10)
    return array(pixels, grid, dims=("y", "x"))


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

    from geosave_engine.geodata.transform.vector import vectorize

    regions = vectorize(band)
    restored = regions.gs.rasterize(band, column="value", dtype="uint8")

    assert restored.dims == dims
    assert restored.gs.geobox == grid
    np.testing.assert_array_equal(restored, pixels)
