"""Converting between geolocated rasters and vector features."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from dask.callbacks import Callback
from shapely.geometry import Polygon

from odc.geo.geobox import GeoBox
from tests.conftest import build_dataarray
from geosave.geodata.transform.vector import rasterize, vectorize
from geosave.geodata.utils.geometry import to_shapely
from tests.geodata.transform.conftest import grid_of


def _sliced_geographic_scene():
    """Build a raster whose sliced coordinates drift from its geobox's own."""
    from odc.geo.geobox import GeoBox

    from tests.conftest import build_dataset

    step = 1 / 3600
    grid = GeoBox.from_bbox(
        (13.0, 45.0, 13.0 + 40 * step, 45.0 + 40 * step),
        crs="EPSG:4326",
        resolution=step,
    )
    whole = build_dataset(
        {"red": (("y", "x"), np.ones(grid.shape, "uint16"))}, grid, nodata=0
    )
    return whole.isel(x=slice(5, 25), y=slice(5, 25))


def test_rasterize_lands_on_the_target_rasters_own_coordinates() -> None:
    import shapely
    import xarray as xr

    from geosave.geodata.transform.nodata import mask

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


def test_a_presence_mask_refuses_a_truthy_fill() -> None:
    import shapely

    scene = _sliced_geographic_scene()
    left, bottom, right, top = grid_of(scene).boundingbox.bbox
    plot = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, right, top)], crs="EPSG:4326"
    )

    with pytest.raises(ValueError, match="presence mask"):
        plot.gs.rasterize(scene, fill=1)


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

    from geosave.geodata.transform.vector import vectorize

    regions = vectorize(band)
    restored = regions.gs.rasterize(band, column="value", dtype="uint8")

    assert restored.dims == dims
    assert restored.gs.geobox == grid
    np.testing.assert_array_equal(restored, pixels)


def test_rasterizing_onto_an_explicit_grid_is_undated(grid):
    plot = gpd.GeoDataFrame(geometry=[grid.extent.geom], crs=grid.crs)
    burned = plot.gs.rasterize(grid)
    assert burned.gs.geobox == grid
    assert "time" not in burned.coords


def test_vectorize_creates_one_row_per_connected_flag_region(raster) -> None:
    flags = build_dataarray(
        np.array([[1, 0, 1], [1, 0, 0], [2, 2, 0]], dtype="uint8"),
        GeoBox.from_bbox(raster.gs.geobox.boundingbox, shape=(3, 3), crs=raster.gs.crs),
        dims=("y", "x"),
        nodata=0,
    )
    vector = vectorize(flags, value_name="class_id")
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
    vector = vectorize(flags, value_name="class_id")
    restored = rasterize(vector, like=flags, column="class_id", fill=0, dtype="uint8")
    assert list(vector) == ["class_id", "geometry"]
    assert vector.class_id.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(restored.values, np.zeros(flags.shape, "uint8"))


def test_vectorize_computes_dask_flags_explicitly(raster) -> None:
    flags = raster.red.chunk({"y": 1, "x": 1})
    started: list[object] = []
    with Callback(pretask=lambda key, *_: started.append(key)):
        vectorize(flags)
    assert started


def test_vectorize_requires_one_spatial_plane(raster) -> None:
    flags = raster.red.expand_dims(time=["2025-01-01"])
    with pytest.raises(ValueError, match="two-dimensional"):
        vectorize(flags)


def test_vectorize_rejects_unsupported_values(raster) -> None:
    flags = raster.red.astype("complex64")
    with pytest.raises(ValueError, match="dtype"):
        vectorize(flags)


def test_vectorize_rejects_same_shape_mask_on_another_grid(raster) -> None:
    shifted = raster.red.assign_coords(x=raster.x + 10)
    with pytest.raises(ValueError, match="exact flags grid"):
        vectorize(raster.red, mask=shifted)


@pytest.mark.parametrize("problem", ["null", "empty"])
def test_rasterize_refuses_a_missing_geometry(raster, problem: str) -> None:
    geometry = None if problem == "null" else Polygon()
    vector = gpd.GeoDataFrame(geometry=[geometry], crs=raster.gs.crs)
    with pytest.raises(ValueError, match="null, empty, or invalid"):
        rasterize(vector, raster)


def test_rasterize_refuses_a_geometry_naming_no_ground(raster) -> None:
    bowtie = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])], crs=raster.gs.crs
    )
    with pytest.raises(ValueError, match="invalid geometry"):
        rasterize(bowtie, raster)


def test_rasterize_without_column_returns_boolean_presence(raster) -> None:
    vector = gpd.GeoDataFrame(
        geometry=[to_shapely(raster.gs.geobox.extent)], crs=raster.gs.geobox.extent.crs
    )
    mask = rasterize(vector, like=raster)
    assert mask.dtype == np.dtype("bool")
    assert mask.values.all()


def test_rasterize_later_rows_win_on_overlap(raster) -> None:
    bounds = raster.gs.geobox.extent.geom
    vector = gpd.GeoDataFrame(
        {"class_id": [1, 2]}, geometry=[bounds, bounds], crs=raster.gs.crs
    )
    result = rasterize(vector, like=raster, column="class_id", dtype="uint8")
    assert (result.values == 2).all()


def test_rasterize_rejects_values_outside_dtype(raster) -> None:
    vector = gpd.GeoDataFrame(
        [{"class_id": 300}],
        geometry=[to_shapely(raster.gs.geobox.extent)],
        crs=raster.gs.geobox.extent.crs,
    )
    with pytest.raises(ValueError, match="uint8"):
        rasterize(vector, like=raster, column="class_id", dtype="uint8")


def test_rasterize_rejects_non_numeric_property(raster) -> None:
    vector = gpd.GeoDataFrame(
        [{"label": "forest"}],
        geometry=[to_shapely(raster.gs.geobox.extent)],
        crs=raster.gs.geobox.extent.crs,
    )
    with pytest.raises(ValueError, match="numeric raster dtype"):
        rasterize(vector, like=raster, column="label")


def test_rasterize_empty_vector_returns_only_fill(raster) -> None:
    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=raster.gs.crs))
    result = rasterize(empty, like=raster, fill=0)
    assert not result.values.any()
