import dask
import numpy as np
import xarray as xr

from geosave.geodata.core.raster.utils.stack import extend_pixels, stack_rasters
from tests.conftest import build_dataset
from tests.geodata.conftest import _refuse, geobox


def build_step(day: str, grid, value: int) -> xr.Dataset:
    """One dated uint16 raster of B04 on `grid`, as an Item merges to."""
    pixels = np.full(grid.shape.yx, value, "uint16")
    raster = build_dataset({"B04": (("y", "x"), pixels)}, grid).chunk()
    return raster.assign_coords(time=np.datetime64(day, "ns"))


def test_rasters_on_one_extent_stack_lazily_without_changing_dtype():
    grid = geobox(shape=(4, 4))
    rasters = [build_step("2025-06-01", grid, 1), build_step("2025-06-06", grid, 2)]

    with dask.config.set(scheduler=_refuse):
        stacked = stack_rasters(rasters, grid, "time")

    assert stacked.B04.dims == ("time", "y", "x")
    assert stacked.B04.dtype == np.dtype("uint16")
    assert stacked.gs.geobox == grid
    np.testing.assert_array_equal(stacked.time.values, [r.time.values for r in rasters])
    np.testing.assert_array_equal(stacked.B04.isel(y=0, x=0).values, [1, 2])


def test_rasters_on_different_extents_stack_onto_the_geobox_with_nan_outside():
    grid = geobox(shape=(4, 4))
    rasters = [
        build_step("2025-06-01", grid, 1),
        build_step("2025-06-06", grid[:, 2:], 2),
    ]

    stacked = stack_rasters(rasters, grid, "time")

    assert stacked.B04.dtype == np.dtype("float32")
    np.testing.assert_array_equal(
        stacked.B04.isel(time=1, y=0).values, [np.nan, np.nan, 2, 2]
    )
    assert stacked.x.attrs == rasters[0].x.attrs
    # The CRS coordinate states the GeoBox's transform, not the first raster's.
    assert (
        stacked.spatial_ref.attrs["GeoTransform"]
        == rasters[0].spatial_ref.attrs["GeoTransform"]
    )
    assert stacked.gs.geobox == grid


def test_a_raster_lacking_a_variable_stacks_nan_for_it():
    grid = geobox(shape=(2, 2))
    full = build_step("2025-06-01", grid, 1).assign(B03=lambda raster: raster.B04)
    lacking = build_step("2025-06-06", grid, 2)

    stacked = stack_rasters([full, lacking], grid, "time")

    assert stacked.B03.dtype == np.dtype("float32")
    assert stacked.B03.isel(time=1).isnull().all()
    assert stacked.B04.dtype == np.dtype("uint16")


def test_pixels_extend_onto_a_larger_geobox_with_nan_outside():
    grid = geobox(shape=(4, 4))
    band = build_step("2025-06-01", grid[1:3, 2:4], 1).B04

    extended = extend_pixels(band, grid, np.dtype("float32"))

    values = extended.compute()
    assert values.shape == (4, 4)
    assert (values[1:3, 2:4] == 1).all()
    assert np.isnan(values[0]).all()
    assert np.isnan(values[1:3, :2]).all()
