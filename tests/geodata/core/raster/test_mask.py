"""Marking pixels nodata and decoding stored values."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from tests.conftest import build_dataset as build_raster


UTM = "EPSG:32633"


CLEAR = [4, 5, 6, 7]


def utm_box() -> GeoBox:
    """Build a small projected grid."""
    return GeoBox.from_bbox((0, 0, 40, 40), crs=UTM, resolution=10)


def scene(*, times: int = 0, nodata: float | int | None = 0) -> xr.Dataset:
    """Build a raster whose pixels all read 7, optionally over a time axis."""
    box = utm_box()
    if times:
        return build_raster(
            {
                "red": (
                    ("time", *("y", "x")),
                    np.full((times, *box.shape), 7, "uint16"),
                )
            },
            box,
            time=np.array(["2025-06-01", "2025-06-11"][:times], "datetime64[ns]"),
            nodata=nodata,
        )
    return build_raster(
        {
            "red": (("y", "x"), np.full(box.shape, 7, "uint16")),
            "nir": (("y", "x"), np.full(box.shape, 8, "uint16")),
        },
        box,
        nodata=nodata,
    )


def classification() -> xr.DataArray:
    """Build a scene classification band, half ground and half cloud."""
    dims = ("y", "x")
    return xr.DataArray(np.array([[4, 4, 9, 9]] * 4), dims=dims)


def test_masking_writes_the_declared_fill_without_promoting_the_dtype() -> None:
    source = scene()
    masked = source.gs.mask(classification().isin(CLEAR))

    assert masked.red.dtype == np.dtype("uint16")
    assert list(masked.red.values[0]) == [7, 7, 0, 0]
    assert masked.gs.geobox == source.gs.geobox
    assert masked.red.attrs["_FillValue"] == 0


def test_a_flat_mask_covers_every_step_of_a_time_series() -> None:
    masked = scene(times=2).gs.mask(classification().isin(CLEAR))

    assert masked.red.shape == (2, 4, 4)
    assert list(masked.red.values[0][0]) == [7, 7, 0, 0]
    assert list(masked.red.values[1][0]) == [7, 7, 0, 0]


def test_a_non_boolean_mask_is_refused() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        scene().gs.mask(classification())


def test_a_variable_carrying_no_fill_is_refused_rather_than_promoted() -> None:
    with pytest.raises(ValueError, match="carries no fill value"):
        scene(nodata=None).gs.mask(classification().isin(CLEAR))


def test_a_given_fill_is_written_and_declared() -> None:
    masked = scene().gs.mask(classification().isin(CLEAR), fill=255)

    assert list(masked.red.values[0]) == [7, 7, 255, 255]
    assert masked.red.attrs["_FillValue"] == 255
    assert masked.red.dtype == np.dtype("uint16")


@pytest.mark.parametrize("fill", [-1, 70000])
def test_a_fill_the_dtype_cannot_hold_is_refused_rather_than_wrapped(
    fill: int,
) -> None:
    with pytest.raises(ValueError, match="does not fit this uint16 band"):
        scene(nodata=None).gs.mask(classification().isin(CLEAR), fill=fill)


def test_a_nan_fill_is_refused_for_an_integer_variable() -> None:
    with pytest.raises(ValueError, match="does not fit this uint16 band"):
        scene(nodata=None).gs.mask(classification().isin(CLEAR), fill=float("nan"))


def test_a_nan_fill_is_accepted_by_a_float_variable() -> None:
    box = utm_box()
    floating = build_raster(
        {"elevation": (("y", "x"), np.ones(box.shape, "float32"))}, box
    )

    masked = floating.gs.mask(classification().isin(CLEAR), fill=float("nan"))

    assert np.isnan(masked.elevation.values[0][-1])


def test_a_bare_numpy_mask_covers_every_step_of_a_time_series() -> None:
    bare = np.array([[True, True, False, False]] * 4)

    masked = scene(times=2).gs.mask(bare)

    assert list(masked.red.values[0][0]) == [7, 7, 0, 0]
    assert list(masked.red.values[1][0]) == [7, 7, 0, 0]


def test_a_bare_numpy_mask_off_the_grid_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="pass a DataArray"):
        scene().gs.mask(np.ones((3, 3), bool))


def test_a_mask_spanning_an_axis_the_raster_lacks_is_refused() -> None:
    dims = ("y", "x")
    extra = xr.DataArray(np.ones((2, 4, 4), bool), dims=("depth", *dims))

    with pytest.raises(ValueError, match="does not carry"):
        scene().gs.mask(extra)


@pytest.mark.parametrize("kind", ["array", "dataset"])
def test_mask_rejects_conflicting_crs_before_computing_pixels(kind):
    from dask.callbacks import Callback

    source = scene().chunk({"y": 2, "x": 2})
    valid = (source.red > 0).gs.write_crs("EPSG:32632")
    if kind == "array":
        source = source.red

    with Callback(pretask=lambda *_: pytest.fail("mask computed pixels")):
        with pytest.raises(ValueError, match="different CRSs"):
            source.gs.mask(valid)


@pytest.mark.parametrize("kind", ["array", "dataset"])
def test_mask_preserves_grid_with_equivalent_crs_carriers(kind):
    from dask.callbacks import Callback

    source = scene().chunk({"y": 2, "x": 2})
    valid = source.red > 0
    valid = valid.assign_coords(spatial_ref=valid.spatial_ref.copy(data=0))
    if kind == "array":
        source = source.red

    with Callback(pretask=lambda *_: pytest.fail("mask computed pixels")):
        masked = source.gs.mask(valid)

    xr.testing.assert_identical(masked.spatial_ref, source.spatial_ref)
    assert masked.gs.geobox == source.gs.geobox
    xr.testing.assert_equal(masked.compute(), source.compute())


def test_dataset_mask_broadcasts_time_to_static_variables():
    source = scene(times=2)
    source["static"] = source.red.isel(time=0, drop=True)
    valid = source.red > 0
    valid[1] = False

    masked = source.gs.mask(valid)

    assert masked.static.dims == ("y", "x", "time")
    np.testing.assert_array_equal(masked.static.isel(time=0), 7)
    np.testing.assert_array_equal(masked.static.isel(time=1), 0)
    assert masked.static.dtype == source.static.dtype
    assert masked.gs.geobox == source.gs.geobox


def test_numpy_mask_uses_spatial_axis_order_with_lazy_transposed_pixels():
    from dask.callbacks import Callback

    source = scene(times=2).red.transpose("x", "time", "y").chunk({"time": 1})
    dims = source.gs.spatial_dims
    mask = np.ones(tuple(source.sizes[dim] for dim in dims), dtype=bool)
    mask[0, 1] = False
    with Callback(pretask=lambda *_: pytest.fail("mask computed pixels")):
        result = source.gs.mask(mask=mask)
    expected = source.where(xr.DataArray(mask, dims=dims), 0)
    xr.testing.assert_equal(result.compute(), expected.compute())


def test_named_time_mask_broadcasts_without_requiring_spatial_axes():
    source = scene(times=2)
    mask = xr.DataArray([True, False], dims="time", coords={"time": source.time})
    result = source.gs.mask(mask)
    np.testing.assert_array_equal(result.red.isel(time=0), 7)
    np.testing.assert_array_equal(result.red.isel(time=1), 0)


def test_mask_uses_each_variables_packing_fill_value():
    source = scene()
    source["nir"] = source.nir.gs.write_nodata(255)
    result = source.gs.mask(classification().isin(CLEAR))
    assert result.red.values[0, -1] == 0
    assert result.nir.values[0, -1] == 255


def test_missing_value_alone_does_not_supply_a_packing_fill_value():
    source = scene(nodata=None).red
    source.attrs["missing_value"] = 99
    with pytest.raises(ValueError, match="no fill value in packing"):
        source.gs.mask(classification().isin(CLEAR))
