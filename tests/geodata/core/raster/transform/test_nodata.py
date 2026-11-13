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


def test_a_bare_numpy_mask_covers_every_step_of_a_time_series() -> None:
    bare = np.array([[True, True, False, False]] * 4)

    masked = scene(times=2).gs.mask(bare)

    assert list(masked.red.values[0][0]) == [7, 7, 0, 0]
    assert list(masked.red.values[1][0]) == [7, 7, 0, 0]


def test_mask_rejects_conflicting_crs_before_computing_pixels():
    from dask.callbacks import Callback

    source = scene().chunk({"y": 2, "x": 2})
    valid = (source.red > 0).odc.assign_crs("EPSG:32632")

    with Callback(pretask=lambda *_: pytest.fail("mask computed pixels")):
        with pytest.raises(ValueError, match="different CRSs"):
            source.gs.mask(valid)


def test_mask_uses_each_variables_packing_fill_value():
    source = scene()
    source["nir"] = source.nir.gs.write_nodata(255)
    result = source.gs.mask(classification().isin(CLEAR))
    assert result.red.values[0, -1] == 0
    assert result.nir.values[0, -1] == 255
