"""Tests for baking class colours into display channels."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from tests.conftest import build_dataarray
from geosave.geodata.transform.color import colorize

GRID = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")


def landcover(classes=None, colors=None) -> xr.DataArray:
    """Build a class band, one pixel of it naming no class."""
    codes = np.array([[1, 1, 1], [1, 1, 9]], dtype="uint8")
    band = build_dataarray(codes, GRID, dims=("y", "x"))
    return band if classes is None else band.gs.write_legend(classes, colors)


def test_colorize_bakes_each_class_colour_onto_the_grid() -> None:
    band = landcover({1: "water"}, {1: "#0000ff"})

    colored = colorize(band)

    assert colored.dims == ("band", "y", "x")
    assert list(colored.band.values) == ["red", "green", "blue"]
    assert colored.gs.geobox == GRID
    assert colored.y.attrs["standard_name"] == "latitude"
    np.testing.assert_array_equal(colored.sel(band="blue")[0], np.ones(3))
    assert np.isnan(colored.values[:, 1, 2]).all()
    xr.testing.assert_identical(band.gs.colorize(), colored)


def test_colorize_refuses_a_band_without_classes() -> None:
    with pytest.raises(ValueError, match="lists no classes"):
        colorize(landcover(None))


def test_colorize_keeps_pixels_lazy_and_preserves_scalar_coordinates() -> None:
    from dask.callbacks import Callback

    band = landcover({1: "water"}, {1: "#0000ff"})
    band = band.assign_coords(time=np.datetime64("2025-01-01")).chunk({"x": 1})
    with Callback(pretask=lambda *_: pytest.fail("colorize computed pixels")):
        colored = colorize(band)
    assert colored.chunks is not None
    xr.testing.assert_identical(colored.time, band.time)
    assert colored.dtype == np.float32
    np.testing.assert_array_equal(colored.sel(band="blue").compute()[0], np.ones(3))
    assert np.isnan(colored.compute().values[:, 1, 2]).all()


def test_colorize_draws_an_uncoloured_class_in_its_palette_colour() -> None:
    from geosave.geodata.utils.color import CATEGORY_PALETTE, parse_color

    colored = colorize(landcover({1: "water"})).compute()

    red, green, blue = (channel / 255 for channel in parse_color(CATEGORY_PALETTE[0]))
    named = np.isfinite(colored.sel(band="red").values)
    assert colored.sel(band="red").values[named][0] == pytest.approx(red)
