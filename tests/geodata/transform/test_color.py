"""Tests for baking class colours into display channels."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave.geodata.core.constructors import array
from geosave.geodata.transform.color import colorize

GRID = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")


def landcover(classes=None, colors=None) -> xr.DataArray:
    """Build a class band, one pixel of it naming no class."""
    codes = np.array([[1, 1, 1], [1, 1, 9]], dtype="uint8")
    band = array(codes, GRID, dims=("y", "x"))
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


def test_colorize_refuses_a_band_without_classes_or_colours() -> None:
    with pytest.raises(ValueError, match="lists no classes"):
        colorize(landcover(None))
    with pytest.raises(ValueError, match="carry no colour"):
        colorize(landcover({1: "water"}))


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
