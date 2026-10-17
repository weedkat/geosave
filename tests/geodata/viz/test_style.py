import colorcet as cc
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geosave.geodata.viz.style import (
    Stretch,
    get_category_colors,
    get_ramp,
    to_rgb255,
)


def band(values) -> xr.DataArray:
    return xr.DataArray(np.asarray(values), dims=("y", "x"))


class TestStretch:
    def test_fits_the_middle_percentiles(self) -> None:
        clim = Stretch().get_limits(np.linspace(0.0, 1.0, 101))

        assert clim == pytest.approx((0.02, 0.98))

    def test_eight_bit_draws_as_stored(self) -> None:
        assert Stretch().get_limits(np.zeros(4, "uint8")) == (0.0, 255.0)

    def test_refuses_flat_channels_and_inverted_limits(self) -> None:
        with pytest.raises(ValueError, match="flat"):
            Stretch().get_limits(np.ones(10))
        with pytest.raises(ValueError, match="inverted"):
            Stretch((1.0, 0.0)).get_limits(np.ones(10))

    def test_refuses_nothing_to_draw(self) -> None:
        with pytest.raises(ValueError, match="nothing to draw"):
            Stretch().get_limits(np.array([np.nan]))

    def test_apply_keeps_a_dataarray_and_its_nan(self) -> None:
        stretched = Stretch((0.0, 10.0)).apply(band([[5.0, 20.0, np.nan]]))

        assert isinstance(stretched, xr.DataArray)
        assert stretched.values[0, :2].tolist() == [0.5, 1.0]
        assert np.isnan(stretched.values[0, 2])

    def test_apply_takes_an_array(self) -> None:
        stretched = Stretch((0.0, 10.0)).apply(np.array([[0.0, 5.0, 10.0, np.nan]]))

        np.testing.assert_allclose(stretched, [[0.0, 0.5, 1.0, np.nan]])


def test_ramp_has_256_colours() -> None:
    ramp = get_ramp("viridis")

    assert ramp.shape == (256, 3) and ramp.dtype == np.uint8


def test_colour_names_read_as_rgb() -> None:
    assert to_rgb255("#ff8000") == (255, 128, 0)


class TestColorKey:
    def test_colours_categories_in_sorted_order(self) -> None:
        colors = get_category_colors(pd.Series(["rice", "palm", None, "rice"]))

        assert colors == {"palm": cc.glasbey_dark[0], "rice": cc.glasbey_dark[1]}

    def test_dark_theme_takes_light_hues(self) -> None:
        assert get_category_colors(pd.Series(["rice"]), "dark") == {
            "rice": cc.glasbey_light[0]
        }
