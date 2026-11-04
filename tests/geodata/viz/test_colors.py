import numpy as np
import pytest
import xarray as xr

from geosave.geodata.viz.colors import (
    find_stretch_limits,
    stretch_channels,
)


def band(values) -> xr.DataArray:
    return xr.DataArray(np.asarray(values), dims=("y", "x"))


class TestStretch:
    def test_fits_the_middle_percentiles(self) -> None:
        clim = find_stretch_limits(np.linspace(0.0, 1.0, 101))

        assert clim == pytest.approx((0.02, 0.98))

    def test_eight_bit_draws_as_stored(self) -> None:
        assert find_stretch_limits(np.zeros(4, "uint8")) == (0.0, 255.0)

    def test_refuses_flat_channels_and_inverted_limits(self) -> None:
        with pytest.raises(ValueError, match="flat"):
            find_stretch_limits(np.ones(10))
        with pytest.raises(ValueError, match="inverted"):
            stretch_channels(np.ones(10), (1.0, 0.0))

    def test_refuses_nothing_to_draw(self) -> None:
        with pytest.raises(ValueError, match="nothing to draw"):
            find_stretch_limits(np.array([np.nan]))

    def test_keeps_a_dataarray_and_its_nan(self) -> None:
        stretched = stretch_channels(band([[5.0, 20.0, np.nan]]), (0.0, 10.0))

        assert isinstance(stretched, xr.DataArray)
        assert stretched.values[0, :2].tolist() == [0.5, 1.0]
        assert np.isnan(stretched.values[0, 2])
