"""Temporal coverage coordinates preserve the underlying raster observations."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from geosave.geodata.core.raster.time import assign_bounds, read_bounds


def test_assign_bounds_preserves_lazy_pixels_and_original_coordinates():
    raster = xr.DataArray(
        da.ones((2, 2), chunks=(1, 2)),
        dims=("time", "x"),
        coords={"time": np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[us]")},
    )
    bounds = np.array(
        [["2025-01-01", "2025-01-31"], ["2025-02-01", "2025-02-28"]],
        dtype="datetime64[us]",
    )
    dated = assign_bounds(raster, bounds)

    assert dated.data is raster.data
    assert "start_datetime" not in raster.coords
    assert dated.start_datetime.dims == ("time",)
    starts, ends = read_bounds(dated.coords)
    np.testing.assert_array_equal(starts, bounds[:, 0])
    np.testing.assert_array_equal(ends, bounds[:, 1])

    shared = assign_bounds(raster, [bounds[0]])
    assert shared.start_datetime.ndim == 0
    with pytest.raises(ValueError, match="both"):
        read_bounds(shared.drop_vars("end_datetime").coords)
    with pytest.raises(ValueError, match="do not fit"):
        assign_bounds(raster, [bounds[0]] * 3)
