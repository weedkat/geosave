from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave_engine.model.encoder.time import time_labels


def _raster(times):
    return xr.Dataset(coords={"time": np.asarray(times, dtype="datetime64[ns]")})


def test_time_labels_preserve_frame_order():
    actual = time_labels(_raster(["2024-02-01", "2023-01-01"]))
    assert [value.isoformat() for value in actual] == [
        "2024-02-01T00:00:00",
        "2023-01-01T00:00:00",
    ]


def test_a_scalar_time_is_one_label():
    one = _raster(["2024-01-02"]).isel(time=0)
    assert [value.isoformat() for value in time_labels(one)] == ["2024-01-02T00:00:00"]


@pytest.mark.parametrize("times", [[], ["NaT"]])
def test_time_labels_reject_empty_or_missing_instants(times):
    with pytest.raises(ValueError, match="time"):
        time_labels(_raster(times))


def test_a_raster_without_time_has_no_labels():
    with pytest.raises(ValueError, match="time"):
        time_labels(xr.Dataset())
