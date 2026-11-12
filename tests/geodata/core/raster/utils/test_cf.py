from __future__ import annotations

import dask.array
import numpy as np


def test_encoding_time_bounds_leaves_the_callers_raster_unchanged():
    from geosave.geodata.core.raster.utils.cf import (
        decode_time_bounds,
        encode_time_bounds,
    )
    from tests.geodata.conftest import build_dated_raster

    raster = build_dated_raster()

    encoded = encode_time_bounds(raster)
    decoded = decode_time_bounds(encoded)

    assert "bounds" not in raster.time.attrs
    assert "units" not in raster.time.encoding
    assert encoded.time.attrs["bounds"] == "time_bounds"
    assert encoded.time_bounds.dims == ("time", "bounds")
    assert isinstance(encoded.red.data, dask.array.Array)
    for name in ("start_datetime", "end_datetime"):
        np.testing.assert_array_equal(decoded[name].values, raster[name].values)
