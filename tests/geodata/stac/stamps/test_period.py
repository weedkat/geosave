"""Coverage coordinates on ODC's grouped timeline."""

import numpy as np
import xarray as xr

from geosave.geodata.stac.stamps.period import (
    fuse_end,
    fuse_start,
    stamp_observation_period,
)
from tests.geodata.stac.stamps.conftest import scene


def test_a_step_spans_its_items_earliest_start_to_latest_end():
    starts = [
        "2025-06-01T10:31:00+00:00",
        "2025-06-01T10:30:00Z",
        "2025-06-01T10:35:00",
    ]
    ends = ["2025-06-01T10:30:40Z", "2025-06-01T10:31:40.5Z"]
    assert fuse_start(starts) == np.datetime64("2025-06-01T10:30:00")
    assert fuse_end(ends) == np.datetime64("2025-06-01T10:31:40.5")
    assert np.isnat(fuse_start([]))


def test_a_step_without_a_stated_period_takes_its_label():
    labels = np.array(["2025-06-01T10:30", "2025-06-05T10:30"], "datetime64[ns]")
    stated = np.array(["2025-06-01T10:00", "NaT"], "datetime64[ns]")
    loaded = xr.Dataset(
        {
            name: ("time", stated, {"units": "1", "nodata": "NaT"})
            for name in ["start_datetime", "end_datetime"]
        },
        coords={"time": labels},
    )
    out = stamp_observation_period(loaded, [scene()])
    assert "start_datetime" in out.coords
    np.testing.assert_array_equal(out["start_datetime"], [stated[0], labels[1]])
    assert out["end_datetime"].attrs == {}
