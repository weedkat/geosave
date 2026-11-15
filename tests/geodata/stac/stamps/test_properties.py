"""Properties ODC loaded as variables, as coordinates marked with their STAC key."""

import numpy as np
import xarray as xr
from odc.stac.model import PropertyLoadRequest

from geosave.geodata.stac.stamps import LoadSource
from geosave.geodata.stac.stamps.properties import (
    fuse_end,
    fuse_start,
    stamp_period,
    stamp_properties,
)
from tests.geodata.stac.stamps.conftest import scene


def loaded_properties(**variables) -> xr.Dataset:
    """Stand in for an ODC load: one (time,) variable per loaded property."""
    data = {}
    for name, values in variables.items():
        data[name] = ("time", np.asarray(values), {"units": "1"})
    return xr.Dataset(data, coords={"time": [np.datetime64("2025-01-01")]})


def test_a_property_variable_becomes_a_coordinate_marked_with_its_key() -> None:
    requests = PropertyLoadRequest.from_user_input(["eo:cloud_cover"])

    out = stamp_properties(
        loaded_properties(eo_cloud_cover=[10.0]),
        LoadSource([scene()], properties=requests),
    )

    assert "eo_cloud_cover" in out.coords
    assert out["eo_cloud_cover"].dims == ("time",)
    assert out.gs.stac_properties == {"eo:cloud_cover": "eo_cloud_cover"}


def test_a_renamed_property_keeps_its_stac_key() -> None:
    requests = PropertyLoadRequest.from_user_input(
        [{"key": "eo:cloud_cover", "name": "cloud"}]
    )

    out = stamp_properties(
        loaded_properties(cloud=[10.0]), LoadSource([scene()], properties=requests)
    )

    assert out["cloud"].attrs["stac_property"] == "eo:cloud_cover"


def test_no_requests_leave_the_load_unchanged() -> None:
    loaded = loaded_properties(other=[1.0])

    out = stamp_properties(loaded, LoadSource([scene()]))

    xr.testing.assert_identical(out, loaded)


def test_a_step_spans_its_items_earliest_start_to_latest_end() -> None:
    starts = [
        "2025-06-01T10:31:00+00:00",
        "2025-06-01T10:30:00Z",
        "2025-06-01T10:35:00",
    ]
    ends = ["2025-06-01T10:30:40Z", "2025-06-01T10:31:40.5Z"]

    assert fuse_start(starts) == np.datetime64("2025-06-01T10:30:00")
    assert fuse_end(ends) == np.datetime64("2025-06-01T10:31:40.5")
    assert np.isnat(fuse_start([]))


def test_a_step_without_a_stated_period_takes_its_label() -> None:
    labels = np.array(["2025-06-01T10:30", "2025-06-05T10:30"], "datetime64[ns]")
    stated = np.array(["2025-06-01T10:00", "NaT"], "datetime64[ns]")
    loaded = xr.Dataset(
        coords={
            "time": labels,
            "start_datetime": ("time", stated, {"units": "1", "nodata": "NaT"}),
            "end_datetime": ("time", stated, {"units": "1", "nodata": "NaT"}),
        }
    )

    out = stamp_period(loaded)

    np.testing.assert_array_equal(out["start_datetime"], [stated[0], labels[1]])
    assert out["end_datetime"].attrs == {}
