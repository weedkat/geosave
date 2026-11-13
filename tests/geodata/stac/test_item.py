"""Items read back as the files and variable attrs they describe."""

from __future__ import annotations


from datetime import UTC, datetime

import pandas as pd
import pystac
import pytest

from geosave.geodata import stac


def test_observation_properties_leave_out_what_a_raster_states():
    kept = stac.item.select_item_properties(
        {
            "datetime": "2025-01-15T00:00:00Z",
            "proj:shape": [10, 10],
            "eo:bands": [],
            "eo:cloud_cover": 4.2,
            "platform": "sentinel-2a",
        }
    )
    assert kept == {"eo:cloud_cover": 4.2, "platform": "sentinel-2a"}


def test_normalized_time_preserves_nominal_date_and_interval():
    from geosave.geodata.stac.item import create_item, normalize_time

    time = normalize_time(
        "2025-06-15T02:00:00+02:00",
        "2025-06-01",
        "2025-07-01",
    )
    properties = {"platform": "sentinel-2a", "instruments": ["msi"]}
    entry = create_item(id="composite", time=time, properties=properties)

    assert entry.datetime == datetime(2025, 6, 15, tzinfo=UTC)
    assert entry.properties["start_datetime"] == "2025-06-01T00:00:00+00:00"
    assert entry.properties["end_datetime"] == "2025-07-01T00:00:00+00:00"
    entry.properties["instruments"].append("other")
    assert properties == {"platform": "sentinel-2a", "instruments": ["msi"]}


@pytest.mark.parametrize(
    ("instant", "start", "end"),
    [
        (None, "2025-07-01", "2025-06-01"),
    ],
)
def test_invalid_observation_time_is_refused(instant, start, end):
    from geosave.geodata.stac.item import normalize_time

    with pytest.raises(ValueError):
        normalize_time(instant, start, end)


def test_an_item_period_is_its_stated_interval_or_its_instant():
    from datetime import UTC, datetime

    from geosave.geodata.stac.item import TIMELESS, read_item_period

    scene = pystac.Item("scene", None, None, datetime(2025, 6, 1, 10, tzinfo=UTC), {})
    composite = pystac.Item(
        "composite",
        None,
        None,
        None,
        {
            "start_datetime": "2025-06-01T00:00:00Z",
            "end_datetime": "2025-07-01T00:00:00Z",
        },
    )
    timeless = pystac.Item(
        "timeless",
        None,
        None,
        None,
        {
            "start_datetime": TIMELESS[0].isoformat(),
            "end_datetime": TIMELESS[1].isoformat(),
        },
    )

    assert read_item_period(scene) == (
        pd.Timestamp("2025-06-01T10:00"),
        pd.Timestamp("2025-06-01T10:00"),
    )
    assert read_item_period(composite) == (
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-07-01"),
    )
    assert read_item_period(timeless) is None
