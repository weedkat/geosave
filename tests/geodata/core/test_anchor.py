"""Where and when data exists: the anchor a search or a load reads."""

from __future__ import annotations

import pandas as pd

from geosave.geodata import GeoAnchor


def _anchor() -> GeoAnchor:
    return GeoAnchor.from_coordinates(52, 13, 500, 10, timespan="2025-06-11")


def test_replacing_the_timespan_keeps_the_grid() -> None:
    anchor = _anchor()

    month = anchor.replace(timespan="2025-06")

    assert month.geobox == anchor.geobox
    start, end = month.timespan
    assert pd.Timestamp(start) == pd.Timestamp("2025-06-01")
    assert pd.Timestamp(end) >= pd.Timestamp("2025-06-30")


def test_replacing_the_timespan_with_none_makes_the_anchor_timeless() -> None:
    assert _anchor().replace(timespan=None).timespan is None


def test_replacing_leaves_the_original_anchor_unchanged() -> None:
    anchor = _anchor()

    anchor.replace(timespan="2025-06")

    assert anchor.format("{start:%Y-%m-%d}") == "2025-06-11"
