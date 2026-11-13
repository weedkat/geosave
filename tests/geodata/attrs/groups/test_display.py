"""How a raster draws: its RGB composite."""

from __future__ import annotations

from geosave.geodata import attrs


def test_rgb_round_trips_through_attrs() -> None:
    display = attrs.Display(rgb=["B04", "B03", "B02"])

    assert attrs.Display.from_attrs(display.to_attrs()) == display
