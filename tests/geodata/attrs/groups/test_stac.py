"""The marker that makes a coordinate a STAC Item property."""

from __future__ import annotations

from geosave.geodata import attrs


def test_stac_property_round_trips_through_attrs() -> None:
    value = attrs.StacProperty.from_attrs({"stac_property": "eo:cloud_cover"})

    assert value.name == "eo:cloud_cover"
    assert attrs.StacProperty(name="platform").to_attrs() == {
        "stac_property": "platform"
    }
