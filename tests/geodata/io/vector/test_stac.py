"""Shared Collection preparation for STAC publication."""

from datetime import UTC, datetime

import pystac

from geosave.geodata import stac
from geosave.geodata.io.vector._stac import prepare_collections


def test_describe_sets_the_extent_of_its_items() -> None:
    definition = stac.create_collection(
        "forest", description="Forest samples", title="Forest"
    )
    item = pystac.Item(
        "s0",
        {"type": "Point", "coordinates": [10.0, 45.0]},
        [10.0, 45.0, 10.0, 45.0],
        datetime(2025, 6, 1, tzinfo=UTC),
        {},
        collection="forest",
    )

    built = prepare_collections([item], [definition])

    assert built["forest"].extent.spatial.bboxes == [[10.0, 45.0, 10.0, 45.0]]
    assert built["forest"].title == "Forest"
