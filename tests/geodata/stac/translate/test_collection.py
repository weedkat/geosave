"""Collections read as the ACDD attrs on a Dataset's root, and described by their Items."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac

from geosave.geodata.attrs import ACDD
from geosave.geodata.stac.translate import collection

FOREST = pystac.Collection(
    "forest",
    "Forest samples",
    pystac.Extent.from_dict(
        {
            "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
            "temporal": {"interval": [[None, None]]},
        }
    ),
    title="Forest",
    keywords=["forest", "label"],
    providers=[pystac.Provider("GeoSave")],
    license="CC-BY-4.0",
)


def test_a_collection_reads_as_the_acdd_attrs_of_its_data() -> None:
    assert collection.read(FOREST) == ACDD(
        id="forest",
        title="Forest",
        summary="Forest samples",
        keywords="forest, label",
        institution="GeoSave",
        license="CC-BY-4.0",
    )


def test_describe_sets_the_extent_of_its_items() -> None:
    item = pystac.Item(
        "s0",
        {"type": "Point", "coordinates": [10.0, 45.0]},
        [10.0, 45.0, 10.0, 45.0],
        datetime(2025, 6, 1, tzinfo=UTC),
        {},
        collection="forest",
    )

    described = collection.describe([item], [FOREST])

    assert described["forest"].extent.spatial.bboxes == [[10.0, 45.0, 10.0, 45.0]]
    assert described["forest"].title == "Forest"
