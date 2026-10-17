"""Collections read as the ACDD attrs on a Dataset's root, and described by their Items."""

from __future__ import annotations

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
    assert (
        collection.read_attrs(FOREST)
        == ACDD(
            id="forest",
            title="Forest",
            summary="Forest samples",
            keywords="forest, label",
            institution="GeoSave",
            license="CC-BY-4.0",
        ).to_attrs()
    )
