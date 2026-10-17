from __future__ import annotations

import pytest

from geosave.geodata.stac import StacQuery


def test_query_rejects_bbox_and_intersects_together() -> None:
    with pytest.raises(ValueError, match="bbox.*intersects"):
        StacQuery(
            collections=["example"],
            bbox=(0, 0, 1, 1),
            intersects={"type": "Point", "coordinates": [0, 0]},
        )
