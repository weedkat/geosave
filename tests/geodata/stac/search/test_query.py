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


def test_set_region_preserves_polygon_and_replaces_selectors():
    from odc.geo.geom import Geometry

    from geosave.geodata import GeoRegion

    polygon = Geometry(
        {"type": "Polygon", "coordinates": [[[0, 0], [100, 0], [0, 100], [0, 0]]]},
        "EPSG:3857",
    )
    region = GeoRegion.from_geometry(polygon)
    query = StacQuery(
        collections=["s2"], bbox=(0, 0, 1, 1), datetime="2020"
    ).set_filter("eo:cloud_cover < 20")
    changed = query.set_region(region)
    assert changed.intersects == polygon.to_crs("EPSG:4326").json
    assert changed.bbox is None
    assert changed.datetime is None
    assert changed.filter == query.filter
    assert query.bbox == (0, 0, 1, 1)
