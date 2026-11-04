from __future__ import annotations

from datetime import UTC, datetime

import pystac
from shapely.geometry import mapping

from geosave.geodata import io
from tests.geodata.conftest import build_raster


def entry(collection="optical"):
    item = pystac.Item(
        "one",
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        [0, 0, 1, 1],
        datetime(2025, 1, 1, tzinfo=UTC),
        {},
        collection=collection,
    )
    item.add_asset(
        "red", pystac.Asset("/tmp/red.tif", media_type=pystac.MediaType.GEOTIFF)
    )
    return item


def create_native_item(tmp_path, name="scene"):
    source = build_raster(times=1).isel(time=0)
    saved = io.formats.geotiff.write_cog(source, tmp_path / f"{name}.tif")
    shape = source.gs.geobox.extent.to_crs("EPSG:4326").geom
    item = pystac.Item(
        name,
        mapping(shape),
        list(shape.bounds),
        datetime(2025, 6, 1, tzinfo=UTC),
        {"eo:cloud_cover": 12.5, "provider:quality": "good"},
    )
    item.add_asset(
        "reflectance",
        pystac.Asset(str(saved), media_type=pystac.MediaType.COG, roles=["data"]),
    )
    item.add_asset(
        "thumbnail",
        pystac.Asset(
            "https://example.com/thumb.jpg",
            media_type=pystac.MediaType.JPEG,
            roles=["thumbnail"],
        ),
    )
    return item, source
