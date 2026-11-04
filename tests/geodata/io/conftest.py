from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pystac
import pytest
import rasterio
from pystac.extensions.projection import ProjectionExtension

from geosave.geodata import GeoAnchor, stac
from geosave.geodata.stac.catalog import from_items


@pytest.fixture
def local_catalog(tmp_path):
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import BoundingBox

    grid = GeoBox.from_bbox(
        BoundingBox(500000, 5000000, 500020, 5000020, crs="EPSG:32633"), resolution=10
    )
    anchor = GeoAnchor.from_geometry(grid.extent, timespan="2025-01")
    timestamp = datetime(2025, 1, 15, tzinfo=timezone.utc)
    bounds = list(grid.geographic_extent.boundingbox)
    collection = pystac.Collection(
        id="local",
        description="Local reflectance",
        license="CC-BY-4.0",
        extent=pystac.Extent(
            pystac.SpatialExtent([bounds]),
            pystac.TemporalExtent([[timestamp, timestamp]]),
        ),
    )
    item = pystac.Item(
        id="scene",
        geometry=grid.geographic_extent.json,
        bbox=bounds,
        datetime=timestamp,
        properties={},
        collection="local",
    )
    ProjectionExtension.ext(item, add_if_missing=True).apply(
        code="EPSG:32633", shape=list(grid.shape), transform=list(grid.transform)
    )
    path = tmp_path / "reflectance.tif"
    values = np.array(
        [[[0, 2000], [2000, 2000]], [[0, 6000], [6000, 6000]]], dtype="uint16"
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=2,
        dtype="uint16",
        crs=grid.crs,
        transform=grid.transform,
        nodata=0,
    ) as destination:
        destination.write(values)
    item.add_asset(
        "reflectance",
        pystac.Asset(
            href=str(path),
            media_type=pystac.MediaType.GEOTIFF,
            roles=["data"],
            extra_fields={
                "raster:bands": [
                    {"data_type": "uint16", "nodata": 0, "scale": 0.0001, "unit": "1"},
                    {
                        "data_type": "uint16",
                        "nodata": 0,
                        "scale": 0.0002,
                        "offset": -0.1,
                        "unit": "1",
                    },
                ]
            },
        ),
    )
    path = from_items([item]).gs.to_geoparquet(
        tmp_path / "local.parquet", collections=[collection]
    )
    client = stac.StacClient.open(path)
    options = dict(
        bands=["red", "nir"],
        groupby="time",
        stac_cfg={
            "local": {"aliases": {"red": ("reflectance", 1), "nir": ("reflectance", 2)}}
        },
    )
    return client, anchor, grid, options
