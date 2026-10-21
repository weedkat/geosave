from __future__ import annotations

from datetime import UTC, datetime, timezone

import numpy as np
import pystac
import pytest
import rasterio
from pystac.extensions.projection import ProjectionExtension
from shapely.geometry import mapping

from geosave.geodata import GeoAnchor, io, stac
from tests.geodata.conftest import build_raster


@pytest.fixture
def local_source(tmp_path):
    anchor = GeoAnchor.from_coordinates(
        45, 12, shape=2, resolution=10, crs="EPSG:32633", timespan="2025-01"
    )
    grid = anchor.geobox
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
    path = stac.create_catalog([item]).gs.to_geoparquet(
        tmp_path / "local.parquet", collections=[collection]
    )
    source = (
        stac.StacClient.open(path)
        .source("local")
        .set_config(
            bands=["red", "nir"],
            groupby="time",
            stac_cfg={
                "local": {
                    "aliases": {"red": ("reflectance", 1), "nir": ("reflectance", 2)}
                }
            },
        )
    )
    return source, anchor


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
    saved = io.geotiff.write_cog(source, tmp_path / f"{name}.tif")
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
