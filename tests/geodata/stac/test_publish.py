"""A region and its rasters state a native STAC Item."""

from datetime import UTC, datetime

import pystac
import pytest

from geosave.geodata import GeoRegion
from geosave.geodata.stac import StacItem
from tests.geodata.conftest import build_raster

EO = "https://stac-extensions.github.io/eo/v1.1.0/schema.json"


def test_item_covers_the_region_at_its_instant():
    # Datetimes, since a date string covers its whole period.
    instant = datetime(2025, 6, 1, 10, 30)
    region = GeoRegion.from_bbox((110.0, -7.0, 111.0, -6.0), timespan=(instant, instant))

    item = StacItem.from_region(region, id="scene", properties={"platform": "sentinel-2b"})

    assert isinstance(item, pystac.Item)
    assert item.id == "scene"
    assert item.datetime == datetime(2025, 6, 1, 10, 30, tzinfo=UTC)
    assert item.bbox == [110.0, -7.0, 111.0, -6.0]
    assert item.geometry["type"] == "Polygon"
    assert item.properties == {"platform": "sentinel-2b"}
    assert item.assets == {}


def test_item_over_a_period_states_its_start_and_end():
    region = build_raster(times=2).gs.region
    given = {"eo:cloud_cover": 12.5}

    item = StacItem.from_region(region, id="scene", properties=given)

    assert item.datetime is None
    assert item.properties == {
        "eo:cloud_cover": 12.5,
        "start_datetime": "2025-06-01T00:00:00Z",
        "end_datetime": "2025-06-02T00:00:00Z",
    }
    # The footprint is the projected extent, restated in WGS84.
    assert item.bbox == pytest.approx(list(region.extent.to_crs("EPSG:4326").boundingbox))
    assert given == {"eo:cloud_cover": 12.5}


def test_timeless_region_is_refused():
    with pytest.raises(ValueError, match="timespan"):
        StacItem.from_region(build_raster().gs.region, id="scene")


def test_raster_asset_states_grid_and_bands_and_declares_extensions():
    raster = build_raster(times=1, packed=True).isel(time=0)
    raster["red"].attrs["long_name"] = "Red reflectance"
    item = StacItem.from_region(build_raster(times=1).gs.region, id="scene")
    item.stac_extensions = [EO]

    asset = item.add_raster_asset("red", raster[["red"]], "/data/scene/red.tif")

    assert item.assets["red"] is asset
    assert asset.href == "/data/scene/red.tif"
    assert asset.title == "Red reflectance"
    assert asset.media_type == pystac.MediaType.COG
    assert asset.roles == ["data"]
    assert asset.extra_fields["proj:code"] == "EPSG:32749"
    assert asset.extra_fields["proj:shape"] == [2, 2]
    (band,) = asset.extra_fields["raster:bands"]
    assert band["data_type"] == "uint16"
    assert band["scale"] == raster["red"].attrs["scale_factor"]
    # A declared extension stays, and each asset extension is declared once.
    assert item.stac_extensions[0] == EO
    assert len(item.stac_extensions) == 3

    item.add_raster_asset("nir", raster[["nir"]], "/data/scene/nir.tif")
    assert len(item.stac_extensions) == 3
