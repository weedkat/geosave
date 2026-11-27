"""Sentinel-1 RTC ingestion loads backscatter under the provider's naming."""

from datetime import datetime

import dask
import numpy as np
import pystac
import pytest
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_origin

from geosave.geodata import GeoRegion
from geosave.ml.pipeline.ingestion.sentinel1_rtc import (
    NODATA,
    Provider,
    Sentinel1RTCIngestor,
)

GRID = GeoBox((4, 4), from_origin(500000, 10000, 10, 10), "EPSG:32649")


def write_item(tmp_path, day: datetime = datetime(2024, 12, 31)) -> pystac.Item:
    """Write VV and VH backscatter, one pixel of it fill, as an interval Item."""
    source = Provider.planetary_computer.value
    footprint = GRID.extent.to_crs("EPSG:4326").geom
    item = pystac.Item(
        f"rtc-{day:%Y%m%d}",
        footprint.__geo_interface__,
        list(footprint.bounds),
        day,
        {
            "start_datetime": f"{day:%Y-%m-%d}T22:10:00Z",
            "end_datetime": f"{day:%Y-%m-%d}T22:10:25Z",
            "sat:orbit_state": "ascending",
        },
        collection=source.collection,
    )
    for band, power in [("VV", 0.25), ("VH", 0.04)]:
        pixels = np.full((4, 4), power, dtype="float32")
        pixels[0, 0] = NODATA
        path = tmp_path / f"{item.id}-{band}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            count=1,
            dtype="float32",
            height=4,
            width=4,
            crs="EPSG:32649",
            transform=GRID.transform,
        ) as dst:
            dst.write(pixels, 1)
        item.add_asset(
            source.assets[band],
            pystac.Asset(str(path), media_type=pystac.MediaType.GEOTIFF),
        )
    return item


def refuse(*args, **kwargs):
    raise AssertionError("loading must stay lazy")


def test_backscatter_loads_lazily_into_the_configured_tree(tmp_path):
    ingestor = Sentinel1RTCIngestor(
        "s1", ["VV", "VH"], groups={"sar": ["VV", "VH"]}, chunks=2
    )
    item = write_item(tmp_path)

    with dask.config.set(scheduler=refuse):
        tree = ingestor.load([item], GRID)

    assert tree.groups == ("/", "/s1", "/s1/sar")
    sar = tree["s1/sar"].to_dataset()
    assert list(sar.data_vars) == ["VV", "VH"]
    assert sar.odc.geobox == GRID
    assert sar.VV.dtype == np.float32
    assert sar.VV.chunks == ((1,), (2, 2), (2, 2))
    # A 25-second acquisition is labelled by its instant.
    assert sar.gs.timespan == (datetime(2024, 12, 31), datetime(2024, 12, 31))

    decoded = sar.gs.mask_and_scale().isel(time=0)
    assert np.isnan(decoded.VV.values[0, 0])
    np.testing.assert_allclose(decoded.VH.values[1:, 1:], 0.04)


def test_a_region_loads_one_step_per_day(tmp_path):
    ingestor = Sentinel1RTCIngestor("s1", ["VV"], groups={"sar": ["VV"]})
    items = [
        write_item(tmp_path, datetime(2024, 12, 31)),
        write_item(tmp_path, datetime(2025, 1, 12)),
    ]
    ingestor.search = lambda region: items
    region = GeoRegion.from_geometry(GRID.extent, timespan=("2024-12-31", "2025-01-12"))

    sar = ingestor(region)["s1/sar"].to_dataset()

    assert sar.time.dt.strftime("%Y-%m-%d").values.tolist() == [
        "2024-12-31",
        "2025-01-12",
    ]
    assert sar.odc.crs == "EPSG:32649"


def test_bands_outside_the_polarizations_are_refused():
    with pytest.raises(ValueError, match="not a Sentinel-1 polarization"):
        Sentinel1RTCIngestor("s1", ["VV", "B04"])


def test_no_items_are_refused():
    ingestor = Sentinel1RTCIngestor("s1", ["VV"], groups={"sar": ["VV"]})

    with pytest.raises(ValueError, match="no Sentinel-1 RTC acquisition"):
        ingestor.load([], GRID)


@pytest.mark.integration
def test_planetary_computer_ingests_ascending_passes():
    region = GeoRegion.from_bbox(
        (5.10, 52.05, 5.13, 52.08), timespan=("2023-07-01", "2023-07-31")
    )
    ingestor = Sentinel1RTCIngestor(
        "s1",
        ["VV", "VH"],
        groups={"sar": ["VV", "VH"]},
        filter="sat:orbit_state = 'ascending'",
    )

    sar = ingestor(region)["s1/sar"].to_dataset()

    assert sar.sizes["time"] == 5
    assert sar.VV.dtype == np.float32
    scene = sar.isel(time=0).gs.mask_and_scale().compute()
    median = float(scene.VV.median())
    assert 0.01 < median < 1.0
