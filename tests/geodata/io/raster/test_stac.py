"""STAC asset reading preserves file pixels and applies catalog descriptions."""

from datetime import UTC, datetime

import dask
import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin

from geosave.geodata.io.raster.stac import open_raster


@pytest.fixture
def item(tmp_path):
    path = tmp_path / "image.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32749",
        transform=from_origin(500_000, 9_000_020, 10, 10),
        nodata=0,
    ) as tiff:
        tiff.write(np.stack([np.full((2, 2), value, "uint8") for value in (1, 2, 3)]))

    entry = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    entry.add_asset("visual", pystac.Asset(str(path), roles=["data"]))
    return entry


def test_unlisted_bands_use_the_asset_key_without_reading_pixels(item):
    original = item.to_dict()

    raster = open_raster([item])

    assert list(raster.data_vars) == ["visual.1", "visual.2", "visual.3"]
    assert dask.is_dask_collection(raster["visual.3"])
    assert raster["visual.3"].gs.nodata == 0
    assert int(raster["visual.3"].max()) == 3
    assert item.to_dict() == original


def test_band_count_mismatch_reports_the_file(item):
    item.assets["visual"].extra_fields["bands"] = [{"name": "red"}, {"name": "green"}]

    with pytest.raises(ValueError, match=r"image.tif holds 3 bands.*names 2"):
        open_raster([item])
