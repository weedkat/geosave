"""STAC asset reading preserves file pixels and applies catalog descriptions."""

from __future__ import annotations

from datetime import UTC, datetime

import dask
import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin

from geosave.geodata import (
    open_vector,
    stac,
)
from geosave.geodata.io.raster.stac import open_raster
from geosave.geodata.stac.translate import table as stac_table
from tests.geodata.conftest import _refuse, build_raster, item_table, read_items
from tests.geodata.stac.conftest import create_native_item


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


def test_native_composite_load_retains_bounds(tmp_path):
    from geosave.geodata.attrs import TimeSpec
    from geosave.geodata.io.raster.stac import open_raster

    ds = build_raster(times=1)
    ds.time.attrs.update(TimeSpec.from_resample("MS").to_attrs())
    original = next(stac.create_items(ds.gs.to_cog(tmp_path / "original.tif")))
    loaded = open_raster([original])
    rebuilt = next(stac.create_items(loaded.gs.to_cog(tmp_path / "rebuilt.tif")))
    assert rebuilt.datetime == original.datetime
    assert rebuilt.properties["start_datetime"] == original.properties["start_datetime"]
    assert rebuilt.properties["end_datetime"] == original.properties["end_datetime"]


@pytest.mark.integration
def test_an_item_table_of_cogs_loads_back(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)
    paths = source.gs.to_cog(
        f"{prefix}/forest_{{time:%Y%m%d}}.tif", storage_options=options
    )
    # GDAL reads a remote TIFF with the keys in its environment, not fsspec options.
    rows = item_table(paths)

    saved = rows.gs.to_geoparquet(f"{prefix}/catalog.parquet", storage_options=options)

    rows = open_vector(saved, storage_options=options)
    assert rows.iloc[0]["assets"]["image"]["href"] == str(paths[0])
    with dask.config.set(scheduler=_refuse):
        restored = read_items(rows)
    np.testing.assert_array_equal(restored.red.values, source.red.values)
    restored.close()


def test_only_data_assets_load(tmp_path):
    entry, source = create_native_item(tmp_path)

    loaded = read_items(stac_table.from_items([entry]))

    # The thumbnail is skipped; the data asset names its bands by its key.
    assert list(loaded.data_vars) == [
        f"reflectance.{index}" for index in range(1, len(source.data_vars) + 1)
    ]
    loaded.close()
