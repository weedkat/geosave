"""Selected STAC Items load independently of discovery."""

import dask.array as da
import numpy as np
import pytest
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox

from geosave.geodata import io, stac
from tests.conftest import build_dataset
from tests.geodata.conftest import create_file_items


@pytest.fixture
def observation(tmp_path):
    grid = GeoBox.from_bbox(
        BoundingBox(500000, 5000000, 500040, 5000040, crs="EPSG:32633"), resolution=10
    )
    raster = build_dataset(
        {"red": (("y", "x"), np.arange(16, dtype="uint16").reshape(4, 4))}, grid
    )
    raster = raster.assign_coords(time=np.datetime64("2025-01-15"))
    path = io.formats.geotiff.write_cog(raster, tmp_path / "red.tif")
    item = next(create_file_items({"red": path}))
    item.collection_id = "local"
    item.properties["eo:cloud_cover"] = 20.0
    return item, raster


def test_properties_follow_fused_time_and_keep_common_attrs(observation):
    item, raster = observation
    other = item.clone()
    other.id = "second"
    other.properties["eo:cloud_cover"] = 40.0
    item.properties["platform"] = other.properties["platform"] = "test"
    loaded = stac.load_dataset(
        [item, other],
        geobox=raster.gs.geobox,
        groupby="time",
        with_properties=["eo:cloud_cover"],
    )
    assert loaded.gs.stac_properties["eo:cloud_cover"].tolist() == [30.0]
    assert loaded.gs.stac_common_properties == {
        "platform": "test",
        "eo:cloud_cover": 30.0,
    }
    assert "eo:cloud_cover" not in loaded.data_vars


def test_loader_rejects_mixed_collections_before_reading(observation):
    item, raster = observation
    other = item.clone()
    other.collection_id = "other"
    with pytest.raises(ValueError, match="one collection"):
        stac.load_dataset([item, other], geobox=raster.gs.geobox)


def test_tree_loads_each_asset_on_its_stated_grid(observation, tmp_path):
    item, raster = observation
    item.properties["platform"] = "test"
    coarse = build_dataset(
        {"swir": (("y", "x"), np.full((2, 2), 7, dtype="uint16"))},
        raster.gs.geobox.zoom_out(2),
    )
    coarse = coarse.assign_coords(time=np.datetime64("2025-01-15"))
    path = io.formats.geotiff.write_cog(coarse, tmp_path / "swir.tif")
    second = next(create_file_items({"swir": path}))
    for key, asset in second.assets.items():
        item.add_asset(key, asset)
    tree = stac.load_datatree([item])
    # One node per STAC asset, each on the grid its proj:* states.
    assert set(tree.gs.datasets) == {"local/red", "local/swir"}
    assert tree["local/red"].gs.geobox == raster.gs.geobox
    assert tree["local/swir"].gs.geobox == coarse.gs.geobox
    assert tree["local"].to_dataset().gs.stac_common_properties["platform"] == "test"
    assert "time" in tree["local"].to_dataset(inherit=False).coords

    grouped = tree.gs.group_grid()
    assert set(grouped.gs.datasets) == {"local/r10m", "local/r20m"}
    assert grouped["local/r10m"].gs.geobox == raster.gs.geobox
    assert grouped["local/r20m"].gs.geobox == coarse.gs.geobox
    np.testing.assert_array_equal(tree["local/swir"].swir.values[0], coarse.swir.values)


def test_signing_happens_at_load_without_mutating_items(observation):
    item, raster = observation
    original = item.assets["red"].href
    item.assets["red"].href = "https://unsigned.test/red.tif"
    urls = []

    def sign(url):
        urls.append(url)
        return original

    loaded = stac.load_dataset([item], geobox=raster.gs.geobox, patch_url=sign)
    np.testing.assert_array_equal(loaded.red.values[0], raster.red.values)
    assert urls == ["https://unsigned.test/red.tif"]
    assert item.assets["red"].href == "https://unsigned.test/red.tif"


def test_same_date_items_mosaic_by_default_and_stay_apart_by_id(observation):
    entry, raster = observation
    other = entry.clone()
    other.id = "second"
    mosaic = stac.load_dataset([entry, other], geobox=raster.gs.geobox)
    assert mosaic.sizes["time"] == 1
    separate = stac.load_dataset([entry, other], geobox=raster.gs.geobox, groupby="id")
    assert separate.sizes["time"] == 2


def _period_item(observation, ident, start, end):
    """Clone the observation as a composite Item covering [start, end)."""
    entry, _ = observation
    period = entry.clone()
    period.id = ident
    period.datetime = None
    period.properties["datetime"] = None
    period.properties.update(start_datetime=start, end_datetime=end)
    return period


def _months(observation):
    june = _period_item(
        observation, "june", "2025-06-01T00:00:00Z", "2025-07-01T00:00:00Z"
    )
    july = _period_item(
        observation, "july", "2025-07-01T00:00:00Z", "2025-08-01T00:00:00Z"
    )
    return june, july


def _days(values):
    return np.array(values, dtype="datetime64[ns]")


def test_composite_items_keep_their_own_periods_in_any_order(observation):
    _, raster = observation
    june, july = _months(observation)

    loaded = stac.load_dataset([july, june], geobox=raster.gs.geobox, groupby="id")

    for label, start in zip(
        loaded.time.values, loaded.start_datetime.values, strict=True
    ):
        assert start == label  # ODC labels a composite at its start
    ends = {
        str(start)[:10]: str(end)[:10]
        for start, end in zip(
            loaded.start_datetime.values, loaded.end_datetime.values, strict=True
        )
    }
    assert ends == {"2025-06-01": "2025-07-01", "2025-07-01": "2025-08-01"}


def test_scene_labels_state_no_period(observation):
    entry, raster = observation
    loaded = stac.load_dataset([entry], geobox=raster.gs.geobox, groupby="id")
    assert "start_datetime" not in loaded.coords


def test_a_solar_day_label_covers_its_local_day(tmp_path):
    from odc.geo.geom import BoundingBox as Box

    # Jakarta, UTC+7: 03:12 and 03:13 UTC share local Jan 10; 18:00 UTC is Jan 11.
    grid = GeoBox.from_bbox(
        Box(106.8, -6.3, 106.81, -6.29, crs="EPSG:4326"), resolution=0.0025
    )
    items = []
    for name, when in (
        ("a", "2025-01-10T03:12"),
        ("b", "2025-01-10T03:13"),
        ("c", "2025-01-10T18:00"),
    ):
        raster = build_dataset(
            {"red": (("y", "x"), np.ones(grid.shape, "uint16"))}, grid
        ).assign_coords(time=np.datetime64(when))
        path = io.formats.geotiff.write_cog(raster, tmp_path / f"{name}.tif")
        entry = next(create_file_items({"red": path}))
        entry.id = name
        items.append(entry)

    loaded = stac.load_dataset(items, grid, groupby="solar_day")

    np.testing.assert_array_equal(
        loaded.start_datetime.values,
        _days(["2025-01-09T17:00", "2025-01-10T17:00"]),
    )
    np.testing.assert_array_equal(
        loaded.end_datetime.values,
        _days(["2025-01-10T17:00", "2025-01-11T17:00"]),
    )


def test_a_composite_keeps_its_own_period_under_solar_day(observation):
    _, raster = observation
    june, _ = _months(observation)

    loaded = stac.load_dataset([june], geobox=raster.gs.geobox, groupby="solar_day")

    np.testing.assert_array_equal(loaded.end_datetime.values, _days(["2025-07-01"]))


def test_a_multi_band_asset_holds_one_variable_per_band(tmp_path):
    grid = GeoBox.from_bbox(
        BoundingBox(500000, 5000000, 500040, 5000040, crs="EPSG:32633"), resolution=10
    )
    pixels = {}
    for value, name in enumerate(["red", "green", "blue"]):
        pixels[name] = (("y", "x"), np.full(grid.shape, value, "uint16"))
    rgb = build_dataset(pixels, grid).assign_coords(time=np.datetime64("2025-01-15"))
    path = io.formats.geotiff.write_cog(rgb, tmp_path / "visual.tif")
    item = next(create_file_items({"visual": path}))
    item.collection_id = "local"

    tree = stac.load_datatree([item])

    assert list(tree["local/visual"].data_vars) == ["red", "green", "blue"]
    np.testing.assert_array_equal(tree["local/visual"].blue.values[0], 2)


def test_items_disagreeing_on_an_asset_grid_are_refused(observation, tmp_path):
    item, raster = observation
    shifted = build_dataset(
        {"red": (("y", "x"), np.ones((4, 4), "uint16"))},
        GeoBox.from_bbox(
            BoundingBox(500040, 5000000, 500080, 5000040, crs="EPSG:32633"),
            resolution=10,
        ),
    ).assign_coords(time=np.datetime64("2025-01-16"))
    other = next(
        create_file_items(
            {"red": io.formats.geotiff.write_cog(shifted, tmp_path / "east.tif")}
        )
    )
    other.collection_id = item.collection_id

    with pytest.raises(ValueError, match="'red'.*grid"):
        stac.load_datatree([item, other])


def test_a_catalog_loads_one_collection_onto_a_chosen_grid(observation):
    from geosave.geodata.stac.catalog import create_catalog

    item, raster = observation
    grid = raster.gs.geobox.zoom_out(2)

    loaded = create_catalog([item]).gs.to_dataset(grid)

    assert loaded.gs.geobox == grid
    assert isinstance(loaded.red.data, da.Array)


def test_a_catalog_of_two_collections_does_not_load_as_one_dataset(observation):
    from geosave.geodata.stac.catalog import create_catalog

    item, raster = observation
    other = item.clone()
    other.id = "dem"
    other.collection_id = "dem"

    with pytest.raises(ValueError, match="one collection"):
        create_catalog([item, other]).gs.to_dataset(raster.gs.geobox)
