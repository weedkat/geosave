"""A region and its rasters state a native STAC Item."""

from datetime import UTC, datetime

import dask
import numpy as np
import xarray as xr
import pystac
import pytest

from geosave.geodata import GeoRegion, attrs, stac, open_raster
from geosave.geodata.stac import StacItem
from tests.geodata.conftest import build_raster, create_test_collection, _refuse

EO = "https://stac-extensions.github.io/eo/v1.1.0/schema.json"


def test_item_covers_the_region_at_its_instant():
    # Datetimes, since a date string covers its whole period.
    instant = datetime(2025, 6, 1, 10, 30)
    region = GeoRegion.from_bbox(
        (110.0, -7.0, 111.0, -6.0), timespan=(instant, instant)
    )

    item = StacItem.from_region(
        region, id="scene", properties={"platform": "sentinel-2b"}
    )

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
    assert item.bbox == pytest.approx(
        list(region.extent.to_crs("EPSG:4326").boundingbox)
    )
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


def build_grid(shape, names, dtype):
    """Build two dated observations of `names` over one 40 m extent."""
    import numpy as np
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    geobox = GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_040.0, 9_000_040.0),
        crs="EPSG:32749",
        shape=shape,
        tight=True,
    )
    coords = dict(xr_coords(geobox, always_yx=True))
    coords["time"] = ("time", np.array(["2025-06-01", "2025-06-02"], "datetime64[ns]"))
    block = np.arange(2 * shape[0] * shape[1], dtype=dtype).reshape(2, *shape) + 1
    variables = {}
    for name in names:
        variables[name] = (("time", "y", "x"), block, {"_FillValue": 0})
    raster = xr.Dataset(variables, coords=coords).chunk({"time": 1})
    raster = raster.gs.write_stac_property("eo:cloud_cover", [24.001125, 40.0])
    raster = raster.gs.write_stac_property("platform", "sentinel-2b")
    collection = create_test_collection("optical", description="Optical")
    return attrs.stamp_attrs(
        raster, attrs.StacCollection(collection=collection.to_dict())
    )


def build_two_grids():
    """Stage `red`/`nir` at 10 m and `scl` at 20 m over one extent and two dates."""
    fine = build_grid((4, 4), ["red", "nir"], "uint16")
    for name in fine.data_vars:
        fine[name].attrs.update(scale_factor=0.0001, add_offset=0.0)
    coarse = build_grid((2, 2), ["scl"], "uint8")
    return fine, coarse


def publish_two_grids(root):
    fine, coarse = build_two_grids()
    items = []
    for position, name in enumerate(["20250601", "20250602"]):
        image, scl = fine.isel(time=position), coarse.isel(time=position)
        item = stac.StacItem.from_region(
            image.gs.region,
            id=name,
            properties={
                "eo:cloud_cover": image[
                    image.gs.stac_properties["eo:cloud_cover"]
                ].item(),
                "platform": "sentinel-2b",
            },
        )
        item.set_collection(fine.gs.stac_collection)
        assets = {key: image[[key]] for key in image.data_vars}
        assets["scl"] = scl
        items.append(stac.publish_item(item, assets=assets, root=root))
    return items, fine, coarse


def test_publish_multiband_image_and_classification_preserves_native_item(tmp_path):
    fine, coarse = build_two_grids()
    image = fine.isel(time=0)
    scl = coarse.isel(time=0)
    scl.scl.attrs.update(flag_values=[1, 2], flag_meanings="land water")
    template = stac.StacItem.from_region(image.gs.region, id="scene")
    item = pystac.Item.from_dict(template.to_dict())
    before = item.to_dict()
    result = stac.publish_item(item, assets={"image": image, "scl": scl}, root=tmp_path)
    assert item.to_dict() == before
    assert list(result.assets) == ["image", "scl"]
    assert [
        band["name"] for band in result.assets["image"].extra_fields["eo:bands"]
    ] == ["red", "nir"]
    classes = result.assets["scl"].extra_fields["raster:bands"][0][
        "classification:classes"
    ]
    assert [entry["name"] for entry in classes] == ["land", "water"]
    for key, source in {"image": image, "scl": scl}.items():
        restored = open_raster(result.assets[key].href, mask_and_scale=False)
        assert restored.gs.geobox == source.gs.geobox
        for name in source.data_vars:
            np.testing.assert_array_equal(
                restored[name].values.squeeze(), source[name].values
            )
            assert restored[name].dtype == source[name].dtype
    loaded = stac.load_native_datasets([result], chunks={})
    assert len(loaded) == 2


@pytest.mark.parametrize(
    "invalid",
    ["timeless", "dimension", "time", "encoding", "dtype", "meaning", "existing"],
)
def test_publish_preflights_all_assets_without_pixels_or_files(tmp_path, invalid):
    valid = build_raster(times=1).isel(time=0).chunk()
    other = valid.copy(deep=True)
    item = stac.StacItem.from_region(valid.gs.region, id="scene")
    if invalid == "timeless":
        other = other.drop_vars("time")
    elif invalid == "dimension":
        other = other.expand_dims(time=[valid.time.values])
    elif invalid == "time":
        other = other.assign_coords(time=np.datetime64("2030-01-01"))
    elif invalid == "encoding":
        other.red.encoding["scale_factor"] = 0.1
    elif invalid == "dtype":
        other["red"] = other.red.astype("float32")
    elif invalid == "meaning":
        other.red.attrs["units"] = "m"
        other.nir.attrs["units"] = "K"
    else:
        item.add_asset("bad", pystac.Asset("existing.tif"))
    with dask.config.set(scheduler=_refuse), pytest.raises(ValueError):
        stac.publish_item(item, assets={"good": valid, "bad": other}, root=tmp_path)
    assert not list(tmp_path.rglob("*.tif"))


def test_split_repeated_labels_publishes_distinct_explicit_items(tmp_path):
    source = build_raster(times=2).chunk()
    source = source.assign_coords(time=source.time.isel(time=[0, 0]).values)
    slices = source.gs.to_tree("s2")["s2"].gs.split(
        dim="time", names=["first", "second"]
    )
    items = []
    for key, leaf in slices["time"].children.items():
        partition = leaf.gs.group({"image": ["red"], "nir": ["nir"]})
        raster = leaf.gs.to_dataset()
        item = stac.StacItem.from_region(raster.gs.region, id=f"acquisition-{key}")
        items.append(
            stac.publish_item(
                item,
                assets={
                    name: node.gs.to_dataset()
                    for name, node in partition.children.items()
                },
                root=tmp_path,
            )
        )
    assert [item.id for item in items] == ["acquisition-first", "acquisition-second"]
    assert items[0].datetime == items[1].datetime


def test_publish_refuses_existing_destination_before_any_new_write(tmp_path):
    source = build_raster(times=1).isel(time=0)
    item = stac.StacItem.from_region(source.gs.region, id="scene")
    stac.publish_item(item, assets={"old": source}, root=tmp_path)
    with pytest.raises(FileExistsError):
        stac.publish_item(item, assets={"new": source, "old": source}, root=tmp_path)
    assert not (tmp_path / "scene/new.tif").exists()


def test_publish_rejects_incompatible_missing_value_recipes(tmp_path):
    source = build_raster(times=1).isel(time=0).astype("float32").chunk()
    source.red.encoding = {"dtype": "uint16", "scale_factor": 0.1, "missing_value": 99}
    source.nir.encoding = {"dtype": "uint16", "scale_factor": 0.1, "missing_value": 98}
    item = stac.StacItem.from_region(source.gs.region, id="scene")
    with (
        dask.config.set(scheduler=_refuse),
        pytest.raises(ValueError, match="encoding"),
    ):
        stac.publish_item(item, assets={"image": source}, root=tmp_path)
    assert not list(tmp_path.rglob("*.tif"))


def test_publish_decoded_pixels_describes_actual_encoded_cog(tmp_path):
    import rasterio

    packed = build_raster(times=1, packed=True).isel(time=0)
    decoded = packed.gs.mask_and_scale().chunk()
    item = stac.StacItem.from_region(decoded.gs.region, id="scene")
    result = stac.publish_item(item, assets={"image": decoded}, root=tmp_path)
    asset = result.assets["image"]
    with rasterio.open(asset.href) as stored:
        for index, key in enumerate(packed.data_vars):
            band = asset.extra_fields["raster:bands"][index]
            assert stored.dtypes[index] == band["data_type"] == "uint16"
            assert stored.scales[index] == pytest.approx(band["scale"])
            assert stored.nodatavals[index] == band["nodata"]
            np.testing.assert_array_equal(stored.read(index + 1), packed[key].values)


def test_publish_observation_period_must_agree_with_item(tmp_path):
    source = build_raster(times=1).gs.write_time_bounds("D").isel(time=0)
    item = stac.StacItem.from_region(source.gs.region, id="daily")
    result = stac.publish_item(item, assets={"image": source}, root=tmp_path)
    assert result.properties["start_datetime"] == item.properties["start_datetime"]
    wrong = source.assign_coords(
        end_datetime=source.end_datetime.values + np.timedelta64(1, "D")
    )
    with pytest.raises(ValueError, match="period"):
        stac.publish_item(item, assets={"wrong": wrong}, root=tmp_path)
    assert not (tmp_path / "daily/wrong.tif").exists()
