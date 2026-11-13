from __future__ import annotations

import pystac
from pystac.extensions.eo import EOExtension
from geosave.geodata import attrs
from geosave.geodata.stac import band


def test_core_band_fields_are_read():
    from geosave.geodata.stac import band

    asset = pystac.Asset(
        "/tmp/multi.tif",
        extra_fields={
            "bands": [
                {"name": "red", "eo:common_name": "red", "raster:scale": 0.01},
                {"name": "nir", "eo:common_name": "nir"},
            ]
        },
    )
    assert band.read_attrs(asset, 0)["scale_factor"] == 0.01


def test_attrs_carry_band_facts_by_variable_name(tmp_path) -> None:
    from datetime import UTC, datetime

    from pystac.extensions.eo import Band

    item = pystac.Item(
        "scene",
        {"type": "Point", "coordinates": [0.0, 0.0]},
        [0.0, 0.0, 0.0, 0.0],
        datetime(2019, 1, 30, tzinfo=UTC),
        {},
    )
    EOExtension.add_to(item)
    listed = {
        "B04": [Band.create(name="B04", common_name="red", center_wavelength=0.665)],
        "visual": [Band.create(name=n) for n in ("B04", "B03", "B02")],
    }
    for key, bands in listed.items():
        item.add_asset(key, pystac.Asset(f"{tmp_path}/{key}.tif", roles=["data"]))
        EOExtension.ext(item.assets[key]).bands = bands

    red = band.read_attrs(item.assets["B04"], 0)
    visual = band.read_attrs(item.assets["visual"], 0)
    assert attrs.EOBand.from_attrs(red).common_name == "red"
    assert visual == {}


def item_with_asset(key, fields):
    from datetime import UTC, datetime

    item = pystac.Item(
        "scene",
        {"type": "Point", "coordinates": [0.0, 0.0]},
        [0.0, 0.0, 0.0, 0.0],
        datetime(2019, 1, 30, tzinfo=UTC),
        {},
    )
    asset = pystac.Asset(f"/tmp/{key}.tif", roles=["data"], extra_fields=fields)
    item.add_asset(key, asset)
    return item


def test_asset_level_raster_fields_apply_to_its_band():
    item = item_with_asset(
        "B04_10m",
        {
            "raster:scale": 0.0001,
            "raster:offset": -0.1,
            "nodata": 0,
            "bands": [{"name": "B04", "eo:common_name": "red"}],
        },
    )
    fields = band.read_band(item, "B04_10m")
    assert fields["raster:scale"] == 0.0001
    assert fields["raster:offset"] == -0.1
    assert fields["nodata"] == 0
    assert fields["eo:common_name"] == "red"
    assert "bands" not in fields


def test_a_band_entry_overrides_the_asset():
    item = item_with_asset(
        "B04", {"raster:scale": 1, "bands": [{"name": "B04", "raster:scale": 2}]}
    )
    assert band.read_band(item, "B04")["raster:scale"] == 2


def test_raster_v1_list_is_read_by_index():
    item = item_with_asset(
        "visual",
        {
            "raster:bands": [{"scale": 1.0}, {"scale": 2.0}, {"scale": 3.0}],
            "eo:bands": [{"name": "B04"}, {"name": "B03"}, {"name": "B02"}],
        },
    )
    fields = band.read_band(item, "visual.2")
    assert fields["scale"] == 2.0
    assert fields["name"] == "B03"


def test_common_name_alias_finds_the_asset():
    item = item_with_asset(
        "B04",
        {
            "eo:bands": [{"name": "B04", "common_name": "red"}],
            "raster:scale": 0.5,
        },
    )
    # ODC reads aliases only from Items that declare the EO extension.
    item.stac_extensions.append(
        "https://stac-extensions.github.io/eo/v1.1.0/schema.json"
    )
    fields = band.read_band(item, "red")
    assert fields["raster:scale"] == 0.5
    assert fields["common_name"] == "red"


def test_unknown_band_returns_empty():
    item = item_with_asset("B04", {"raster:scale": 0.5})
    assert band.read_band(item, "missing") == {}
