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
