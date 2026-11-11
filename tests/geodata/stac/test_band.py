from __future__ import annotations

import pystac


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
    assert band.read_aliases(asset) == [("red", "red"), ("nir", "nir")]
    assert band.read_attrs(asset, 0)["scale_factor"] == 0.01
