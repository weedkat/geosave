"""Band lookup, STAC field precedence and nonconflicting declarations."""

import pytest

from geosave.geodata import stac
from geosave.geodata.stac.stamps import stamp_eo, stamp_raster
from geosave.geodata.stac.stamps.bands import read_band_fields
from tests.geodata.stac.stamps.conftest import pixels, scene, with_asset


def item_with_asset(key, fields):
    return with_asset(scene(), key, **fields)


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
    fields = read_band_fields(item.assets["B04_10m"], 0)
    assert fields["raster:scale"] == 0.0001
    assert fields["raster:offset"] == -0.1
    assert fields["nodata"] == 0
    assert fields["eo:common_name"] == "red"
    assert "bands" not in fields


def test_a_band_entry_overrides_the_asset():
    item = item_with_asset(
        "B04", {"raster:scale": 1, "bands": [{"name": "B04", "raster:scale": 2}]}
    )
    assert read_band_fields(item.assets["B04"], 0)["raster:scale"] == 2


def test_raster_v1_list_is_read_by_index():
    item = item_with_asset(
        "visual",
        {
            "raster:bands": [{"scale": 1.0}, {"scale": 2.0}, {"scale": 3.0}],
            "eo:bands": [{"name": "B04"}, {"name": "B03"}, {"name": "B02"}],
        },
    )
    fields = read_band_fields(item.assets["visual"], 1)
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
    stamped = stamp_raster(pixels("red"), [item])
    assert stamped.red.attrs["scale_factor"] == 0.5


def test_band_stamps_accept_the_loaders_alias_configuration(local_catalog):
    client, region, grid, _ = local_catalog
    items = client.search_items(
        stac.StacQuery(collections=["local"]).set_region(region)
    )
    # The config points the published red alias at the second (NIR) asset band.
    cfg = {"local": {"aliases": {"red": ("reflectance", 2)}}}
    loaded = stac.load_dataset(
        items,
        bands=["red"],
        geobox=grid,
        chunks={},
        stac_cfg=cfg,
        stamps=("raster", "eo"),
    )

    assert loaded.red.attrs["scale_factor"] == 0.0002
    assert loaded.red.attrs["common_name"] == "nir"
    assert loaded.gs.mask_and_scale().red.values[0, 1, 1] == pytest.approx(1.1)


def test_unknown_band_keeps_its_attrs():
    item = item_with_asset("B04", {"raster:scale": 0.5})
    assert stamp_raster(pixels("missing"), [item]).missing.attrs == {}


def test_band_attrs_keep_what_items_agree_on_and_the_loaded_ones() -> None:
    items = [
        with_asset(
            scene(id),
            "red",
            **{
                "eo:bands": [
                    {
                        "common_name": "red",
                        "center_wavelength": wave,
                        "description": "Reflectance",
                    }
                ]
            },
        )
        for id, wave in [("a", 0.665), ("b", 0.675)]
    ]
    loaded = pixels("red")
    loaded["red"].attrs["long_name"] = "Custom description"

    stamped = stamp_eo(loaded, items)

    assert stamped.red.attrs == {
        "common_name": "red",
        "long_name": "Custom description",
    }
