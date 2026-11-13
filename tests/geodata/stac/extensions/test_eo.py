"""A variable's name and spectral facts as its EO band, and back."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
import xarray as xr
from pystac.extensions.eo import Band

from geosave.geodata.attrs import EOBand
from geosave.geodata.stac.extensions import eo
from geosave.geodata.stac.stamps import STAMPS, LoadSource
from tests.geodata.conftest import build_raster


def test_a_spectral_variable_names_its_band() -> None:
    scene = build_raster().gs.write_spectral(
        "red", center_wavelength=0.665, target=["red"]
    )

    red = eo.build_band(scene.red.assign_attrs(long_name="Red"))
    nir = eo.build_band(scene.nir)

    assert (red.name, red.common_name, red.center_wavelength) == ("red", "red", 0.665)
    assert red.description == "Red"
    assert (nir.name, nir.common_name) == ("nir", None)


def test_spectral_facts_and_the_description_are_stamped_onto_the_variable() -> None:
    band = Band.create(
        name="B04", common_name="red", center_wavelength=0.665, description="Red"
    )

    assert eo.read_attrs(band) == {
        "common_name": "red",
        "center_wavelength": 0.665,
        "long_name": "Red",
    }


def _item(id: str = "scene", **asset_fields) -> pystac.Item:
    item = pystac.Item(
        id=id,
        geometry=None,
        bbox=None,
        datetime=datetime(2025, 1, 1, tzinfo=UTC),
        properties={},
    )
    item.add_asset(
        "red",
        pystac.Asset(
            href=f"https://example.com/{id}.tif",
            media_type=pystac.MediaType.GEOTIFF,
            extra_fields=asset_fields,
        ),
    )
    return item


def _loaded(*names: str) -> xr.Dataset:
    pixels = {}
    for name in names:
        pixels[name] = (("y", "x"), [[1]])
    return xr.Dataset(pixels)


def _coloured_item(colours: list[str | None]) -> pystac.Item:
    """One Item with an asset per colour; None states no colour."""
    item = _item()
    del item.assets["red"]
    for index, colour in enumerate(colours):
        bands = [] if colour is None else [{"name": f"b{index}", "common_name": colour}]
        item.add_asset(
            f"b{index}",
            pystac.Asset(
                href=f"https://example.com/b{index}.tif",
                media_type=pystac.MediaType.GEOTIFF,
                extra_fields={"eo:bands": bands},
            ),
        )
    return item


def test_eo_and_rgb_stamps_are_registered() -> None:
    assert STAMPS["eo"] is eo.stamp_eo
    assert STAMPS["rgb"] is eo.stamp_rgb


def test_spectral_facts_reach_the_variable() -> None:
    item = _item(
        **{
            "eo:bands": [
                {"name": "B04", "common_name": "red", "center_wavelength": 0.665}
            ]
        }
    )

    stamped = eo.stamp_eo(_loaded("red"), LoadSource([item], None, "id"))

    assert stamped.red.gs.spectral == EOBand(common_name="red", center_wavelength=0.665)


def test_eo_stamp_writes_the_description_as_long_name() -> None:
    item = _item(**{"eo:bands": [{"description": "Reflectance"}]})

    stamped = eo.stamp_eo(_loaded("red"), LoadSource([item], None, "id"))

    assert stamped.red.attrs["long_name"] == "Reflectance"


def test_eo_stamp_keeps_only_what_items_agree_on() -> None:
    first = _item("a", **{"eo:bands": [{"common_name": "red", "center_wavelength": 1}]})
    second = _item(
        "b", **{"eo:bands": [{"common_name": "red", "center_wavelength": 2}]}
    )

    stamped = eo.stamp_eo(_loaded("red"), LoadSource([first, second], None, "id"))

    assert stamped.red.attrs == {"common_name": "red"}


def test_eo_stamp_keeps_the_loaded_attrs() -> None:
    item = _item(**{"eo:bands": [{"description": "Reflectance"}]})
    loaded = _loaded("red")
    loaded["red"].attrs["long_name"] = "Mine"

    stamped = eo.stamp_eo(loaded, LoadSource([item], None, "id"))

    assert stamped.red.attrs["long_name"] == "Mine"


def _stamp_rgb(item: pystac.Item) -> xr.Dataset:
    return eo.stamp_rgb(_loaded(*item.assets), LoadSource([item], None, "id"))


def test_loading_states_the_rgb_composite_the_band_colours_name() -> None:
    stamped = _stamp_rgb(_coloured_item(["blue", "red", "green"]))

    assert stamped.gs.rgb == ("b1", "b2", "b0")


def test_loading_states_no_composite_without_a_blue_band() -> None:
    stamped = _stamp_rgb(_coloured_item([None, "red", "green"]))

    assert stamped.gs.rgb is None


def test_rgb_reads_stac_not_eo_attrs() -> None:
    item = _coloured_item(["red", "green", "blue"])

    # Only the rgb stamp runs, so no variable carries a common_name attr.
    stamped = _stamp_rgb(item)

    assert all("common_name" not in v.attrs for v in stamped.data_vars.values())
    assert stamped.gs.rgb == ("b0", "b1", "b2")
