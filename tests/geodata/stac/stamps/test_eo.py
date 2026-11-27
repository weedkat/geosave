"""STAC spectral facts on loaded variables."""

from geosave.geodata.attrs import EOBand
from geosave.geodata.stac.stamps import stamp_eo
from tests.geodata.stac.stamps.conftest import pixels, scene, with_asset


def test_eo_writes_the_spectral_facts() -> None:
    item = with_asset(
        scene(),
        "red",
        **{
            "eo:bands": [
                {"name": "B04", "common_name": "red", "center_wavelength": 0.665}
            ]
        },
    )

    stamped = stamp_eo(pixels("red"), [item])

    assert stamped.red.gs.spectral == EOBand(common_name="red", center_wavelength=0.665)


def test_eo_writes_the_description_as_long_name() -> None:
    item = with_asset(scene(), "red", **{"eo:bands": [{"description": "Reflectance"}]})

    stamped = stamp_eo(pixels("red"), [item])

    assert stamped.red.attrs["long_name"] == "Reflectance"


def test_eo_keeps_only_what_items_agree_on() -> None:
    first = with_asset(
        scene("a"),
        "red",
        **{"eo:bands": [{"common_name": "red", "center_wavelength": 1}]},
    )
    second = with_asset(
        scene("b"),
        "red",
        **{"eo:bands": [{"common_name": "red", "center_wavelength": 2}]},
    )

    stamped = stamp_eo(pixels("red"), [first, second])

    assert stamped.red.attrs == {"common_name": "red"}
