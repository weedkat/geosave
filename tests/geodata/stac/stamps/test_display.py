"""Display titles and RGB composites."""

import pystac

from geosave.geodata.stac.stamps import stamp_rgb, stamp_title
from tests.geodata.stac.stamps.conftest import pixels, scene, with_asset


def test_title_names_a_one_band_asset() -> None:
    item = with_asset(
        scene(),
        "red",
        title="Red reflectance",
        **{"eo:bands": [{"common_name": "red"}]},
    )

    stamped = stamp_title(pixels("red"), [item])

    assert stamped.red.attrs["long_name"] == "Red reflectance"


def test_title_skips_multi_band_assets() -> None:
    item = with_asset(scene(), "red", title="Red", **{"raster:bands": [{}, {}]})

    stamped = stamp_title(pixels("red.1", "red.2"), [item])

    assert "long_name" not in stamped["red.1"].attrs
    assert "long_name" not in stamped["red.2"].attrs


def test_title_yields_to_the_band_description() -> None:
    item = with_asset(
        scene(), "red", title="Red", **{"eo:bands": [{"description": "Reflectance"}]}
    )

    stamped = stamp_title(pixels("red"), [item])

    assert "long_name" not in stamped.red.attrs


def coloured(colours: list[str | None]) -> tuple[pystac.Item, list[str]]:
    """Return an Item with one asset per colour, None stating no colour."""
    item = scene()
    names = []
    for index, colour in enumerate(colours):
        bands = []
        if colour is not None:
            bands = [{"name": f"b{index}", "common_name": colour}]
        with_asset(item, f"b{index}", **{"eo:bands": bands})
        names.append(f"b{index}")
    return item, names


def test_rgb_states_the_composite_the_colours_name() -> None:
    item, names = coloured(["blue", "red", "green"])

    stamped = stamp_rgb(pixels(*names), [item])

    assert stamped.gs.rgb == ("b1", "b2", "b0")


def test_rgb_states_nothing_without_a_blue_band() -> None:
    item, names = coloured([None, "red", "green"])

    stamped = stamp_rgb(pixels(*names), [item])

    assert stamped.gs.rgb is None


def test_rgb_reads_stac_not_attrs_another_stamp_wrote() -> None:
    item, names = coloured(["red", "green", "blue"])

    stamped = stamp_rgb(pixels(*names), [item])

    for variable in stamped.data_vars.values():
        assert "common_name" not in variable.attrs
    assert stamped.gs.rgb == ("b0", "b1", "b2")
