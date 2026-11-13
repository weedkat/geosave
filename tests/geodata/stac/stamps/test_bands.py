"""Band stamps: packing, spectrum, legend, title and the RGB composite."""

from datetime import UTC, datetime

import dask.array as da
import pystac
import pytest
import xarray as xr

from geosave.geodata import attrs
from geosave.geodata.attrs import EOBand
from geosave.geodata.stac.stamps import LoadSource
from geosave.geodata.stac.stamps.bands import (
    read_band,
    stamp_classification,
    stamp_eo,
    stamp_raster,
    stamp_rgb,
    stamp_title,
    write_band_attrs,
)
from tests.geodata.conftest import build_raster, raster_item
from tests.geodata.stac.stamps.conftest import pixels, scene, with_asset


# reading a band


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
    fields = read_band(item, "B04_10m")
    assert fields["raster:scale"] == 0.0001
    assert fields["raster:offset"] == -0.1
    assert fields["nodata"] == 0
    assert fields["eo:common_name"] == "red"
    assert "bands" not in fields


def test_a_band_entry_overrides_the_asset():
    item = item_with_asset(
        "B04", {"raster:scale": 1, "bands": [{"name": "B04", "raster:scale": 2}]}
    )
    assert read_band(item, "B04")["raster:scale"] == 2


def test_raster_v1_list_is_read_by_index():
    item = item_with_asset(
        "visual",
        {
            "raster:bands": [{"scale": 1.0}, {"scale": 2.0}, {"scale": 3.0}],
            "eo:bands": [{"name": "B04"}, {"name": "B03"}, {"name": "B02"}],
        },
    )
    fields = read_band(item, "visual.2")
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
    fields = read_band(item, "red")
    assert fields["raster:scale"] == 0.5
    assert fields["common_name"] == "red"


def test_unknown_band_returns_empty():
    item = item_with_asset("B04", {"raster:scale": 0.5})
    assert read_band(item, "missing") == {}


def test_band_attrs_keep_what_items_agree_on_and_the_loaded_ones() -> None:
    items = [with_asset(scene("a"), "red"), with_asset(scene("b"), "red")]
    stated = {"a": {"x": 1, "y": 1, "z": 1}, "b": {"x": 1, "y": 2, "z": 1}}
    loaded = pixels("red")
    loaded["red"].attrs["z"] = "mine"

    stamped = write_band_attrs(
        loaded, LoadSource(items), lambda item, name: stated[item.id]
    )

    assert stamped.red.attrs == {"x": 1, "z": "mine"}


# raster


def reflectance(id: str, *, bands=None, **asset_fields):
    """Return an Item whose `reflectance` asset states red and nir, unpacked."""
    item = raster_item(
        {"reflectance": ("/data/reflectance.tif", build_raster())},
        id=id,
        input_datetime=datetime(2025, 1, 1, tzinfo=UTC),
    )
    fields = item.assets["reflectance"].extra_fields
    for entry in fields["raster:bands"]:
        entry.pop("scale", None)
        entry.pop("offset", None)
    if bands is not None:
        for entry, stated in zip(fields["raster:bands"], bands, strict=True):
            entry.update(stated)
    fields.update(asset_fields)
    return item


def loaded_fill(*names: str) -> xr.Dataset:
    """Stand in for an ODC result: lazy variables with the fill ODC sets."""
    variables = {}
    for name in names:
        variables[name] = xr.DataArray(
            da.zeros((2, 2), dtype="uint16"), dims=("y", "x"), attrs={"nodata": 0}
        )
    return xr.Dataset(variables)


def test_raster_writes_packing_stated_on_the_asset() -> None:
    items = [reflectance("a", **{"raster:scale": 1e-4, "raster:offset": -0.1})]

    stamped = stamp_raster(loaded_fill("red", "nir"), LoadSource(items))

    assert stamped.red.attrs["scale_factor"] == 1e-4
    assert stamped.red.attrs["add_offset"] == -0.1
    assert stamped.nir.attrs["scale_factor"] == 1e-4


def test_raster_writes_packing_stated_per_band() -> None:
    bands = [
        {"scale": 1e-4, "offset": -0.1, "unit": "1"},
        {"scale": 2e-4, "offset": 0.0, "unit": "1"},
    ]
    items = [reflectance("a", bands=bands)]

    stamped = stamp_raster(loaded_fill("red", "nir"), LoadSource(items))

    assert stamped.red.attrs["scale_factor"] == 1e-4
    assert stamped.nir.attrs["scale_factor"] == 2e-4
    assert stamped.red.attrs["units"] == "1"


@pytest.mark.parametrize("field, value", [("scale", 2.0), ("offset", 1.0)])
def test_raster_refuses_items_packing_a_band_differently(field, value) -> None:
    first = reflectance("a", bands=[{"scale": 1.0, "offset": 0.0}, {}])
    second = reflectance("b", bands=[{field: value}, {}])

    with pytest.raises(ValueError, match="red.*incompatible source scale/offset"):
        stamp_raster(loaded_fill("red"), LoadSource([first, second]))

    # A conflict on a band that was not loaded does not matter.
    stamp_raster(loaded_fill("nir"), LoadSource([first, second]))


def test_raster_without_packing_leaves_values_unscaled() -> None:
    stamped = stamp_raster(loaded_fill("red"), LoadSource([reflectance("a")]))

    assert "scale_factor" not in stamped.red.attrs
    assert "add_offset" not in stamped.red.attrs


def test_raster_keeps_units_only_when_items_agree() -> None:
    first = reflectance("a", bands=[{"unit": "1"}, {"unit": "1"}])
    second = reflectance("b", bands=[{"unit": "m"}, {"unit": "1"}])

    stamped = stamp_raster(loaded_fill("red", "nir"), LoadSource([first, second]))

    assert "units" not in stamped.red.attrs
    assert stamped.nir.attrs["units"] == "1"


def test_raster_keeps_the_loaded_attrs_and_declares_the_fill() -> None:
    loaded = loaded_fill("red")
    loaded["red"].attrs["scale_factor"] = 5.0
    items = [reflectance("a", bands=[{"scale": 1e-4}, {}])]

    stamped = stamp_raster(loaded, LoadSource(items))

    assert stamped.red.attrs["scale_factor"] == 5.0
    assert "nodata" not in stamped.red.attrs
    assert stamped.red.rio.nodata == 0


def test_raster_reads_no_pixels() -> None:
    items = [reflectance("a", **{"raster:scale": 1e-4})]

    stamped = stamp_raster(loaded_fill("red"), LoadSource(items))

    assert stamped.red.chunks is not None


# eo


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

    stamped = stamp_eo(pixels("red"), LoadSource([item]))

    assert stamped.red.gs.spectral == EOBand(common_name="red", center_wavelength=0.665)


def test_eo_writes_the_description_as_long_name() -> None:
    item = with_asset(scene(), "red", **{"eo:bands": [{"description": "Reflectance"}]})

    stamped = stamp_eo(pixels("red"), LoadSource([item]))

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

    stamped = stamp_eo(pixels("red"), LoadSource([first, second]))

    assert stamped.red.attrs == {"common_name": "red"}


# classification


def test_classification_writes_flags_and_colours() -> None:
    classes = [
        {"value": 0, "name": "background"},
        {"value": 1, "name": "forest", "color_hint": "00FF00"},
    ]
    item = with_asset(
        scene(), "red", **{"raster:bands": [{"classification:classes": classes}]}
    )

    stamped = stamp_classification(pixels("red"), LoadSource([item]))

    legend = attrs.CFFlags.from_attrs(stamped.red.attrs)
    assert legend.classes == {0: "background", 1: "forest"}
    assert legend.colors[1] == "#00FF00"


def test_classification_leaves_unclassified_bands_alone() -> None:
    item = with_asset(scene(), "red")

    stamped = stamp_classification(pixels("red"), LoadSource([item]))

    assert stamped.red.attrs == {}


# title


def test_title_names_a_one_band_asset() -> None:
    item = with_asset(
        scene(),
        "red",
        title="Red reflectance",
        **{"eo:bands": [{"common_name": "red"}]},
    )

    stamped = stamp_title(pixels("red"), LoadSource([item]))

    assert stamped.red.attrs["long_name"] == "Red reflectance"


def test_title_skips_multi_band_assets() -> None:
    item = with_asset(scene(), "red", title="Red", **{"raster:bands": [{}, {}]})

    stamped = stamp_title(pixels("red.1", "red.2"), LoadSource([item]))

    assert "long_name" not in stamped["red.1"].attrs
    assert "long_name" not in stamped["red.2"].attrs


def test_title_yields_to_the_band_description() -> None:
    item = with_asset(
        scene(), "red", title="Red", **{"eo:bands": [{"description": "Reflectance"}]}
    )

    stamped = stamp_title(pixels("red"), LoadSource([item]))

    assert "long_name" not in stamped.red.attrs


# rgb


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

    stamped = stamp_rgb(pixels(*names), LoadSource([item]))

    assert stamped.gs.rgb == ("b1", "b2", "b0")


def test_rgb_states_nothing_without_a_blue_band() -> None:
    item, names = coloured([None, "red", "green"])

    stamped = stamp_rgb(pixels(*names), LoadSource([item]))

    assert stamped.gs.rgb is None


def test_rgb_reads_stac_not_attrs_another_stamp_wrote() -> None:
    item, names = coloured(["red", "green", "blue"])

    stamped = stamp_rgb(pixels(*names), LoadSource([item]))

    for variable in stamped.data_vars.values():
        assert "common_name" not in variable.attrs
    assert stamped.gs.rgb == ("b0", "b1", "b2")
