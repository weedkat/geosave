"""A variable's name and spectral facts as its EO band, and back."""

from __future__ import annotations

from pystac.extensions.eo import Band

from geosave.geodata.stac.extensions import eo
from tests.geodata.conftest import build_raster
from tests.geodata.stac.extensions.conftest import _empty


def test_a_spectral_variable_names_its_band() -> None:
    scene = build_raster().gs.write_spectral(
        "red", center_wavelength=0.665, target="red"
    )

    red = eo.from_dataarray(scene.red.assign_attrs(long_name="Red"))
    nir = eo.from_dataarray(scene.nir)

    assert (red.name, red.common_name, red.center_wavelength) == ("red", "red", 0.665)
    assert red.description == "Red"
    assert (nir.name, nir.common_name) == ("nir", None)


def test_spectral_facts_and_the_description_are_stamped_onto_the_variable() -> None:
    band = Band.create(
        name="B04", common_name="red", center_wavelength=0.665, description="Red"
    )

    assert eo.stamp(_empty("B04"), band).attrs == {
        "common_name": "red",
        "center_wavelength": 0.665,
        "long_name": "Red",
    }


def test_a_band_stating_only_its_name_stamps_no_attrs() -> None:
    assert eo.stamp(_empty("B08"), Band({"name": "B08"})).attrs == {}
