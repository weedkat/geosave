"""The EO module names every band, adding spectral facts where a band has them."""

from __future__ import annotations

from datetime import UTC, datetime

import pystac
from pystac.extensions.eo import Band, EOExtension

from geosave.geodata.stac.extensions import eo

from tests.geodata.conftest import build_raster


def _asset(href: str = "scene.tif") -> pystac.Asset:
    item = pystac.Item("scene", None, None, datetime(2025, 1, 1, tzinfo=UTC), {})
    item.add_asset("image", pystac.Asset(href, roles=["data"]))
    return item.assets["image"]


def test_a_spectral_raster_names_every_band() -> None:
    scene = build_raster().gs.write_spectral(
        "red", center_wavelength=0.665, target="red"
    )
    scene = scene.assign(red=scene.red.assign_attrs(long_name="Red"))
    asset = _asset()

    eo.write(asset, scene)

    red, nir = EOExtension.ext(asset).bands
    assert (red.name, red.common_name, red.center_wavelength) == ("red", "red", 0.665)
    assert red.description == "Red"
    assert (nir.name, nir.common_name) == ("nir", None)
    assert asset.owner.stac_extensions == [EOExtension.get_schema_uri()]


def test_a_raster_with_no_spectral_facts_still_names_its_bands() -> None:
    dem = build_raster()[["nir"]].astype("float32").rename(nir="height")
    asset = _asset()

    eo.write(asset, dem)

    (height,) = EOExtension.ext(asset).bands
    assert (height.name, height.common_name, height.center_wavelength) == (
        "height",
        None,
        None,
    )
    assert asset.owner.stac_extensions == [EOExtension.get_schema_uri()]


def test_spectral_facts_and_the_description_are_read_from_a_band() -> None:
    band = Band.create(
        name="B04", common_name="red", center_wavelength=0.665, description="Red"
    )

    assert eo.read(band) == {
        "common_name": "red",
        "center_wavelength": 0.665,
        "long_name": "Red",
    }


def test_a_band_stating_only_its_name_reads_nothing() -> None:
    assert eo.read(Band({"name": "B08"})) == {}
