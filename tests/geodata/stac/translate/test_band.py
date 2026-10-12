from __future__ import annotations

from pystac.extensions.eo import Band
from pystac.extensions.raster import RasterBand

from geosave.geodata.stac.translate import band


def test_a_band_states_units_and_packing_as_attrs() -> None:
    raster = RasterBand.create(nodata=0, unit="1", scale=0.0001)
    spectral = Band.create(name="red", description="Red")

    assert band.read(raster, spectral) == {
        "units": "1",
        "long_name": "Red",
        "scale_factor": 0.0001,
    }


def test_a_bare_band_states_no_attrs() -> None:
    assert band.read(RasterBand({"data_type": "float32"}), Band({"name": "dem"})) == {}
