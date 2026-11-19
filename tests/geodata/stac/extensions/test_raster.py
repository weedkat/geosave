"""A variable's stored type, fill, packing and unit as its Raster band, and back."""

from __future__ import annotations

from tests.conftest import write_cogs

import dask
import numpy as np
from pystac.extensions.raster import RasterBand

from geosave.geodata import open_raster
from geosave.geodata.stac.extensions import raster
from tests.geodata.conftest import build_raster, flat
from tests.geodata.stac.extensions.conftest import _empty


def _refuse(*args, **kwargs):
    raise AssertionError("describing the band computed pixels")


def test_a_saved_cog_states_what_it_stores(scene, tmp_path) -> None:
    (file, *_) = flat(
        write_cogs(scene, tmp_path, format_paths="forest_{time:%Y%m%d}.tif")
    )

    with open_raster(file) as saved, dask.config.set(scheduler=_refuse):
        bands = [raster.build_band(saved[name]) for name in saved.data_vars]

    assert [band.data_type for band in bands] == ["uint16", "uint16"]
    assert bands[0].nodata == 0


def test_packing_and_units_are_stamped_onto_the_variable() -> None:
    band = RasterBand.create(
        nodata=0, data_type="uint16", unit="1", scale=0.0001, offset=-0.1
    )

    stamped = _empty("B04").assign_attrs(raster.read_attrs(band))

    assert stamped.attrs == {"scale_factor": 0.0001, "add_offset": -0.1, "units": "1"}


def test_a_band_round_trips_through_its_variable() -> None:
    red = build_raster(packed=True).red.assign_attrs(units="1")

    stamped = _empty("red").assign_attrs(raster.read_attrs(raster.build_band(red)))

    # The band spells the float32 packing at its own precision: 1e-4, not
    # 9.999999747378752e-05, which reads back as the same float32.
    assert np.float32(stamped.gs.packing.scale_factor) == red.attrs["scale_factor"]
    assert np.float32(stamped.gs.packing.add_offset) == red.attrs["add_offset"]
    # The file reader supplies effective nodata; STAC adds scaling and units.
    assert stamped.gs.nodata is None
    assert stamped.attrs["units"] == "1"


def test_decoded_pixels_describe_their_encoded_storage_without_execution():
    stored = build_raster(packed=True).chunk().red.assign_attrs(units="1")
    decoded = stored.gs.mask_and_scale()

    with dask.config.set(scheduler=_refuse):
        band = raster.build_band(decoded)

    assert band.data_type == "uint16"
    assert band.nodata == 0
    assert band.scale == 0.0001
    assert band.offset == 0
    assert band.unit == "1"
    assert decoded.dtype == "float32"
    assert "scale_factor" not in decoded.attrs
