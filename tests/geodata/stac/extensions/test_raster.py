"""A variable's stored type, fill, packing and unit as its Raster band, and back."""

from __future__ import annotations

import dask
import numpy as np
import pytest
from pystac.extensions.raster import RasterBand

from geosave.geodata import open_raster
from geosave.geodata.stac.extensions import raster
from tests.geodata.conftest import build_raster, flat
from tests.geodata.stac.extensions.conftest import _empty


def _refuse(*args, **kwargs):
    raise AssertionError("describing the band computed pixels")


def test_a_saved_cog_states_what_it_stores(scene, tmp_path) -> None:
    (file, *_) = flat(
        scene.gs.to_cog(
            tmp_path, split_asset=(), format_paths="forest_{time:%Y%m%d}.tif"
        )
    )

    with open_raster(file) as saved, dask.config.set(scheduler=_refuse):
        bands = [raster.build_band(saved[name]) for name in saved.data_vars]

    assert [band.data_type for band in bands] == ["uint16", "uint16"]
    assert bands[0].nodata == 0


def test_a_dtype_stac_cannot_name_is_other() -> None:
    mask = build_raster().nir.astype(bool)

    assert raster.build_band(mask).data_type == "other"


@pytest.mark.parametrize(
    ("fill", "expected"), [(np.nan, "nan"), (np.inf, "inf"), (-np.inf, "-inf")]
)
def test_a_nonfinite_fill_uses_stac_strings(fill, expected) -> None:
    height = build_raster().nir.astype("float32")
    height.attrs["_FillValue"] = fill

    assert raster.build_band(height).nodata == expected


def test_packing_and_units_are_stamped_onto_the_variable() -> None:
    band = RasterBand.create(
        nodata=0, data_type="uint16", unit="1", scale=0.0001, offset=-0.1
    )

    stamped = _empty("B04").assign_attrs(raster.read_attrs(band))

    assert stamped.attrs == {"scale_factor": 0.0001, "add_offset": -0.1, "units": "1"}


def test_a_band_stating_only_its_type_stamps_no_attrs() -> None:
    band = RasterBand({"data_type": "float32", "nodata": 0})

    assert raster.read_attrs(band) == {}


def test_a_band_round_trips_through_its_variable() -> None:
    red = build_raster(packed=True).red.assign_attrs(units="1")

    stamped = _empty("red").assign_attrs(raster.read_attrs(raster.build_band(red)))

    assert stamped.gs.packing.scale_factor == red.gs.packing.scale_factor
    assert stamped.gs.packing.add_offset == red.gs.packing.add_offset
    # The file reader supplies effective nodata; STAC adds scaling and units.
    assert stamped.gs.nodata is None
    assert stamped.attrs["units"] == "1"
