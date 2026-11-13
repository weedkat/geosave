"""A variable's stored type, fill, packing and unit as its Raster band, and back."""

from __future__ import annotations

from tests.conftest import write_cogs

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pystac.extensions.raster import RasterBand

from geosave.geodata import open_raster
from geosave.geodata.stac.extensions import raster
from geosave.geodata.stac.stamps import STAMPS, LoadSource
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


def _item(id: str, *, bands=None, **asset_fields):
    """Build an Item whose `reflectance` asset states red and nir."""
    from datetime import UTC, datetime

    from tests.geodata.conftest import raster_item

    source = build_raster()
    item = raster_item(
        {"reflectance": ("/data/reflectance.tif", source)},
        id=id,
        input_datetime=datetime(2025, 1, 1, tzinfo=UTC),
    )
    extra = item.assets["reflectance"].extra_fields
    for entry in extra["raster:bands"]:
        entry.pop("scale", None)
        entry.pop("offset", None)
    if bands is not None:
        for entry, fields in zip(extra["raster:bands"], bands, strict=True):
            entry.update(fields)
    extra.update(asset_fields)
    return item


def _loaded(*names: str) -> xr.Dataset:
    """Stand in for the ODC result: variables with the fill ODC would set."""
    variables = {}
    for name in names:
        variables[name] = xr.DataArray(
            da.zeros((2, 2), dtype="uint16"),
            dims=("y", "x"),
            attrs={"nodata": 0},
        )
    return xr.Dataset(variables)


def _stamp(items, *names: str) -> xr.Dataset:
    return raster.stamp_raster(_loaded(*names), LoadSource(items, None, "id"))


def test_raster_stamp_is_registered() -> None:
    assert STAMPS["raster"] is raster.stamp_raster


def test_raster_stamp_writes_packing_from_asset_level_fields() -> None:
    items = [_item("a", **{"raster:scale": 1e-4, "raster:offset": -0.1})]

    stamped = _stamp(items, "red", "nir")

    assert stamped.red.attrs["scale_factor"] == 1e-4
    assert stamped.red.attrs["add_offset"] == -0.1
    assert stamped.nir.attrs["scale_factor"] == 1e-4


def test_raster_stamp_writes_packing_from_band_lists() -> None:
    bands = [
        {"scale": 1e-4, "offset": -0.1, "unit": "1"},
        {"scale": 2e-4, "offset": 0.0, "unit": "1"},
    ]
    stamped = _stamp([_item("a", bands=bands)], "red", "nir")

    assert stamped.red.attrs["scale_factor"] == 1e-4
    assert stamped.nir.attrs["scale_factor"] == 2e-4
    assert stamped.red.attrs["units"] == "1"


@pytest.mark.parametrize("field, value", [("scale", 2.0), ("offset", 1.0)])
def test_raster_stamp_refuses_items_packing_a_band_differently(field, value) -> None:
    first = _item("a", bands=[{"scale": 1.0, "offset": 0.0}, {}])
    second = _item("b", bands=[{field: value}, {}])

    with pytest.raises(ValueError, match="red.*incompatible source scale/offset"):
        _stamp([first, second], "red")

    # A conflict on an unloaded band does not raise.
    assert "nir" in _stamp([first, second], "nir")


def test_raster_stamp_without_packing_leaves_values_unscaled() -> None:
    stamped = _stamp([_item("a")], "red")

    assert "scale_factor" not in stamped.red.attrs
    assert "add_offset" not in stamped.red.attrs


def test_raster_stamp_keeps_units_only_when_items_agree() -> None:
    first = _item("a", bands=[{"unit": "1"}, {"unit": "1"}])
    second = _item("b", bands=[{"unit": "m"}, {"unit": "1"}])

    stamped = _stamp([first, second], "red", "nir")

    assert "units" not in stamped.red.attrs
    assert stamped.nir.attrs["units"] == "1"


def test_raster_stamp_keeps_the_loaded_attrs_and_writes_nodata() -> None:
    items = [_item("a", bands=[{"scale": 1e-4}, {}])]
    loaded = _loaded("red")
    loaded["red"].attrs["scale_factor"] = 5.0

    stamped = raster.stamp_raster(loaded, LoadSource(items, None, "id"))

    assert stamped.red.attrs["scale_factor"] == 5.0
    assert "nodata" not in stamped.red.attrs
    assert stamped.red.rio.nodata == 0
