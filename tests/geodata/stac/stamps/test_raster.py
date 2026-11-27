"""Scale, offset and units for loaded STAC bands."""

from datetime import UTC, datetime

import dask.array as da
import pytest
import xarray as xr

from geosave.geodata.stac.stamps import stamp_nodata, stamp_raster
from tests.geodata.conftest import build_raster, raster_item
from tests.geodata.stac.stamps.conftest import scene, with_asset


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
    """Stand in for lazy loaded variables with their output fill declared."""
    variables = {}
    for name in names:
        variables[name] = xr.DataArray(
            da.zeros((2, 2), dtype="uint16"), dims=("y", "x"), attrs={"_FillValue": 0}
        )
    return xr.Dataset(variables)


def test_nodata_stamp_preserves_each_output_fill_and_ignores_time_properties():
    ds = loaded_fill("red", "nir")
    ds.red.attrs = {"nodata": 0, "scale_factor": 0.01}
    ds.nir.attrs = {"nodata": 65535}
    ds["cloud"] = xr.DataArray([15], dims="time", attrs={"nodata": -1})

    stamped = stamp_nodata(ds, [])

    assert stamped.red.attrs == {"_FillValue": 0, "scale_factor": 0.01}
    assert stamped.nir.gs.nodata == 65535
    assert stamped.cloud.identical(ds.cloud)
    assert stamped.red.data is ds.red.data
    assert ds.red.attrs["nodata"] == 0


def test_raster_writes_packing_stated_on_the_asset() -> None:
    items = [reflectance("a", **{"raster:scale": 1e-4, "raster:offset": -0.1})]

    stamped = stamp_raster(loaded_fill("red", "nir"), items)

    assert stamped.red.attrs["scale_factor"] == 1e-4
    assert stamped.red.attrs["add_offset"] == -0.1
    assert stamped.nir.attrs["scale_factor"] == 1e-4


def test_raster_writes_packing_stated_per_band() -> None:
    bands = [
        {"scale": 1e-4, "offset": -0.1, "unit": "1"},
        {"scale": 2e-4, "offset": 0.0, "unit": "1"},
    ]
    items = [reflectance("a", bands=bands)]

    stamped = stamp_raster(loaded_fill("red", "nir"), items)

    assert stamped.red.attrs["scale_factor"] == 1e-4
    assert stamped.nir.attrs["scale_factor"] == 2e-4
    assert stamped.red.attrs["units"] == "1"


@pytest.mark.parametrize("field, value", [("scale", 2.0), ("offset", 1.0)])
def test_raster_refuses_items_packing_a_band_differently(field, value) -> None:
    first = reflectance("a", bands=[{"scale": 1.0, "offset": 0.0}, {}])
    second = reflectance("b", bands=[{field: value}, {}])

    with pytest.raises(ValueError, match="red.*incompatible source scale/offset"):
        stamp_raster(loaded_fill("red"), [first, second])

    # A conflict on a band that was not loaded does not matter.
    stamp_raster(loaded_fill("nir"), [first, second])


def test_raster_without_packing_leaves_values_unscaled() -> None:
    stamped = stamp_raster(loaded_fill("red"), [reflectance("a")])

    assert "scale_factor" not in stamped.red.attrs
    assert "add_offset" not in stamped.red.attrs


def test_raster_checks_only_items_that_supply_the_loaded_band() -> None:
    packed = reflectance("a", bands=[{"scale": 1e-4}, {}])
    missing = with_asset(scene("b"), "swir")

    stamped = stamp_raster(loaded_fill("red"), [packed, missing])

    assert stamped.red.attrs["scale_factor"] == 1e-4

    unscaled = reflectance("c")
    with pytest.raises(ValueError, match="red.*incompatible source scale/offset"):
        stamp_raster(loaded_fill("red"), [packed, unscaled])


def test_raster_keeps_units_only_when_items_agree() -> None:
    first = reflectance("a", bands=[{"unit": "1"}, {"unit": "1"}])
    second = reflectance("b", bands=[{"unit": "m"}, {"unit": "1"}])

    stamped = stamp_raster(loaded_fill("red", "nir"), [first, second])

    assert "units" not in stamped.red.attrs
    assert stamped.nir.attrs["units"] == "1"


def test_raster_preserves_the_loaded_scale_and_fill() -> None:
    loaded = loaded_fill("red")
    loaded["red"].attrs["scale_factor"] = 5.0
    items = [reflectance("a", bands=[{"scale": 1e-4}, {}])]

    stamped = stamp_raster(loaded, items)

    assert stamped.red.attrs["scale_factor"] == 5.0
    assert "nodata" not in stamped.red.attrs
    assert stamped.red.rio.nodata == 0


def test_raster_reads_no_pixels() -> None:
    items = [reflectance("a", **{"raster:scale": 1e-4})]

    stamped = stamp_raster(loaded_fill("red"), items)

    assert stamped.red.chunks is not None
