"""Marking pixels nodata and decoding stored values."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from tests.conftest import build_dataset as build_raster
from tests.conftest import build_tree
from geosave.geodata.transform.nodata import mask, to_nan, unpack
from tests.geodata.conftest import geobox, titled


UTM = "EPSG:32633"


CLEAR = [4, 5, 6, 7]


def utm_box() -> GeoBox:
    """Build a small projected grid."""
    return GeoBox.from_bbox((0, 0, 40, 40), crs=UTM, resolution=10)


def scene(*, times: int = 0, nodata: float | int | None = 0) -> xr.Dataset:
    """Build a raster whose pixels all read 7, optionally over a time axis."""
    box = utm_box()
    if times:
        return build_raster(
            {
                "red": (
                    ("time", *("y", "x")),
                    np.full((times, *box.shape), 7, "uint16"),
                )
            },
            box,
            time=np.array(["2025-06-01", "2025-06-11"][:times], "datetime64[ns]"),
            nodata=nodata,
        )
    return build_raster(
        {
            "red": (("y", "x"), np.full(box.shape, 7, "uint16")),
            "nir": (("y", "x"), np.full(box.shape, 8, "uint16")),
        },
        box,
        nodata=nodata,
    )


def classification() -> xr.DataArray:
    """Build a scene classification band, half ground and half cloud."""
    dims = ("y", "x")
    return xr.DataArray(np.array([[4, 4, 9, 9]] * 4), dims=dims)


def test_masking_writes_the_declared_fill_without_promoting_the_dtype() -> None:
    source = scene()
    masked = mask(source, classification().isin(CLEAR))

    assert masked.red.dtype == np.dtype("uint16")
    assert list(masked.red.values[0]) == [7, 7, 0, 0]
    assert masked.gs.geobox == source.gs.geobox
    assert masked.red.attrs["_FillValue"] == 0


def test_a_flat_mask_covers_every_step_of_a_time_series() -> None:
    masked = mask(scene(times=2), classification().isin(CLEAR))

    assert masked.red.shape == (2, 4, 4)
    assert list(masked.red.values[0][0]) == [7, 7, 0, 0]
    assert list(masked.red.values[1][0]) == [7, 7, 0, 0]


def test_masking_a_band_returns_a_band() -> None:
    masked = mask(scene().red, classification().isin(CLEAR))

    assert isinstance(masked, xr.DataArray)
    assert list(masked.values[0]) == [7, 7, 0, 0]


def test_masking_a_tree_masks_every_group() -> None:
    source = build_tree({"optical": scene(), "other": scene()}).chunk()
    source.attrs = {"title": "training scene", "foreign": {"site": 7}}
    source.coords["site"] = "field-1"

    masked = mask(source, classification().isin(CLEAR))

    assert masked.gs.groups == ("optical", "other")
    assert masked.attrs == source.attrs
    xr.testing.assert_identical(masked.coords["site"], source.coords["site"])
    for group in masked.gs.rasters.values():
        assert group.red.chunks is not None
        assert list(group.red.values[0]) == [7, 7, 0, 0]


def test_a_non_boolean_mask_is_refused() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        mask(scene(), classification())


def test_a_variable_carrying_no_fill_is_refused_rather_than_promoted() -> None:
    with pytest.raises(ValueError, match="carries no fill value"):
        mask(scene(nodata=None), classification().isin(CLEAR))


def test_a_given_fill_is_written_and_declared() -> None:
    masked = mask(scene(), classification().isin(CLEAR), fill=255)

    assert list(masked.red.values[0]) == [7, 7, 255, 255]
    assert masked.red.attrs["_FillValue"] == 255
    assert masked.red.dtype == np.dtype("uint16")


@pytest.mark.parametrize("fill", [-1, 70000])
def test_a_fill_the_dtype_cannot_hold_is_refused_rather_than_wrapped(
    fill: int,
) -> None:
    with pytest.raises(ValueError, match="does not fit this uint16 band"):
        mask(scene(nodata=None), classification().isin(CLEAR), fill=fill)


def test_a_nan_fill_is_refused_for_an_integer_variable() -> None:
    with pytest.raises(ValueError, match="does not fit this uint16 band"):
        mask(scene(nodata=None), classification().isin(CLEAR), fill=float("nan"))


def test_a_nan_fill_is_accepted_by_a_float_variable() -> None:
    box = utm_box()
    floating = build_raster(
        {"elevation": (("y", "x"), np.ones(box.shape, "float32"))}, box
    )

    masked = mask(floating, classification().isin(CLEAR), fill=float("nan"))

    assert np.isnan(masked.elevation.values[0][-1])


def test_a_bare_numpy_mask_covers_every_step_of_a_time_series() -> None:
    bare = np.array([[True, True, False, False]] * 4)

    masked = mask(scene(times=2), bare)

    assert list(masked.red.values[0][0]) == [7, 7, 0, 0]
    assert list(masked.red.values[1][0]) == [7, 7, 0, 0]


def test_a_bare_numpy_mask_off_the_grid_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="names no axes"):
        mask(scene(), np.ones((3, 3), bool))


def test_a_mask_spanning_an_axis_the_raster_lacks_is_refused() -> None:
    dims = ("y", "x")
    extra = xr.DataArray(np.ones((2, 4, 4), bool), dims=("depth", *dims))

    with pytest.raises(ValueError, match="does not carry"):
        mask(scene(), extra)


def test_blanking_leaves_the_source_alone() -> None:
    # xarray's mask coder pops what it reads, so it must be handed a copy.
    source = scene()
    source["red"][0, 0] = 0

    blanked = to_nan(source)

    assert source.red.attrs["_FillValue"] == 0
    assert source.red.dtype == np.dtype("uint16")
    assert np.isnan(blanked.red.values[0, 0])
    assert np.isnan(to_nan(source).red.values[0, 0])


def test_odc_nodata_is_decoded_lazily_without_changing_input():
    import dask.array as da

    source = xr.DataArray(np.array([0, 7]), dims="x", attrs={"nodata": 0}).chunk(x=1)
    result = to_nan(source)
    assert isinstance(result.data, da.Array)
    np.testing.assert_allclose(result, [np.nan, 7], equal_nan=True)
    assert "nodata" not in result.attrs
    assert source.attrs == {"nodata": 0}


def test_preparation_preserves_coordinates_on_independent_dimensions():
    source = scene().assign_coords(class_label=("class", ["water", "land"]))
    source.red.attrs["scale_factor"] = 0.01
    result = source.gs.to_nan().gs.unpack()
    xr.testing.assert_identical(result.class_label, source.class_label)


def test_a_packed_16_bit_band_blanks_to_float32_as_read_from_disk() -> None:
    # A file's packing arrives as Python floats, which must not widen the result.
    band = xr.DataArray(
        np.array([[0, 5000]], "uint16"),
        dims=("y", "x"),
        attrs={"_FillValue": 0.0, "scale_factor": 1e-4},
        name="red",
    )

    blanked = to_nan(band)

    assert blanked.dtype == np.float32
    assert blanked.attrs["scale_factor"] == 1e-4


def stored(
    value: int = 1000,
    *,
    scale: float | None = 1e-4,
    offset: float | None = None,
    nodata: int | None = None,
    dtype: str = "uint16",
) -> xr.Dataset:
    """Build a raster whose `red` holds `value` as a stored digital number."""
    box = geobox(shape=(2, 2))
    raster = build_raster(
        {"red": (("y", "x"), np.full(box.shape, value, dtype))}, box, nodata=nodata
    )
    packing = {}
    if scale is not None:
        packing["scale_factor"] = scale
    if offset is not None:
        packing["add_offset"] = offset
    if not packing:
        return raster
    return raster.gs.write_packing(
        packing.get("scale_factor", 1.0), packing.get("add_offset", 0.0), target="red"
    )


def test_an_offset_is_added_after_the_scale() -> None:
    physical = unpack(stored(1000, scale=1e-4, offset=5.0))

    assert physical.red.values[0, 0] == pytest.approx(5.1)
    assert "scale_factor" not in physical.red.attrs
    assert "add_offset" not in physical.red.attrs


def test_an_offset_alone_shifts_without_scaling() -> None:
    physical = unpack(stored(1000, scale=None, offset=5.0))

    assert physical.red.values[0, 0] == pytest.approx(1005.0)


def test_a_variable_carrying_no_packing_passes_through_untouched() -> None:
    plain = stored(1000, scale=None)

    physical = unpack(plain)

    assert physical.red.values[0, 0] == 1000
    assert physical.red.dtype == np.dtype("uint16")


def test_unpacking_leaves_the_source_alone() -> None:
    # xarray's scale coder pops what it reads, so it must be handed a copy.
    source = stored(1000, scale=1e-4)

    first = unpack(source)

    assert source.red.attrs["scale_factor"] == pytest.approx(1e-4)
    assert source.red.dtype == np.dtype("uint16")
    assert first.red.values[0, 0] == pytest.approx(unpack(source).red.values[0, 0])


def test_other_attrs_and_coordinates_ride_through() -> None:
    source = stored(1000, scale=1e-4)
    source.attrs["title"] = "scene"
    source["red"].attrs.update({"long_name": "Red", "units": "1"})

    physical = unpack(source)

    assert physical.attrs["title"] == "scene"
    assert physical.red.attrs["long_name"] == "Red"
    assert physical.gs.geobox == source.gs.geobox


def test_a_variable_that_is_not_packed_rides_alongside_one_that_is() -> None:
    source = stored(1000, scale=1e-4)
    source["dem"] = (("y", "x"), np.full(geobox(shape=(2, 2)).shape, 5.0, "float32"))

    physical = unpack(source)

    assert sorted(str(name) for name in physical.data_vars) == ["dem", "red"]
    assert physical.dem.values[0, 0] == pytest.approx(5.0)
    assert physical.red.values[0, 0] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("fill", "scale", "offset", "dtype"),
    [
        (0, 1e-4, 5.0, "int32"),
        (65535, 1e-4, None, "uint16"),
        (-9999, 0.01, None, "int32"),
        (0, 1e-4, None, "int32"),
        (0, None, 5.0, "int32"),
    ],
    ids=["offset", "large fill", "negative fill", "fixed point", "offset only"],
)
def test_unpacking_refuses_a_variable_still_marking_its_nodata(
    fill: int, scale: float | None, offset: float | None, dtype: str
) -> None:
    packed = stored(1000, scale=scale, offset=offset, nodata=fill, dtype=dtype)

    with pytest.raises(ValueError, match="marks its nodata pixels"):
        unpack(packed)


def test_blanking_the_nodata_first_keeps_it_apart_from_a_reading() -> None:
    packed = stored(1000, scale=1e-4, offset=5.0, nodata=0, dtype="int32")
    packed["red"][0, 0] = 0

    physical = unpack(to_nan(packed))

    assert np.isnan(physical.red.values[0, 0])
    assert physical.red.values[0, 1] == pytest.approx(5.1)


def test_a_variable_marking_nodata_but_carrying_no_packing_is_left_alone() -> None:
    plain = stored(1000, scale=None, nodata=0, dtype="int32")

    physical = unpack(plain)

    assert physical.red.values[0, 0] == 1000
    assert physical.red.gs.nodata == 0


def test_unpacking_refuses_a_variable_whose_values_are_class_codes() -> None:
    labelled = stored(1, scale=1e-4, nodata=None, dtype="int32").gs.write_legend(
        {0: "bg", 1: "palm"}, target="red"
    )

    # Packing decodes a measurement; a class code measures nothing.
    with pytest.raises(ValueError, match="holds class codes and declares packing"):
        unpack(labelled)


def test_a_tree_unpacks_group_by_group_and_stays_lazy() -> None:
    tree = build_tree(
        {
            "optical": stored(1000, scale=1e-4).chunk(),
            "dem": stored(7, scale=None).chunk(),
        }
    )
    tree = titled(tree)
    tree.coords["site"] = "field-1"

    physical = unpack(tree)

    assert physical.gs.groups == ("optical", "dem")
    assert physical["optical"].gs.to_dataset().red.chunks is not None
    assert physical["optical"].gs.to_dataset().red.values[0, 0] == pytest.approx(0.1)
    assert physical["dem"].gs.to_dataset().red.dtype == np.dtype("uint16")
    assert physical.attrs["title"] == "scene"
    assert physical.coords["site"].item() == "field-1"


def test_a_tree_blanks_nodata_group_by_group_and_stays_lazy() -> None:
    tree = build_tree(
        {
            "optical": stored(0, scale=None, nodata=0).chunk(),
            "dem": stored(7, scale=None).chunk(),
        }
    )
    tree = titled(tree)
    tree.coords["site"] = "field-1"

    blanked = to_nan(tree)

    assert blanked["optical"].gs.to_dataset().red.chunks is not None
    assert np.isnan(blanked["optical"].gs.to_dataset().red.values).all()
    assert blanked["dem"].gs.to_dataset().red.dtype == np.dtype("uint16")
    assert blanked.attrs["title"] == "scene"
    assert blanked.coords["site"].item() == "field-1"


@pytest.mark.parametrize(
    ("dtype", "unpacked"),
    [
        ("uint8", "float32"),
        ("uint16", "float32"),
        ("int16", "float32"),
        ("int32", "float64"),
        # Blanking nodata first leaves 16-bit numbers stored as float32.
        ("float32", "float32"),
        ("float64", "float64"),
    ],
)
def test_unpacking_reads_small_integers_as_float32(dtype: str, unpacked: str) -> None:
    physical = unpack(stored(100, scale=1e-4, dtype=dtype))

    assert physical.red.dtype == np.dtype(unpacked)
    assert physical.red.values[0, 0] == pytest.approx(0.01)
