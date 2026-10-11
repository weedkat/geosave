"""Marking pixels nodata and decoding stored values."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.core.constructors import raster as build_raster
from geosave_engine.geodata.core.stack import stack as build_stack
from geosave_engine.geodata.transform.nodata import mask, to_nan, unpack
from tests.geodata.conftest import titled

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
    masked = mask(scene(), classification().isin(CLEAR))

    assert masked.red.dtype == np.dtype("uint16")
    assert list(masked.red.values[0]) == [7, 7, 0, 0]


def test_masking_keeps_the_grid_and_the_variable_attrs() -> None:
    source = scene()

    masked = mask(source, classification().isin(CLEAR))

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


def test_masking_a_stack_masks_every_group() -> None:
    source = build_stack({"optical": scene(), "other": scene()})

    masked = mask(source, classification().isin(CLEAR))

    assert masked.gs.groups == ("optical", "other")
    for group in masked.gs.rasters.values():
        assert list(group.red.values[0]) == [7, 7, 0, 0]


def test_a_non_boolean_mask_is_refused() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        mask(scene(), classification())


def test_a_variable_carrying_no_fill_is_refused_rather_than_promoted() -> None:
    with pytest.raises(ValueError, match="carries no fill value"):
        mask(scene(nodata=None), classification().isin(CLEAR))


def test_a_given_fill_is_written_and_declared() -> None:
    masked = mask(scene(nodata=None), classification().isin(CLEAR), fill=255)

    assert list(masked.red.values[0]) == [7, 7, 255, 255]
    assert masked.red.attrs["_FillValue"] == 255
    assert masked.red.dtype == np.dtype("uint16")


def test_a_given_fill_overrides_what_the_variable_declared() -> None:
    masked = mask(scene(), classification().isin(CLEAR), fill=99)

    assert list(masked.red.values[0]) == [7, 7, 99, 99]
    assert masked.red.attrs["_FillValue"] == 99


@pytest.mark.parametrize("fill", [-1, 70000])
def test_a_fill_the_dtype_cannot_hold_is_refused_rather_than_wrapped(
    fill: int,
) -> None:
    with pytest.raises(ValueError, match="is no uint16 value"):
        mask(scene(nodata=None), classification().isin(CLEAR), fill=fill)


def test_a_nan_fill_is_refused_for_an_integer_variable() -> None:
    with pytest.raises(ValueError, match="is no uint16 value"):
        mask(scene(nodata=None), classification().isin(CLEAR), fill=float("nan"))


def test_a_nan_fill_is_accepted_by_a_float_variable() -> None:
    box = utm_box()
    floating = build_raster(
        {"elevation": (("y", "x"), np.ones(box.shape, "float32"))}, box
    )

    masked = mask(floating, classification().isin(CLEAR), fill=float("nan"))

    assert np.isnan(masked.elevation.values[0][-1])


def test_a_bare_numpy_mask_is_read_as_the_grid() -> None:
    bare = np.array([[True, True, False, False]] * 4)

    masked = mask(scene(), bare)

    assert list(masked.red.values[0]) == [7, 7, 0, 0]
    assert masked.red.dtype == np.dtype("uint16")


def test_a_bare_numpy_mask_covers_every_step_of_a_time_series() -> None:
    bare = np.array([[True, True, False, False]] * 4)

    masked = mask(scene(times=2), bare)

    assert list(masked.red.values[0][0]) == [7, 7, 0, 0]
    assert list(masked.red.values[1][0]) == [7, 7, 0, 0]


def test_a_bare_numpy_mask_off_the_grid_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="names no axes"):
        mask(scene(), np.ones((3, 3), bool))


def test_a_mask_that_misses_the_grid_is_refused() -> None:
    off_grid = xr.DataArray(np.array([True, False]), dims="depth")

    with pytest.raises(ValueError, match="does not cover the grid"):
        mask(scene(), off_grid)


def test_a_mask_spanning_an_axis_the_raster_lacks_is_refused() -> None:
    dims = ("y", "x")
    extra = xr.DataArray(np.ones((2, 4, 4), bool), dims=("depth", *dims))

    with pytest.raises(ValueError, match="does not carry"):
        mask(scene(), extra)


def test_cropping_with_a_mask_keeps_the_dtype_and_declared_fill() -> None:
    import geopandas as gpd
    import shapely

    source = scene()
    half = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 20, 40)], crs=UTM)

    cropped = source.gs.crop(half, mask=True)

    assert cropped.red.dtype == np.dtype("uint16")
    assert cropped.red.attrs["_FillValue"] == 0
    # odc's own apply_mask writes NaN, promoting the variable to float.
    from odc.geo.geom import box as geom_box

    assert source.odc.crop(
        geom_box(0, 0, 20, 40, crs=UTM), apply_mask=True
    ).red.dtype == np.dtype("float32")


def test_blanking_leaves_the_source_alone() -> None:
    # xarray's mask coder pops what it reads, so it must be handed a copy.
    source = scene()
    source["red"][0, 0] = 0

    blanked = to_nan(source)

    assert source.red.attrs["_FillValue"] == 0
    assert source.red.dtype == np.dtype("uint16")
    assert np.isnan(blanked.red.values[0, 0])
    assert np.isnan(to_nan(source).red.values[0, 0])


def test_blanking_leaves_the_packing_alone() -> None:
    packed = scene().gs.write_packing(1e-4, target="red")

    assert to_nan(packed).red.attrs["scale_factor"] == 1e-4


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


def test_tree_mask_preserves_root_metadata_and_coordinates():
    source = build_stack({"optical": scene()})
    source.attrs = {"title": "my training scene"}
    source.coords["site"] = "field-1"
    masked = mask(source, classification().isin(CLEAR))
    assert masked.attrs == source.attrs
    xr.testing.assert_identical(masked.coords["site"], source.coords["site"])
    assert list(masked["optical"].red.values[0]) == [7, 7, 0, 0]


def test_masking_a_stack_keeps_its_root_attrs_and_stays_lazy(
    stack: xr.DataTree,
) -> None:
    chunked = titled(stack.chunk())
    valid = np.array([[True, False], [True, True]])

    masked = mask(chunked, valid, fill=0)

    red = masked.gs.rasters["optical"].red
    assert red.chunks is not None
    assert red.values.tolist() == [[1000, 0], [3000, 0]]
    assert masked.attrs["title"] == "scene"


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


def packed_box() -> GeoBox:
    """Build a small projected grid."""
    return GeoBox.from_bbox((300000, 5000000, 300020, 5000020), crs=UTM, resolution=10)


def stored(
    value: int = 1000,
    *,
    scale: float | None = 1e-4,
    offset: float | None = None,
    nodata: int | None = None,
    dtype: str = "uint16",
) -> xr.Dataset:
    """Build a raster whose `red` holds `value` as a stored digital number."""
    box = packed_box()
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


def test_stored_numbers_read_out_as_physical_values() -> None:
    physical = unpack(stored(1000, scale=1e-4))

    assert physical.red.values[0, 0] == pytest.approx(0.1)


def test_an_offset_is_added_after_the_scale() -> None:
    physical = unpack(stored(1000, scale=1e-4, offset=5.0))

    assert physical.red.values[0, 0] == pytest.approx(5.1)


def test_an_offset_alone_shifts_without_scaling() -> None:
    physical = unpack(stored(1000, scale=None, offset=5.0))

    assert physical.red.values[0, 0] == pytest.approx(1005.0)


def test_a_variable_carrying_no_packing_passes_through_untouched() -> None:
    plain = stored(1000, scale=None)

    physical = unpack(plain)

    assert physical.red.values[0, 0] == 1000
    assert physical.red.dtype == np.dtype("uint16")


def test_the_packing_leaves_attrs_with_the_values_it_described() -> None:
    physical = unpack(stored(1000, scale=1e-4, offset=5.0))

    assert "scale_factor" not in physical.red.attrs
    assert "add_offset" not in physical.red.attrs


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
    source["dem"] = (("y", "x"), np.full(packed_box().shape, 5.0, "float32"))

    physical = unpack(source)

    assert sorted(str(name) for name in physical.data_vars) == ["dem", "red"]
    assert physical.dem.values[0, 0] == pytest.approx(5.0)
    assert physical.red.values[0, 0] == pytest.approx(0.1)


def test_a_band_unpacks_as_a_band() -> None:
    physical = unpack(stored(1000, scale=1e-4).red)

    assert isinstance(physical, xr.DataArray)
    assert physical.values[0, 0] == pytest.approx(0.1)


def test_unpacking_stays_lazy_while_its_raster_is() -> None:
    physical = unpack(stored(1000, scale=1e-4).chunk())

    assert physical.red.chunks is not None
    assert physical.red.compute().values[0, 0] == pytest.approx(0.1)


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


def test_a_stack_unpacks_group_by_group_and_stays_lazy() -> None:
    tree = build_stack(
        {
            "optical": stored(1000, scale=1e-4).chunk(),
            "dem": stored(7, scale=None).chunk(),
        }
    )
    tree = titled(tree)
    tree.coords["site"] = "field-1"

    physical = unpack(tree)

    rasters = physical.gs.rasters
    assert physical.gs.groups == ("optical", "dem")
    assert rasters["optical"].red.chunks is not None
    assert rasters["optical"].red.values[0, 0] == pytest.approx(0.1)
    assert rasters["dem"].red.dtype == np.dtype("uint16")
    assert physical.attrs["title"] == "scene"
    assert physical.coords["site"].item() == "field-1"


def test_a_stack_blanks_nodata_group_by_group_and_stays_lazy() -> None:
    tree = build_stack(
        {
            "optical": stored(0, scale=None, nodata=0).chunk(),
            "dem": stored(7, scale=None).chunk(),
        }
    )
    tree = titled(tree)
    tree.coords["site"] = "field-1"

    blanked = to_nan(tree)

    rasters = blanked.gs.rasters
    assert rasters["optical"].red.chunks is not None
    assert np.isnan(rasters["optical"].red.values).all()
    assert rasters["dem"].red.dtype == np.dtype("uint16")
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
