"""Native CF conversions preserve values, metadata and lazy execution."""

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest

from geosave.geodata.warnings import DroppedAttrsWarning
import xarray as xr

from geosave.geodata import attrs
from tests.conftest import build_tree
from tests.geodata.conftest import build_raster


def test_decode_and_encode_match_native_cf_and_are_idempotent():
    source = build_raster(packed=True).chunk({"x": 1, "y": 1})
    source.red.attrs.update(scale_factor=0.0001, add_offset=-0.1)
    expected = xr.decode_cf(source, decode_coords=False)

    with Callback(pretask=lambda *_: pytest.fail("conversion read pixels")):
        decoded = source.gs.mask_and_scale()
        restored = decoded.gs.scale_and_fill()

    xr.testing.assert_allclose(decoded, expected)
    assert decoded.red.dtype == expected.red.dtype
    assert isinstance(decoded.red.data, da.Array)
    assert decoded.gs.geobox == source.gs.geobox
    assert "scale_factor" not in decoded.red.attrs
    assert decoded.red.encoding["scale_factor"] == 0.0001
    assert decoded.red.encoding["dtype"] == source.red.dtype
    assert decoded.red.encoding["_FillValue"] == 0
    assert restored.red.encoding.get("scale_factor") is None
    assert restored.red.attrs["scale_factor"] == 0.0001
    xr.testing.assert_equal(restored, source)
    xr.testing.assert_identical(decoded.gs.mask_and_scale(), decoded)
    xr.testing.assert_identical(restored.gs.scale_and_fill(), restored)
    assert source.red.attrs["_FillValue"] == 0
    assert "scale_factor" not in source.red.encoding


def test_an_unnamed_array_keeps_name_coordinates_and_original_dtype():
    raw = build_raster(packed=True).red.rename(None)
    decoded = raw.gs.mask_and_scale()
    restored = decoded.gs.scale_and_fill()

    assert decoded.name is None
    assert restored.name is None
    xr.testing.assert_equal(restored, raw)
    xr.testing.assert_identical(decoded.coords.to_dataset(), raw.coords.to_dataset())


def test_decoding_a_dimension_named_value_does_not_materialize_an_index():
    raw = xr.DataArray(
        da.from_array(np.array([0, 1], "uint16"), chunks=1), dims="value"
    )
    raw.attrs["_FillValue"] = 0
    with Callback(pretask=lambda *_: pytest.fail("decoding read pixels")):
        decoded = raw.gs.mask_and_scale()
    assert isinstance(decoded.data, da.Array)
    np.testing.assert_allclose(decoded, [np.nan, 1], equal_nan=True)


def test_nodata_only_offset_only_and_missing_codes_use_native_decoding():
    source = xr.Dataset(
        {
            "mask": ("x", np.array([0, 7], "uint16"), {"nodata": 0}),
            "offset": ("x", np.array([1, 2], "int16"), {"add_offset": 5.0}),
            "codes": ("x", np.array([-9, -8], "int16"), {"missing_value": [-9, -8]}),
            "plain": ("x", np.array([1, 2], "uint16")),
        }
    ).assign_coords(class_label=("class", ["water", "land"]))
    decoded = source.gs.mask_and_scale()

    np.testing.assert_allclose(decoded["mask"], [np.nan, 7], equal_nan=True)
    np.testing.assert_array_equal(decoded.offset, [6, 7])
    assert np.isnan(decoded.codes).all()
    assert decoded.plain.dtype == source.plain.dtype
    assert decoded["mask"].encoding["_FillValue"] == 0
    xr.testing.assert_identical(decoded.class_label, source.class_label)


@pytest.mark.parametrize(
    "metadata", [{"scale_factor": 1}, {"scale_factor": "1", "add_offset": "0"}]
)
def test_identity_scaling_preserves_fractional_values_and_nan(metadata):
    source = xr.DataArray([2.5, np.nan, 4.25], dims="time", attrs=metadata).chunk(
        time=1
    )

    with Callback(pretask=lambda *_: pytest.fail("decoding read pixels")):
        decoded = source.gs.mask_and_scale()

    assert decoded.dtype.kind == "f"
    np.testing.assert_allclose(decoded, [2.5, np.nan, 4.25], equal_nan=True)
    assert source.attrs == metadata


def test_tree_converts_independent_nodes_without_decoding_coordinates():
    raw = build_raster(packed=True).chunk()
    tree = build_tree({"a": raw, "b/deep": raw})
    tree.attrs["title"] = "scene"
    tree.coords["site"] = "field"
    decoded = tree.gs.mask_and_scale()
    restored = decoded.gs.scale_and_fill()

    assert decoded.gs.groups == tree.gs.groups
    assert isinstance(decoded["b/deep"].red.data, da.Array)
    assert decoded.attrs["title"] == "scene"
    assert decoded.site.item() == "field"
    xr.testing.assert_equal(restored, tree)


@pytest.mark.parametrize(
    ("values", "encoding", "message"),
    [
        ([300.0], {"dtype": "uint8", "scale_factor": 1.0}, "overflow"),
        ([np.nan], {"dtype": "uint8", "scale_factor": 1.0}, "fill value"),
        ([0.0, np.nan], {"dtype": "uint8", "_FillValue": 0}, "collide"),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_integer_encoding_rejects_silent_corruption(values, encoding, message, lazy):
    physical = xr.DataArray(values, dims="x")
    if lazy:
        physical = physical.chunk(x=1)
    physical.encoding = encoding

    if lazy:
        with Callback(pretask=lambda *_: pytest.fail("encoding computed pixels")):
            stored = physical.gs.scale_and_fill()
        with pytest.raises(ValueError, match=message):
            stored.compute()
    else:
        with pytest.raises(ValueError, match=message):
            physical.gs.scale_and_fill()


def test_integer_quantization_matches_the_native_encoder():
    physical = xr.DataArray([0.5, 1.5, 2.5, np.nan], dims="x")
    physical.encoding = {"dtype": "uint8", "scale_factor": 1.0, "_FillValue": 255}
    stored = physical.gs.scale_and_fill()

    np.testing.assert_array_equal(stored, [0, 2, 2, 255])
    assert stored.dtype == np.uint8
    assert stored.attrs["_FillValue"] == 255


def test_declarations_preserve_each_other_and_combination_keeps_shared_scaling():
    raw = build_raster()[["red"]].gs.write_nodata(65535)
    raw = raw.gs.write_scale_offset(0.1, -0.1)
    assert raw.red.gs.nodata == 65535
    assert raw.red.gs.packing.scale_factor == 0.1
    cleared = raw.gs.write_scale_offset(None)
    assert cleared.red.gs.nodata == 65535
    cleared = raw.gs.write_nodata(None)
    assert cleared.red.gs.packing.scale_factor == 0.1

    left = raw.red.attrs
    right = raw.red.gs.write_nodata(0).attrs
    with pytest.warns(DroppedAttrsWarning, match="_FillValue"):
        shared = attrs.combine_attrs([left, right])
    assert shared == {"scale_factor": 0.1, "add_offset": -0.1}


def test_class_codes_can_be_masked_but_cannot_be_scaled():
    raw = xr.DataArray(np.array([1, 255], "uint8"), dims="x")
    raw = raw.gs.write_legend({1: "trees"}).gs.write_nodata(255)
    assert np.isnan(raw.gs.mask_and_scale()[1])
    with pytest.raises(ValueError, match="class codes"):
        raw.gs.write_scale_offset(0.1).gs.mask_and_scale()
