"""Encoded and decoded rasters both save without double scaling."""

import numpy as np
import pytest
import xarray as xr
import zarr

from geosave.geodata import io
from geosave.geodata.io.formats import geotiff, netcdf
from geosave.geodata.io.formats import zarr as zarr_io
from tests.geodata.conftest import build_raster, create_file_items


@pytest.mark.parametrize(
    ("suffix", "writer"),
    [
        ("tif", geotiff.write_cog),
        ("nc", netcdf.write_netcdf),
        ("zarr", zarr_io.write_zarr),
    ],
)
@pytest.mark.parametrize("decoded", [False, True])
def test_both_representations_round_trip_per_variable_packing(
    tmp_path, suffix, writer, decoded
):
    source = build_raster().gs.write_nodata(65535)
    source = source.gs.write_scale_offset(0.0001, -0.1, target="red")
    source = source.gs.write_scale_offset(0.0002, -0.2, target="nir")
    for name in source.data_vars:
        source[name].data = np.array([[0, 2000], [3000, 65535]], "uint16")
    expected = source.gs.mask_and_scale()
    written = expected if decoded else source
    path = tmp_path / f"scene.{suffix}"

    writer(written.chunk({"x": 1, "y": 1}), path)

    with io.open_raster(path, mask_and_scale=False) as restored:
        assert restored.gs.geobox == source.gs.geobox
        for name in source.data_vars:
            assert restored[name].dtype == source[name].dtype
            np.testing.assert_array_equal(restored[name], source[name])
            assert "scale_factor" not in restored[name].encoding
            assert (
                restored[name].attrs["scale_factor"]
                == source[name].attrs["scale_factor"]
            )
            assert restored[name].attrs["_FillValue"] == 65535

    with io.open_raster(path, mask_and_scale=True) as restored:
        for name in source.data_vars:
            np.testing.assert_allclose(
                restored[name], expected[name], atol=1e-7, equal_nan=True
            )
            assert "scale_factor" not in restored[name].attrs
            assert (
                restored[name].encoding["scale_factor"]
                == source[name].attrs["scale_factor"]
            )
            assert restored[name].encoding["_FillValue"] == 65535

    # Rewriting an externally decoded read must also encode exactly once.
    rewritten = tmp_path / f"rewritten.{suffix}"
    with io.open_raster(path, mask_and_scale=True) as loaded:
        writer(loaded, rewritten)
    with io.open_raster(rewritten, mask_and_scale=False) as restored:
        np.testing.assert_array_equal(restored.red, source.red)

    if suffix == "zarr":
        assert zarr.open_group(path)["red"].fill_value == 65535


def test_nodata_only_decoding_restores_integer_geotiff_storage(tmp_path):
    raw = build_raster().gs.write_nodata(65535)
    raw.red.data = np.array([[0, 1000], [2000, 65535]], "uint16")
    decoded = raw.gs.mask_and_scale()
    path = tmp_path / "nodata.tif"

    geotiff.write_cog(decoded, path)
    with io.open_raster(path) as stored:
        assert stored.red.dtype == raw.red.dtype
        assert stored.red.gs.nodata == 65535
        np.testing.assert_array_equal(stored.red, raw.red)


def test_explicit_output_encoding_uses_current_physical_values(tmp_path):
    source = build_raster(packed=True)
    result = source.gs.mask_and_scale() * 2
    path = tmp_path / "new.nc"
    settings = {}
    for name in result.data_vars:
        settings[name] = {
            "dtype": "uint16",
            "scale_factor": 0.0002,
            "_FillValue": 65535,
        }

    netcdf.write_netcdf(result, path, encoding=settings)
    with io.open_raster(path, mask_and_scale=True) as restored:
        xr.testing.assert_allclose(restored, result)


def test_asset_window_masks_float32_missing_codes_without_scaling(tmp_path):
    from geosave.geodata.stac.publish import read_scene_pixels

    source = build_raster()[["red"]].astype("float32")
    source.red.data = np.array([[0.1, 2], [3, 4]], dtype="float32")
    source.red.attrs = {"missing_value": 0.1, "scale_factor": 2.0, "add_offset": 10.0}
    path = tmp_path / "float.tif"
    geotiff.write_cog(source, path)
    entry = next(iter(create_file_items({"red": path})))

    window = read_scene_pixels(entry, source.gs.geobox)
    decoded = source.gs.mask_and_scale()

    assert window.gs.geobox == source.gs.geobox
    np.testing.assert_array_equal(np.isnan(window.red), np.isnan(decoded.red))
    np.testing.assert_allclose(window.red, [[np.nan, 2], [3, 4]], equal_nan=True)
    np.testing.assert_allclose(decoded.red, [[np.nan, 14], [16, 18]], equal_nan=True)
