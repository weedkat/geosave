from __future__ import annotations


import numpy as np
import pytest
import xarray as xr

from tests.conftest import build_dataset
from tests.geodata.conftest import build_raster, geobox
from tests.geodata.core.raster.conftest import optical
from geosave.geodata import attrs

from affine import Affine
from odc.geo.geobox import GeoBox
from tests.conftest import build_dataarray

from dask.callbacks import Callback


def test_a_band_reads_with_the_grid_trailing() -> None:
    box = geobox()
    scrambled = build_dataset(
        {"pixels": (("y", "x", "band"), np.zeros((8, 8, 3), "uint16"))}, box
    )["pixels"].assign_coords(band=["r", "g", "b"])

    pixels = scrambled.gs.to_numpy()

    assert pixels.shape == (3, 8, 8)


def test_a_band_carrying_no_band_axis_reads_without_one() -> None:
    band = optical()["B04"]

    pixels = band.gs.to_numpy()
    stacked = optical()[["B04"]].gs.to_numpy()

    assert pixels.shape == (2, 8, 8)
    assert stacked.shape == (2, 1, 8, 8)


def test_a_band_converts_without_gaining_a_band_axis(torch) -> None:
    tensor, context = optical().B04.gs.to_tensor(context={"transform": "transform", "crs": "crs"})

    assert list(context) == ["transform", "crs"]

    assert tensor.dtype is torch.uint16
    assert tuple(tensor.shape) == (2, 8, 8)


def test_split_defaults_to_bands_and_preserves_pixel_interpretation():
    image = build_raster(packed=True).gs.to_array()
    split = image.gs.to_dataset()
    assert list(split.data_vars) == image.band.values.tolist()
    assert split.red.attrs == image.attrs
    assert split.gs.geobox == image.gs.geobox


def test_to_dataset_recovers_only_the_attrs_every_band_shared() -> None:
    scene = (
        build_raster()
        .gs.write_packing(attrs.CFPacking(scale_factor=1e-4))
        .gs.write_nodata(0, target=["red"])
        .gs.write_nodata(65535, target=["nir"])
    )

    decoded = scene.gs.mask_and_scale()
    restored = decoded.gs.to_array().gs.to_dataset(dim="band")

    xr.testing.assert_allclose(restored.red, decoded.red)
    xr.testing.assert_allclose(restored.nir, decoded.nir)
    assert restored.red.gs.packing is None
    assert restored.nir.gs.packing is None
    shared = scene.gs.write_nodata(0).gs.to_array().gs.to_dataset(dim="band")
    assert shared.red.gs.nodata == shared.nir.gs.nodata == 0


def _band(dtype: str) -> xr.DataArray:
    return xr.DataArray(np.zeros(2, dtype), dims="x")


def test_class_codes_must_fit_the_bands_dtype() -> None:
    with pytest.raises(ValueError, match="uint8"):
        _band("uint8").gs.write_flags({0: "bg", 300: "palm"})


def _coded_band(**band_attrs: object) -> xr.DataArray:
    return xr.DataArray(np.zeros((2, 2), "uint8"), dims=("y", "x"), attrs=band_attrs)


def test_a_legend_marks_class_codes() -> None:
    coded = _coded_band().gs.write_flags({0: "bg", 1: "palm"})

    assert coded.gs.is_categorical


def test_a_plain_integer_band_is_a_measurement() -> None:
    assert not _coded_band(units="1").gs.is_categorical


@pytest.mark.parametrize(
    ("method", "suffix"), [("to_zarr", ".zarr"), ("to_netcdf", ".nc")]
)
def test_band_store_returns_native_path_after_deferred_write(tmp_path, method, suffix):
    from dask.delayed import Delayed
    from upath import UPath
    from geosave.geodata import open_raster

    band = build_raster(times=2).red
    destination = tmp_path / f"red{suffix}"
    pending = getattr(band.gs, method)(destination, compute=False)
    assert isinstance(pending, Delayed)
    assert not destination.exists()
    saved = pending.compute()
    assert isinstance(saved, UPath)
    assert saved == destination
    with open_raster(saved) as restored:
        np.testing.assert_array_equal(restored.red, band)
        assert restored.gs.geobox == band.gs.geobox


GRID = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")


def landcover(classes=None, colors=None) -> xr.DataArray:
    """Build a class band, one pixel of it naming no class."""
    codes = np.array([[1, 1, 1], [1, 1, 9]], dtype="uint8")
    band = build_dataarray(codes, GRID, dims=("y", "x"))
    return band if classes is None else band.gs.write_flags(classes, colors)


def test_colorize_bakes_each_class_colour_onto_the_grid() -> None:
    band = landcover({1: "water"}, {1: "#0000ff"})

    colored = band.gs.colorize()

    assert colored.dims == ("band", "y", "x")
    assert list(colored.band.values) == ["red", "green", "blue"]
    assert colored.gs.geobox == GRID
    np.testing.assert_array_equal(colored.sel(band="blue")[0], np.ones(3))
    assert np.isnan(colored.values[:, 1, 2]).all()


def test_colorize_refuses_a_band_without_classes() -> None:
    with pytest.raises(ValueError, match="lists no classes"):
        landcover(None).gs.colorize()


def test_colorize_keeps_pixels_lazy_and_preserves_scalar_coordinates() -> None:

    band = landcover({1: "water"}, {1: "#0000ff"})
    band = band.assign_coords(time=np.datetime64("2025-01-01")).chunk({"x": 1})
    with Callback(pretask=lambda *_: pytest.fail("colorize computed pixels")):
        colored = band.gs.colorize()
    assert colored.chunks is not None
    xr.testing.assert_identical(colored.time, band.time)
    assert colored.dtype == np.float32
    np.testing.assert_array_equal(colored.sel(band="blue").compute()[0], np.ones(3))
    assert np.isnan(colored.compute().values[:, 1, 2]).all()


def test_class_codes_keep_their_values_and_legend_after_masking():
    raw = xr.DataArray(np.array([1, 255], "uint8"), dims="x")
    raw = raw.gs.write_flags({1: "trees"}).gs.write_nodata(255)
    assert np.isnan(raw.gs.mask_and_scale()[1])
    assert raw.gs.mask_and_scale().values[0] == 1
    assert raw.gs.mask_and_scale().gs.flags == raw.gs.flags


def test_stacking_and_splitting_keep_dataset_attrs_apart_from_pixel_meaning() -> None:
    scene = build_raster(times=1, packed=True).assign_attrs(
        title="scene", **attrs.StacCollection(collection={"id": "s2"}).to_attrs()
    )
    scene["red"].attrs["long_name"] = "Red"

    restored = scene.gs.to_array().gs.to_dataset("band")

    assert restored.attrs == scene.attrs
    assert restored.red.gs.packing == scene.red.gs.packing
    # A variable's own attrs ride along `band` as coords and come back to it.
    assert restored.red.attrs["long_name"] == "Red"
    assert "long_name" not in restored.nir.attrs
    assert "title" not in restored.red.attrs
