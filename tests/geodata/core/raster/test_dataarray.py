from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.conftest import build_dataset
from tests.geodata.conftest import build_raster, geobox
from tests.geodata.core.raster.conftest import optical
from tests.geodata.conftest import create_file_items
from geosave.geodata import attrs

from affine import Affine
from odc.geo.geobox import GeoBox
from tests.conftest import build_dataarray

import dask.array as da
from dask.callbacks import Callback

from geosave.geodata import open_raster


UTM = "EPSG:32633"


def test_a_band_reads_with_the_grid_trailing() -> None:
    scrambled = xr.DataArray(
        np.zeros((8, 8, 3), "uint16"),
        dims=("y", "x", "band"),
        coords={"band": ["r", "g", "b"]},
    ).odc.assign_crs(UTM)

    assert scrambled.gs.to_numpy().shape == (3, 8, 8)


def test_a_band_keeps_its_other_axes_ahead_of_the_bands() -> None:
    cube = xr.DataArray(
        np.zeros((3, 2, 8, 8), "uint16"),
        dims=("band", "time", "y", "x"),
        coords={"band": ["r", "g", "b"], "time": [0, 1]},
    ).odc.assign_crs(UTM)

    assert cube.gs.to_numpy().shape == (2, 3, 8, 8)


def test_a_band_carrying_no_band_axis_reads_without_one() -> None:
    band = optical()["B04"]

    assert band.gs.to_numpy().shape == (2, 8, 8)
    assert optical()[["B04"]].gs.to_numpy().shape == (2, 1, 8, 8)


def test_a_band_converts_without_gaining_a_band_axis(torch) -> None:
    tensor = optical().B04.gs.to_tensor()

    assert tensor.dtype is torch.uint16
    assert tuple(tensor.shape) == (2, 8, 8)


def test_to_tensor_copies_read_only_pixels(torch) -> None:
    band = optical().B04
    band.values.flags.writeable = False
    before = band.values.copy()

    tensor = band.gs.to_tensor()
    tensor.fill_(0)

    np.testing.assert_array_equal(band.values, before)


def test_to_dataset_gives_every_band_the_arrays_attrs() -> None:
    box = geobox()
    built = build_dataset(
        {
            "B04": (("y", "x"), np.zeros(box.shape, "uint16")),
            "B03": (("y", "x"), np.zeros(box.shape, "uint16")),
        },
        box,
    ).gs.write_nodata(0)
    built.attrs = {"title": "S2"}

    split = built.gs.to_array().gs.to_dataset(dim="band")

    assert split.attrs == {}
    assert split.B04.attrs == split.B03.attrs == {"_FillValue": 0}


def test_statistics_reads_a_lazy_source_once() -> None:
    import dask.array as da
    from dask import delayed

    reads = []

    @delayed
    def pixels():
        reads.append(1)
        return np.array([[1.0, 2.0], [3.0, 0.0]])

    values = da.from_delayed(pixels(), shape=(2, 2), dtype="float64")
    band = xr.DataArray(values, dims=("y", "x"), name="red", attrs={"nodata": 0})
    summary = band.gs.statistics()

    assert isinstance(summary, pd.DataFrame)
    assert summary.loc["red", "minimum"] == 1.0
    assert summary.loc["red", "maximum"] == 3.0
    assert summary.loc["red", "mean"] == 2.0
    assert summary.loc["red", "stddev"] == pytest.approx(np.sqrt(2 / 3))
    assert summary.loc["red", "valid_percent"] == 75.0
    assert len(reads) == 1


def test_split_defaults_to_bands_and_preserves_pixel_interpretation():
    image = build_raster(packed=True).gs.to_array()
    split = image.gs.to_dataset()
    assert list(split.data_vars) == image.band.values.tolist()
    assert split.red.attrs == image.attrs
    assert split.gs.geobox == image.gs.geobox


def test_multiband_array_statistics_keep_band_order_and_shared_nodata() -> None:
    source = xr.Dataset(
        {
            "nir": (("y", "x"), [[0.0, 10.0, 20.0]]),
            "red": (("y", "x"), [[0.0, 1.0, 3.0]]),
        }
    ).gs.write_nodata(0)

    table = source.gs.to_array().gs.statistics()

    pd.testing.assert_frame_equal(table, source.gs.statistics())
    assert list(table.index) == ["nir", "red"]
    assert table.loc["nir", "mean"] == 15.0
    assert table.loc["red", "mean"] == 2.0
    assert table.loc["red", "valid_percent"] == pytest.approx(200 / 3)


def test_multiband_statistics_refuse_an_absent_band() -> None:
    image = xr.DataArray(
        [[1.0, 3.0], [np.nan, np.nan]],
        dims=("band", "x"),
        coords={"band": ["red", "nir"]},
    )

    with pytest.raises(ValueError, match="nir holds no present pixel"):
        image.gs.statistics()


def test_an_edit_on_the_stacked_array_reaches_every_band():
    source = build_dataset(
        {
            "red": (("y", "x"), np.ones(geobox().shape, "int16")),
            "nir": (("y", "x"), np.ones(geobox().shape, "int16")),
        },
        geobox(),
    ).gs.write_nodata(0)
    restored = source.gs.to_array().gs.write_nodata(-1).gs.to_dataset(dim="band")
    assert restored.gs.nodata == {"red": -1, "nir": -1}


def test_a_band_indexes_as_its_one_variable_raster(tmp_path: Path) -> None:

    band = build_raster(times=2)["red"].isel(time=0)
    path = band.gs.to_cog(tmp_path / "red.tif")

    (item,) = create_file_items(path, id_format="red")

    assert item.id == "red"
    assert list(item.assets) == ["red"]
    assert item.assets["red"].href == str(path)


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
    assert shared.gs.nodata == {"red": 0, "nir": 0}


def _band(dtype: str) -> xr.DataArray:
    return xr.DataArray(np.zeros(2, dtype), dims="x")


def test_nan_marks_absence_in_a_float_band() -> None:
    assert np.isnan(_band("float32").gs.write_nodata(float("nan")).gs.nodata)


def test_class_codes_must_fit_the_bands_dtype() -> None:
    with pytest.raises(ValueError, match="uint8"):
        _band("uint8").gs.write_flags({0: "bg", 300: "palm"})


def _coded_band(**band_attrs: object) -> xr.DataArray:
    return xr.DataArray(np.zeros((2, 2), "uint8"), dims=("y", "x"), attrs=band_attrs)


def test_a_legend_marks_class_codes() -> None:
    coded = _coded_band().gs.write_flags({0: "bg", 1: "palm"})

    assert coded.gs.is_categorical


def test_a_colour_map_marks_class_codes_without_flags() -> None:
    assert _coded_band(color_map='{"1": "#00ff00"}').gs.is_categorical


def test_a_plain_integer_band_is_a_measurement() -> None:
    assert not _coded_band(units="1").gs.is_categorical


def test_cf_flags_with_meanings_mark_class_codes() -> None:
    band = _coded_band(flag_masks=[1, 2], flag_meanings="cloud shadow")

    assert band.gs.is_categorical


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


def test_to_dataset_wraps_a_named_multiband_array_lazily() -> None:
    import dask.array as da

    source = xr.DataArray(
        da.ones((2, 3, 4), chunks=(1, 3, 4)),
        dims=("band", "y", "x"),
        coords={"band": ["red", "nir"]},
        name="reflectance",
        attrs={"units": "1"},
    )

    wrapped = source.to_dataset()
    split = source.gs.to_dataset(dim="band")

    assert list(wrapped.data_vars) == ["reflectance"]
    xr.testing.assert_identical(wrapped.reflectance, source)
    assert isinstance(wrapped.reflectance.data, da.Array)
    assert list(split.data_vars) == ["red", "nir"]
    assert isinstance(split.red.data, da.Array)
    assert split.red.attrs == split.nir.attrs == {"units": "1"}


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
    from dask.callbacks import Callback

    band = landcover({1: "water"}, {1: "#0000ff"})
    band = band.assign_coords(time=np.datetime64("2025-01-01")).chunk({"x": 1})
    with Callback(pretask=lambda *_: pytest.fail("colorize computed pixels")):
        colored = band.gs.colorize()
    assert colored.chunks is not None
    xr.testing.assert_identical(colored.time, band.time)
    assert colored.dtype == np.float32
    np.testing.assert_array_equal(colored.sel(band="blue").compute()[0], np.ones(3))
    assert np.isnan(colored.compute().values[:, 1, 2]).all()


def test_colorize_draws_an_uncoloured_class_in_its_palette_colour() -> None:
    from geosave.geodata.utils.color import CATEGORY_PALETTE, parse_color

    colored = landcover({1: "water"}).gs.colorize().compute()

    red, green, blue = (channel / 255 for channel in parse_color(CATEGORY_PALETTE[0]))
    named = np.isfinite(colored.sel(band="red").values)
    assert colored.sel(band="red").values[named][0] == pytest.approx(red)


def test_an_unnamed_array_keeps_name_coordinates_and_original_dtype():
    raw = build_raster(packed=True).red.rename(None)
    decoded = raw.gs.mask_and_scale()

    assert decoded.name is None
    assert decoded.encoding["dtype"] == raw.dtype
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


@pytest.mark.parametrize("fill", [-3.40282e38, 0.1])
def test_declared_float_nodata_decodes_without_repair(fill):
    source = xr.DataArray(np.array([fill, 2], "float32"), dims="x").chunk(x=1)
    before = source.copy(deep=True)

    with Callback(pretask=lambda *_: pytest.fail("decoding read pixels")):
        declared = source.gs.write_nodata(fill)
        decoded = declared.gs.mask_and_scale()

    np.testing.assert_allclose(decoded, [np.nan, 2], equal_nan=True)
    assert declared.attrs["_FillValue"] == source.dtype.type(fill)
    xr.testing.assert_identical(source, before)


def test_unsigned_cf_pixels_decode_without_a_packing_group():
    source = xr.DataArray(
        da.from_array(np.array([-1, 1], "int16"), chunks=1),
        dims="x",
        attrs={"_Unsigned": "true"},
        name="red",
    )
    before = source.copy(deep=True)
    with Callback(pretask=lambda *_: pytest.fail("decoding read pixels")):
        decoded = source.gs.mask_and_scale()
    assert decoded.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(decoded, [65535, 1])
    assert decoded.encoding["_Unsigned"] == "true"
    xr.testing.assert_identical(source, before)


def test_class_codes_keep_their_values_and_legend_after_masking():
    raw = xr.DataArray(np.array([1, 255], "uint8"), dims="x")
    raw = raw.gs.write_flags({1: "trees"}).gs.write_nodata(255)
    assert np.isnan(raw.gs.mask_and_scale()[1])
    assert raw.gs.mask_and_scale().values[0] == 1
    assert raw.gs.mask_and_scale().gs.flags == raw.gs.flags


@pytest.mark.parametrize("method", ["to_cog", "to_gtiff"])
@pytest.mark.parametrize("multiband", [False, True])
def test_array_file_roundtrip_preserves_packing_grid_and_values(
    method, multiband, tmp_path
):
    source = build_raster(times=0, packed=True).red.chunk()
    if multiband:
        source = xr.concat(
            [source, source], dim=xr.IndexVariable("band", ["red", "nir"])
        )
    original = source.copy(deep=True)
    decoded = source.gs.mask_and_scale()
    path = tmp_path / "image.tif"
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        pending = getattr(decoded.gs, method)(path, compute=False)
    assert tasks == []
    assert not path.exists()
    saved = pending.compute()
    assert saved == path
    with open_raster(saved) as restored:
        assert restored.gs.geobox == source.gs.geobox
        for name, variable in restored.data_vars.items():
            expected = source.sel(band=name) if multiband else source
            np.testing.assert_array_equal(variable, expected)
            assert variable.dtype == expected.dtype
            assert variable.gs.packing == expected.gs.packing
    xr.testing.assert_identical(source, original)
    with pytest.raises(FileExistsError):
        getattr(decoded.gs, method)(path)
    assert getattr(decoded.gs, method)(path, overwrite=True) == path


@pytest.mark.parametrize("method", ["to_cog", "to_gtiff"])
def test_array_writer_rejects_unselected_observations_and_split_options(
    method, tmp_path
):
    source = build_raster(times=2).red.chunk()
    with pytest.raises(ValueError, match="one instant"):
        getattr(source.gs, method)(tmp_path / "dates.tif", compute=False)
    with pytest.raises(TypeError, match="split_item"):
        getattr(source.gs, method)(tmp_path / "dates.tif", split_item=("time",))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bands", [None, [1, 2]])
def test_array_file_accepts_unnamed_and_numbered_bands(tmp_path, bands):
    source = build_raster(times=0).red.rename(None)
    if bands is not None:
        source = source.expand_dims(band=bands)
    path = source.gs.to_gtiff(tmp_path / "bands.tif")
    with open_raster(path) as restored:
        assert list(restored.data_vars) == (["asset"] if bands is None else ["1", "2"])
        for variable in restored.data_vars.values():
            np.testing.assert_array_equal(variable, build_raster(times=0).red)
