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

    split = built.gs.to_dataarray().gs.to_dataset(dim="band")

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


def test_to_dataset_names_an_unstacked_band_after_itself() -> None:
    box = geobox()
    built = build_dataset(
        {"B04": (("y", "x"), np.zeros(box.shape, "uint16"))}, box
    ).gs.write_scale_offset(1e-4)

    split = built["B04"].gs.to_dataset()

    assert split.gs.variables == ("B04",)
    assert split["B04"].attrs["scale_factor"] == pytest.approx(1e-4)


def test_to_dataset_refuses_bands_it_cannot_name() -> None:
    box = geobox()
    stacked = build_dataset(
        {"B04": (("y", "x"), np.zeros(box.shape, "uint16"))}, box
    ).gs.to_dataarray()

    with pytest.raises(ValueError, match="labels none of them"):
        stacked.drop_vars("band").gs.to_dataset(dim="band")


def test_multiband_array_statistics_keep_band_order_and_shared_nodata() -> None:
    source = xr.Dataset(
        {
            "nir": (("y", "x"), [[0.0, 10.0, 20.0]]),
            "red": (("y", "x"), [[0.0, 1.0, 3.0]]),
        }
    ).gs.write_nodata(0)

    table = source.gs.to_dataarray().gs.statistics()

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
    restored = source.gs.to_dataarray().gs.write_nodata(-1).gs.to_dataset(dim="band")
    assert restored.gs.nodata == {"red": -1, "nir": -1}


def test_a_band_indexes_as_its_one_variable_raster(tmp_path: Path) -> None:

    band = build_raster(times=2)["red"].isel(time=0)
    paths = band.gs.to_cog(tmp_path, format_paths="red.tif")

    (item,) = create_file_items(paths, id_format="red")

    assert item.id == "red"
    assert list(item.assets) == ["red"]
    assert item.assets["red"].href == str(paths[0])


def test_to_dataset_recovers_only_the_attrs_every_band_shared() -> None:
    scene = (
        build_raster()
        .gs.write_scale_offset(1e-4)
        .gs.write_nodata(0, target="red")
        .gs.write_nodata(65535, target="nir")
    )

    decoded = scene.gs.mask_and_scale()
    restored = decoded.gs.to_dataarray().gs.to_dataset(dim="band")

    xr.testing.assert_allclose(restored.red, decoded.red)
    xr.testing.assert_allclose(restored.nir, decoded.nir)
    assert restored.red.gs.packing is None
    assert restored.nir.gs.packing is None
    shared = scene.gs.write_nodata(0).gs.to_dataarray().gs.to_dataset(dim="band")
    assert shared.gs.nodata == {"red": 0, "nir": 0}


def _band(dtype: str) -> xr.DataArray:
    return xr.DataArray(np.zeros(2, dtype), dims="x")


def test_nan_marks_absence_in_a_float_band_only() -> None:
    assert np.isnan(_band("float32").gs.write_nodata(float("nan")).gs.nodata)
    with pytest.raises(ValueError, match="uint16"):
        _band("uint16").gs.write_nodata(float("nan"))


def test_a_fill_overflowing_a_float_band_is_refused() -> None:
    with pytest.raises(ValueError, match="inf"):
        _band("float32").gs.write_nodata(1e39)


def test_an_integer_band_refuses_a_fill_it_would_wrap() -> None:
    with pytest.raises(ValueError, match="65535"):
        _band("uint16").gs.write_nodata(-1)


def test_class_codes_must_fit_the_bands_dtype() -> None:
    with pytest.raises(ValueError, match="uint8"):
        _band("uint8").gs.write_legend({0: "bg", 300: "palm"})


def _coded_band(**band_attrs: object) -> xr.DataArray:
    return xr.DataArray(np.zeros((2, 2), "uint8"), dims=("y", "x"), attrs=band_attrs)


def test_a_legend_marks_class_codes() -> None:
    coded = _coded_band().gs.write_legend({0: "bg", 1: "palm"})

    assert coded.gs.is_categorical


def test_a_palette_band_marks_class_codes_without_a_legend() -> None:
    assert _coded_band(colorinterp="palette").gs.is_categorical


@pytest.mark.parametrize("colorinterp", ["gray", "red"])
def test_other_interpretations_say_nothing(colorinterp: str) -> None:
    assert not _coded_band(colorinterp=colorinterp).gs.is_categorical


def test_a_plain_integer_band_is_a_measurement() -> None:
    assert not _coded_band(units="1").gs.is_categorical


def test_a_half_spelled_listing_still_marks_class_codes() -> None:
    # A guard answers yes or no; it must not raise on attrs it cannot parse.
    assert _coded_band(flag_masks=[1, 2]).gs.is_categorical


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

    wrapped = source.gs.to_dataset()
    split = source.gs.to_dataset(dim="band")

    assert list(wrapped.data_vars) == ["reflectance"]
    xr.testing.assert_identical(wrapped.reflectance, source)
    assert isinstance(wrapped.reflectance.data, da.Array)
    assert list(split.data_vars) == ["red", "nir"]
    assert isinstance(split.red.data, da.Array)
    assert split.red.attrs == split.nir.attrs == {"units": "1"}


def test_expand_rasters_splits_by_position_with_labels_as_scalar_coords():
    import dask.array as da
    from dask.callbacks import Callback

    array = xr.DataArray(
        da.arange(24).reshape((2, 3, 2, 2)),
        dims=("time", "level", "y", "x"),
        coords={"time": [10, 20]},
        name="air",
    )
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        slices = array.gs.expand_rasters(["time", "level"])
    assert tasks == []
    assert isinstance(slices, tuple)
    assert len(slices) == 6
    assert [piece.time.item() for piece in slices] == [10, 10, 10, 20, 20, 20]
    # A dimension without coordinates labels each slice by its position.
    assert [piece.level.item() for piece in slices] == [0, 1, 2, 0, 1, 2]
    np.testing.assert_array_equal(slices[4], array.isel(time=1, level=1))
    assert slices[4].dims == ("y", "x")
    assert slices[0].chunks is not None
    (whole,) = array.gs.expand_rasters(())
    xr.testing.assert_identical(whole, array)


def test_expand_rasters_keeps_repeated_labels_as_separate_slices():
    array = xr.DataArray(np.ones((3, 2, 2)), dims=("time", "y", "x"))
    array = array.assign_coords(time=[1, 1, 2])
    assert [piece.time.item() for piece in array.gs.expand_rasters("time")] == [1, 1, 2]


@pytest.mark.parametrize(
    ("dim", "message"),
    [("levle", "absent"), ("y", "spatial"), (("time", "time"), "repeat")],
)
def test_expand_rasters_rejects_unsplittable_dimensions(dim, message):
    array = build_raster(times=2).red
    with pytest.raises(ValueError, match=message):
        array.gs.expand_rasters(dim)
