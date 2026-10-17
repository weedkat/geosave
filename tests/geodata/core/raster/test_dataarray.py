from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from geosave.geodata import GeoAnchor
from geosave.geodata.core.raster.create import array, raster
from geosave.geodata.transform.vector import vectorize
from tests.geodata.conftest import build_raster, geobox
from tests.geodata.core.raster.conftest import optical
from tests.geodata.conftest import flat

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


def test_a_band_casts_the_way_a_raster_does() -> None:
    band = optical()["B04"]

    assert band.gs.to_numpy(dtype="float32").dtype == np.float32


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


def test_unnamed_array_has_no_variable_identity(raster: xr.Dataset) -> None:
    assert raster.red.rename(None).gs.variables == ()


def test_to_raster_gives_every_band_the_arrays_attrs() -> None:
    box = geobox()
    built = raster(
        {
            "B04": (("y", "x"), np.zeros(box.shape, "uint16")),
            "B03": (("y", "x"), np.zeros(box.shape, "uint16")),
        },
        box,
    ).gs.write_nodata(0)
    built.attrs = {"title": "S2"}

    split = built.gs.to_array().gs.to_raster()

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


def test_to_raster_names_an_unstacked_band_after_itself() -> None:
    box = geobox()
    built = raster(
        {"B04": (("y", "x"), np.zeros(box.shape, "uint16"))}, box
    ).gs.write_packing(1e-4)

    split = built["B04"].gs.to_raster()

    assert split.gs.variables == ("B04",)
    assert split["B04"].attrs["scale_factor"] == pytest.approx(1e-4)


def test_to_raster_refuses_bands_it_cannot_name() -> None:
    box = geobox()
    stacked = raster(
        {"B04": (("y", "x"), np.zeros(box.shape, "uint16"))}, box
    ).gs.to_array()

    with pytest.raises(ValueError, match="labels none of them"):
        stacked.drop_vars("band").gs.to_raster()


def test_statistics_summarise_only_the_present_pixels() -> None:
    box = geobox()
    pixels = np.arange(64, dtype="uint16").reshape(box.shape)
    band = raster({"red": (("y", "x"), pixels)}, box, nodata=0)["red"]

    summary = band.gs.statistics()

    # GDAL summarises the present pixels, so 0 is absence rather than a reading.
    assert summary.loc["red", "minimum"] == 1.0
    assert summary.loc["red", "mean"] == pytest.approx(32.0)
    assert summary.loc["red", "valid_percent"] == pytest.approx(98.4375)


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


@pytest.mark.parametrize("name", [None, "red"])
def test_single_array_statistics_use_the_array_name(name) -> None:
    band = xr.DataArray([1.0, np.nan, 3.0], dims="x", name=name)

    table = band.gs.statistics()

    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == [name]
    assert table.iloc[0]["mean"] == 2.0
    assert table.iloc[0]["valid_percent"] == pytest.approx(200 / 3)


def test_multiband_statistics_refuse_an_absent_band() -> None:
    image = xr.DataArray(
        [[1.0, 3.0], [np.nan, np.nan]],
        dims=("band", "x"),
        coords={"band": ["red", "nir"]},
    )

    with pytest.raises(ValueError, match="nir holds no present pixel"):
        image.gs.statistics()


def test_multiband_statistics_require_band_labels() -> None:
    image = xr.DataArray([[1, 3], [10, 20]], dims=("band", "x"))

    with pytest.raises(ValueError, match="labels none"):
        image.gs.statistics()


def test_statistics_refuse_a_band_holding_no_present_pixel() -> None:
    box = geobox()
    band = raster({"red": (("y", "x"), np.zeros(box.shape, "uint16"))}, box, nodata=0)[
        "red"
    ]

    with pytest.raises(ValueError, match="no present pixel"):
        band.gs.statistics()


def test_an_edit_on_the_stacked_array_reaches_every_band():
    source = raster(
        {
            "red": (("y", "x"), np.ones(geobox().shape, "int16")),
            "nir": (("y", "x"), np.ones(geobox().shape, "int16")),
        },
        geobox(),
    ).gs.write_nodata(0)
    restored = source.gs.to_array().gs.write_nodata(-1).gs.to_raster()
    assert restored.gs.nodata == {"red": -1, "nir": -1}


def test_a_band_indexes_as_its_one_variable_raster(tmp_path: Path) -> None:
    from geosave.geodata.stac import create_items

    band = build_raster(times=2)["red"].isel(time=0)
    (scene,) = band.gs.to_cog(tmp_path / "red.tif")
    (file,) = scene.written.values()

    (item,) = create_items(scene, id_format="red")

    assert item.id == "red"
    assert list(item.assets) == ["red"]
    assert item.assets["red"].href == str(file)


def test_a_dated_band_writes_through_the_raster_template(tmp_path: Path) -> None:
    band = build_raster(times=2)["red"]

    paths = flat(band.gs.to_cog(f"{tmp_path}/red_{{time:%Y%m%d}}.tif"))

    assert [Path(f) for f in paths] == [
        tmp_path / "red_20250601.tif",
        tmp_path / "red_20250602.tif",
    ]


def test_to_raster_recovers_only_the_attrs_every_band_shared() -> None:
    scene = (
        build_raster()
        .gs.write_packing(1e-4)
        .gs.write_nodata(0, target="red")
        .gs.write_nodata(65535, target="nir")
    )

    restored = scene.gs.to_array().gs.to_raster()

    assert restored.gs.packing == scene.gs.packing
    assert restored.gs.nodata == {"red": None, "nir": None}
    shared = scene.gs.write_nodata(0).gs.to_array().gs.to_raster()
    assert shared.gs.nodata == {"red": 0, "nir": 0}


def test_vectorize_creates_one_row_per_connected_flag_region(raster) -> None:
    flags = array(
        np.array([[1, 0, 1], [1, 0, 0], [2, 2, 0]], dtype="uint8"),
        GeoAnchor.from_bbox(
            raster.gs.geobox.boundingbox,
            shape=(3, 3),
            crs=raster.gs.crs,
        ).geobox,
        dims=("y", "x"),
        nodata=0,
    )

    vector = flags.gs.vectorize(value_name="class_id")

    assert list(vector.class_id) == [1, 1, 2]
    assert vector.gs.crs == flags.gs.crs
    assert vector.geometry.is_valid.all()


def test_vectorize_preserves_the_value_schema_when_every_pixel_is_nodata(
    raster,
) -> None:
    flags = array(
        np.zeros(raster.gs.geobox.shape, dtype="uint8"),
        raster.gs.geobox,
        dims=("y", "x"),
        nodata=0,
    )

    vector = flags.gs.vectorize(value_name="class_id")
    restored = vector.gs.rasterize(
        like=flags,
        column="class_id",
        fill=0,
        dtype="uint8",
    )

    assert list(vector) == ["class_id", "geometry"]
    assert vector.class_id.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(restored.values, np.zeros(flags.shape, "uint8"))


def test_vector_method_delegates_to_vector_transform(raster) -> None:
    flags = raster.red.astype("uint8")

    direct = vectorize(flags)
    through_method = flags.gs.vectorize()

    assert direct.equals(through_method)


def test_vectorize_computes_dask_flags_explicitly(raster) -> None:
    flags = raster.red.chunk({"y": 1, "x": 1})
    started: list[object] = []

    with Callback(pretask=lambda key, *_: started.append(key)):
        flags.gs.vectorize()

    assert started


def test_vectorize_requires_one_spatial_plane(raster) -> None:
    flags = raster.red.expand_dims(time=["2025-01-01"])

    with pytest.raises(ValueError, match="two-dimensional"):
        flags.gs.vectorize()


def test_vectorize_rejects_unsupported_values(raster) -> None:
    flags = raster.red.astype("complex64")

    with pytest.raises(ValueError, match="dtype"):
        flags.gs.vectorize()


def test_vectorize_rejects_same_shape_mask_on_another_grid(raster) -> None:
    shifted = raster.red.assign_coords(x=raster.x + 10)

    with pytest.raises(ValueError, match="exact flags grid"):
        raster.red.gs.vectorize(mask=shifted)


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


def test_a_float_band_holds_class_codes_exactly() -> None:
    labelled = _band("float32").gs.write_legend({0: "bg", 1: "palm"})

    assert labelled.gs.legend.classes == {0: "bg", 1: "palm"}


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
