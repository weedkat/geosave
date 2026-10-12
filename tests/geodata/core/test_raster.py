from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo import CRS

from geosave_engine.geodata import GeoArray, GeoRaster, GeoStack
from geosave_engine.geodata.core.constructors import raster
from geosave_engine.geodata.io import gdal

from tests.geodata.conftest import build_raster, geobox

UTM = "EPSG:32633"


@pytest.mark.parametrize(
    ("times", "template", "count"),
    [
        (0, "scene.tif", 1),
        (0, "{band}.tif", 2),
        (1, "scene.tif", 1),
        (2, "{time:%Y%m%d}.tif", 2),
        (2, "{time:%Y%m%d}_{band}.tif", 4),
    ],
)
def test_cog_export_returns_readable_files(
    tmp_path: Path, times: int, template: str, count: int
) -> None:
    written = build_raster(times=times)

    paths = [file.href for file in written.gs.to_cog(f"{tmp_path}/{template}")]

    assert len(paths) == count
    for path in paths:
        with gdal.read(path) as restored:
            expected = written.sel(time=restored.time) if times else written
            assert restored.gs.geobox == written.gs.geobox
            for name in restored.data_vars:
                np.testing.assert_array_equal(restored[name], expected[name])


def test_cog_export_takes_fields_and_creation_options_as_keywords(
    tmp_path: Path,
) -> None:
    import rasterio

    (file,) = build_raster().gs.to_cog(
        f"{tmp_path}/{{prefix}}.tif", prefix="s2", compress="deflate"
    )
    path = Path(file.href)

    assert path == tmp_path / "s2.tif"
    with rasterio.open(path) as src:
        assert src.compression.name.lower() == "deflate"


def test_cog_export_refuses_a_path_that_is_not_a_tiff(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="end the path in .tif"):
        build_raster().gs.to_cog(tmp_path / "scene")


def test_to_cog_takes_no_id() -> None:
    assert "id" not in inspect.signature(GeoRaster.to_cog).parameters


def test_crs_name_stays_short_however_the_crs_was_built() -> None:
    coded = raster({"red": (("y", "x"), np.zeros((8, 8), "uint16"))}, geobox())
    named = raster(
        {
            "red": (
                ("y", "x"),
                np.zeros((8, 8), "uint16"),
            )
        },
        geobox(crs="+proj=laea +lat_0=52 +lon_0=10"),
    )

    assert coded.gs.crs_name == "EPSG:32633"
    assert coded.red.gs.crs_name == "EPSG:32633"
    # A CRS read back off spatial_ref prints its whole WKT, which this refuses to.
    assert len(str(coded.gs.crs)) > 100
    assert named.gs.crs_name == "unknown"


def test_write_nodata_is_read_by_odc_too() -> None:
    box = geobox()

    built = raster({"red": (("y", "x"), np.zeros(box.shape, "uint16"))}, box, nodata=0)

    assert built.red.attrs["_FillValue"] == 0
    assert built.red.gs.nodata == 0
    assert built.red.odc.nodata == 0


def test_raster_states_no_fill_when_none_is_given() -> None:
    box = geobox()

    built = raster({"red": (("y", "x"), np.zeros(box.shape, "uint16"))}, box)

    assert "nodata" not in built.red.attrs
    assert "_FillValue" not in built.red.attrs


def test_raster_keeps_pixels_untouched() -> None:
    box = geobox()
    values = np.arange(64, dtype="float32").reshape(box.shape)

    built = raster({"red": (("y", "x"), values)}, box)

    assert np.array_equal(built.red.values, values)
    assert built.red.dtype == np.dtype("float32")


def test_to_array_keeps_what_every_band_states_alike() -> None:
    box = geobox()
    built = raster(
        {
            "B04": (("y", "x"), np.zeros(box.shape, "uint16")),
            "B03": (("y", "x"), np.zeros(box.shape, "uint16")),
        },
        box,
    ).gs.write_packing(1e-4)
    built = built.assign(
        B04=built.B04.assign_attrs(long_name="red"), B03=built.B03.assign_attrs(long_name="green")
    )

    stacked = built.gs.to_array()

    assert stacked.band.values.tolist() == ["B04", "B03"]
    assert stacked.attrs == {"scale_factor": pytest.approx(1e-4)}


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


def test_one_nodata_spelled_two_ways_stacks_as_one() -> None:
    source = raster(
        {
            "red": (("y", "x"), np.ones(geobox().shape, "int16")),
            "nir": (("y", "x"), np.ones(geobox().shape, "int16")),
        },
        geobox(),
    )
    source.red.attrs = {"nodata": 0, "long_name": "red"}
    source.nir.attrs = {"_FillValue": 0, "long_name": "nir"}

    assert source.gs.to_array().attrs == {"_FillValue": 0}


def test_bands_with_different_nodata_stack_without_one() -> None:
    source = raster(
        {
            "red": (("y", "x"), np.ones(geobox().shape, "int16")),
            "nir": (("y", "x"), np.ones(geobox().shape, "int16")),
        },
        geobox(),
    )
    source.red.attrs = {"_FillValue": 0, "units": "1"}
    source.nir.attrs = {"_FillValue": -1, "units": "1"}

    assert source.gs.to_array().attrs == {"units": "1"}


def test_nan_nodata_stacks() -> None:
    source = raster({"red": (("y", "x"), np.ones(geobox().shape, "float32"))}, geobox())
    source.red.attrs = {"nodata": np.float32("nan")}

    assert np.isnan(source.gs.to_array().gs.nodata)


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
    built = raster({"B04": (("y", "x"), np.zeros(box.shape, "uint16"))}, box).gs.write_packing(1e-4)

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


def test_nodata_reads_back_what_was_written() -> None:
    box = geobox()
    pixels = np.zeros(box.shape, "uint16")
    built = raster(
        {"red": (("y", "x"), pixels), "nir": (("y", "x"), pixels)},
        box,
    ).gs.write_nodata(0, target="red")

    assert built.red.gs.nodata == 0
    assert built.nir.gs.nodata is None
    assert built.gs.nodata == {"red": 0, "nir": None}


def test_raster_statistics_tabulate_each_variable_as_its_band_does() -> None:
    box = geobox()
    pixels = np.arange(64, dtype="uint16").reshape(box.shape)
    built = raster(
        {
            "red": (("y", "x"), pixels),
            "nir": (("y", "x"), pixels * 2),
        },
        box,
        nodata=0,
    )

    table = built.gs.statistics()

    assert list(table.index) == ["red", "nir"]
    assert list(table.columns) == [
        "minimum",
        "maximum",
        "mean",
        "stddev",
        "valid_percent",
    ]
    pd.testing.assert_frame_equal(table.loc[["nir"]], built.nir.gs.statistics())


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


def test_dataset_statistics_have_one_row_per_variable_with_band_dimensions() -> None:
    source = xr.Dataset(
        {"image": (("band", "x"), [[1, 3], [10, 20]])},
        coords={"band": ["red", "nir"]},
    )

    table = source.gs.statistics()

    assert list(table.index) == ["image"]
    assert table.loc["image", "mean"] == 8.5


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


def test_write_crs_replaces_grid_coordinate_attrs_and_keeps_the_root() -> None:
    built = raster({"red": (("y", "x"), np.ones(geobox().shape))}, geobox())
    built.attrs = {"title": "S2"}
    built.x.attrs["crs"] = "EPSG:4326"

    written = built.gs.write_crs()

    assert written.attrs == {"title": "S2"}
    assert CRS(written.x.attrs["crs"]) == CRS(UTM)
    assert written.x.attrs["axis"] == "X"


def test_write_crs_keeps_a_one_pixel_grid() -> None:
    built = raster(
        {"red": (("y", "x"), np.ones((1, 1)))},
        geobox(shape=(1, 1)),
    )

    assert built.gs.write_crs().odc.geobox == built.odc.geobox


@pytest.mark.parametrize(
    "writer",
    [
        GeoRaster.to_cog,
        GeoRaster.to_zarr,
        GeoRaster.to_netcdf,
        GeoStack.to_cog,
        GeoStack.to_zarr,
        GeoStack.to_netcdf,
        GeoArray.to_cog,
    ],
)
def test_writers_take_no_catalog_argument(writer) -> None:
    assert "catalog" not in inspect.signature(writer).parameters


@pytest.mark.parametrize(
    "writer",
    [
        GeoRaster.to_zarr,
        GeoRaster.to_netcdf,
        GeoStack.to_cog,
        GeoStack.to_zarr,
        GeoStack.to_netcdf,
        GeoArray.to_cog,
    ],
)
def test_writers_that_name_nothing_take_no_id(writer) -> None:
    assert "id" not in inspect.signature(writer).parameters


def test_a_band_indexes_as_its_one_variable_raster(tmp_path: Path) -> None:
    from geosave_engine.geodata.io.vector import stac_geoparquet
    from geosave_engine.geodata.stac import create_items

    band = build_raster(times=2)["red"].isel(time=0)
    (file,) = band.gs.to_cog(tmp_path / "red.tif")

    (item,) = stac_geoparquet.to_items(create_items([file]))

    assert item.id == "red"
    assert list(item.assets) == ["red"]
    assert item.assets["red"].href == file.href


def _classes() -> xr.Dataset:
    box = geobox()
    codes = np.zeros(box.shape, "uint8")
    return raster({"label": (("y", "x"), codes), "other": (("y", "x"), codes)}, box)


NAMES = {0: "water", 1: "trees"}
COLORS = {0: "#419bdf", 1: "#397d49"}


def test_write_legend_names_and_colours_each_class() -> None:
    labelled = _classes().gs.write_legend(NAMES, COLORS, target="label")

    legend = labelled.label.gs.legend
    assert legend.classes == NAMES
    assert legend.color_map == {0: "#419BDF", 1: "#397D49"}
    assert labelled.other.gs.legend is None


def test_write_legend_without_colours_names_the_classes() -> None:
    labelled = _classes().gs.write_legend(NAMES, target="label")

    assert labelled.label.gs.legend.classes == NAMES


def test_write_legend_defaults_to_every_variable() -> None:
    labelled = _classes().gs.write_legend(NAMES)

    assert all(
        labelled[name].gs.legend.classes == NAMES
        for name in ("label", "other")
    )


def test_write_legend_refuses_a_variable_it_does_not_hold() -> None:
    with pytest.raises(KeyError):
        _classes().gs.write_legend(NAMES, target="missing")


def test_a_band_writes_its_own_legend() -> None:
    band = _classes()["label"].gs.write_legend(NAMES, COLORS)

    assert band.gs.legend.classes == NAMES


def test_a_written_legend_survives_a_geotiff_round_trip(tmp_path: Path) -> None:
    from geosave_engine.geodata.io import geotiff

    labelled = _classes()[["label"]].gs.write_legend(NAMES, COLORS)

    restored = gdal.read(geotiff.write_cog(labelled, tmp_path / "label.tif"))

    assert restored.label.gs.legend.classes == NAMES


def test_a_dated_band_writes_through_the_raster_template(tmp_path: Path) -> None:
    band = build_raster(times=2)["red"]

    paths = band.gs.to_cog(f"{tmp_path}/red_{{time:%Y%m%d}}.tif")

    assert [Path(f.href) for f in paths] == [
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


def test_a_legend_without_colours_writes_to_netcdf(tmp_path: Path) -> None:
    from geosave_engine.geodata import open_raster

    labelled = _classes()[["label"]].gs.write_legend(NAMES, None)

    restored = open_raster(labelled.gs.to_netcdf(tmp_path / "label.nc"))

    assert restored.label.gs.legend == labelled.label.gs.legend


def test_a_raster_crops_to_its_own_bounds() -> None:
    scene = build_raster().gs.write_nodata(0)

    cut = scene.gs.crop(scene.gs.bounds)

    assert cut.gs.geobox == scene.gs.geobox
