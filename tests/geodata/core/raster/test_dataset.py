from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo import CRS

from geosave.geodata import GeoDataArray, GeoDataset, GeoDataTree, attrs
from geosave.geodata.core.raster.create import raster
from geosave.geodata.io import gdal
from tests.geodata.conftest import build_raster, geobox
from tests.geodata.core.raster.conftest import optical
from tests.geodata.conftest import flat

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

    paths = [str(file) for file in flat(written.gs.to_cog(f"{tmp_path}/{template}"))]

    assert len(paths) == count
    for path in paths:
        with gdal.open_raster(path) as restored:
            expected = written.sel(time=restored.time) if times else written
            assert restored.gs.geobox == written.gs.geobox
            for name in restored.data_vars:
                np.testing.assert_array_equal(restored[name], expected[name])


def test_cog_export_takes_creation_options_as_keywords(
    tmp_path: Path,
) -> None:
    import rasterio

    (file,) = flat(build_raster().gs.to_cog(f"{tmp_path}/s2.tif", compress="deflate"))
    path = Path(file)

    assert path == tmp_path / "s2.tif"
    with rasterio.open(path) as src:
        assert src.compression.name.lower() == "deflate"


def test_cog_export_refuses_a_path_that_is_not_a_tiff(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="names no raster format"):
        flat(build_raster().gs.to_cog(tmp_path / "scene"))


def test_to_cog_takes_no_id() -> None:
    assert "id" not in inspect.signature(GeoDataset.to_cog).parameters


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
        B04=built.B04.assign_attrs(long_name="red"),
        B03=built.B03.assign_attrs(long_name="green"),
    )

    stacked = built.gs.to_array()

    assert stacked.band.values.tolist() == ["B04", "B03"]
    assert stacked.attrs == {"scale_factor": pytest.approx(1e-4)}


def test_one_nodata_spelled_two_ways_trees_as_one() -> None:
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


def test_bands_with_different_nodata_tree_without_one() -> None:
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


def test_dataset_statistics_have_one_row_per_variable_with_band_dimensions() -> None:
    source = xr.Dataset(
        {"image": (("band", "x"), [[1, 3], [10, 20]])},
        coords={"band": ["red", "nir"]},
    )

    table = source.gs.statistics()

    assert list(table.index) == ["image"]
    assert table.loc["image", "mean"] == 8.5


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
        GeoDataset.to_cog,
        GeoDataset.to_zarr,
        GeoDataset.to_netcdf,
        GeoDataTree.to_cog,
        GeoDataTree.to_zarr,
        GeoDataTree.to_netcdf,
        GeoDataArray.to_cog,
    ],
)
def test_writers_take_no_catalog_argument(writer) -> None:
    assert "catalog" not in inspect.signature(writer).parameters


@pytest.mark.parametrize(
    "writer",
    [
        GeoDataset.to_zarr,
        GeoDataset.to_netcdf,
        GeoDataTree.to_cog,
        GeoDataTree.to_zarr,
        GeoDataTree.to_netcdf,
        GeoDataArray.to_cog,
    ],
)
def test_writers_that_name_nothing_take_no_id(writer) -> None:
    assert "id" not in inspect.signature(writer).parameters


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

    assert all(labelled[name].gs.legend.classes == NAMES for name in ("label", "other"))


def test_write_legend_refuses_a_variable_it_does_not_hold() -> None:
    with pytest.raises(KeyError):
        _classes().gs.write_legend(NAMES, target="missing")


def test_a_band_writes_its_own_legend() -> None:
    band = _classes()["label"].gs.write_legend(NAMES, COLORS)

    assert band.gs.legend.classes == NAMES


def test_a_written_legend_survives_a_geotiff_round_trip(tmp_path: Path) -> None:
    from geosave.geodata.io import geotiff

    labelled = _classes()[["label"]].gs.write_legend(NAMES, COLORS)

    restored = gdal.open_raster(geotiff.write_cog(labelled, tmp_path / "label.tif"))

    assert restored.label.gs.legend.classes == NAMES


def test_a_legend_without_colours_writes_to_netcdf(tmp_path: Path) -> None:
    from geosave.geodata import open_raster

    labelled = _classes()[["label"]].gs.write_legend(NAMES, None)

    restored = open_raster(labelled.gs.to_netcdf(tmp_path / "label.nc"))

    assert restored.label.gs.legend == labelled.label.gs.legend


def test_a_raster_crops_to_its_own_bounds() -> None:
    scene = build_raster().gs.write_nodata(0)

    cut = scene.gs.crop(scene.gs.bounds)

    assert cut.gs.geobox == scene.gs.geobox


def _bands() -> xr.Dataset:
    band = (("y", "x"), np.zeros((2, 2), dtype="uint16"))
    return xr.Dataset({"red": band, "green": band, "nir": band}, coords={"y": [0, 1]})


def test_a_write_without_target_reaches_every_data_variable() -> None:
    written = _bands().gs.write_nodata(0)

    assert [written[name].gs.nodata for name in ("red", "green", "nir")] == [0, 0, 0]


def test_a_write_reaches_only_the_variables_it_targets() -> None:
    written = _bands().gs.write_nodata(0, target=["nir", "red"])

    assert written["green"].gs.nodata is None
    assert written["nir"].gs.nodata == written["red"].gs.nodata == 0


@pytest.mark.parametrize("target", ["blue", ["red", "blue"], ["y"]])
def test_a_target_that_is_not_a_data_variable_is_refused(target) -> None:
    with pytest.raises(KeyError, match="carries \\['red', 'green', 'nir'\\]"):
        _bands().gs.write_nodata(0, target=target)


def test_write_acdd_replaces_every_acdd_key_on_the_root() -> None:
    from geosave.geodata.attrs import ACDD

    ds = xr.Dataset(attrs={"title": "stale", "license": "CC0", "history": "kept"})

    written = ds.gs.write_acdd(ACDD(title="Forest"))

    assert written.attrs == {"history": "kept", "title": "Forest"}
    assert written.gs.acdd == ACDD(title="Forest")


def test_to_numpy_trees_variables_after_the_leading_axes() -> None:
    stacked = optical(times=3).gs.to_numpy()

    assert stacked.shape == (3, 2, 8, 8)
    assert stacked.dtype == np.dtype("uint16")


def test_to_numpy_trees_in_the_order_xarray_holds_the_variables() -> None:
    stacked = optical()[["B08", "B04"]].gs.to_numpy()

    assert stacked[0, :, 0, 0].tolist() == [8, 4]


def test_to_numpy_reads_an_unplaced_raster() -> None:
    unplaced = raster(
        {
            "pixels": (
                ("y", "x"),
                np.zeros((4, 4), "uint8"),
            )
        }
    )

    assert unplaced.gs.to_numpy().shape == (1, 4, 4)


def test_to_numpy_refuses_a_raster_carrying_no_variables() -> None:
    with pytest.raises(ValueError, match="no variables to stack"):
        optical()[[]].gs.to_numpy()


def test_to_numpy_refuses_variables_on_different_axes() -> None:
    source = optical()
    timeless = raster({"dem": (("y", "x"), np.zeros((8, 8), "uint16"))}, geobox())
    mixed = source.assign(dem=timeless.dem)

    with pytest.raises(ValueError, match="different axes"):
        mixed.gs.to_numpy()


def test_to_numpy_refuses_variables_on_different_dtypes() -> None:
    source = optical()
    mixed = source.assign(ndvi=source.B04.astype("float32"))

    with pytest.raises(ValueError, match="different dtypes"):
        mixed.gs.to_numpy()


def test_to_numpy_casts_variables_onto_one_dtype() -> None:
    source = optical()
    mixed = source.assign(ndvi=source.B04.astype("float32"))

    assert mixed.gs.to_numpy(dtype="float32").dtype == np.dtype("float32")


def test_to_numpy_refuses_a_raster_already_carrying_the_stacking_axis() -> None:
    conflicting = raster(
        {
            "pixels": (
                ("band", "y", "x"),
                np.zeros((3, 4, 4), "uint8"),
            )
        },
        None,
    )

    with pytest.raises(ValueError, match="already carries a 'band' dimension"):
        conflicting.gs.to_numpy()


def test_to_tensor_keeps_the_pixel_dtype_and_the_to_numpy_layout(torch) -> None:
    tensor = optical().gs.to_tensor()

    assert tensor.dtype is torch.uint16
    assert tuple(tensor.shape) == (2, 2, 8, 8)
    assert tensor.is_contiguous()
    torch.testing.assert_close(tensor, torch.tensor(optical().gs.to_numpy()))


def test_to_tensor_computes_a_lazy_raster(torch) -> None:
    tensor = optical().chunk({"y": 4}).gs.to_tensor()

    assert tuple(tensor.shape) == (2, 2, 8, 8)
    tensor.fill_(0)


def test_to_tensor_refuses_variables_on_different_dtypes(torch) -> None:
    mixed = optical().assign(B08=optical().B08.astype("float32"))

    with pytest.raises(ValueError, match="dtype"):
        mixed.gs.to_tensor()


def build_bands(*colours: str | None) -> xr.Dataset:
    """Bands whose names do not hint at their declared color interpretation."""
    pixels = xr.DataArray(np.ones((2, 2), "uint8"), dims=("y", "x"))
    raster = xr.Dataset({f"b{band}": pixels for band, _ in enumerate(colours)})
    for band, colour in enumerate(colours):
        if colour is not None:
            raster[f"b{band}"].attrs |= attrs.GDALBand(colorinterp=colour).to_attrs()
    return raster


def test_rgb_names_the_three_colour_bands() -> None:
    assert build_bands("red", "green", "blue").gs.rgb == ("b0", "b1", "b2")


def test_rgb_follows_what_bands_draw_not_their_order() -> None:
    assert build_bands("blue", "red", "green").gs.rgb == ("b1", "b2", "b0")


def test_rgb_passes_over_a_band_drawing_nothing() -> None:
    assert build_bands("red", None, "green", "blue").gs.rgb == ("b0", "b2", "b3")


def test_rgb_takes_the_first_of_two_bands_drawing_one_colour() -> None:
    assert build_bands("red", "red", "green", "blue").gs.rgb == ("b0", "b2", "b3")


def test_rgb_is_none_where_no_band_draws_blue() -> None:
    # A raster whose bands state no blue channel composes no colour of its own.
    assert build_bands("gray", "red", "green").gs.rgb is None


def test_a_plot_without_bands_refuses_a_raster_drawing_no_colour() -> None:
    with pytest.raises(ValueError, match="write_rgb"):
        build_bands("gray", "red", "green").gs.plot()


def test_write_rgb_names_the_channels_a_composite_draws() -> None:
    raster = build_bands(None, None, None, None)

    written = raster.gs.write_rgb("b2", "b1", "b0")

    assert written.gs.rgb == ("b2", "b1", "b0")
    assert "colorinterp" not in written.b3.attrs


def test_write_rgb_repoints_the_channels_it_replaces() -> None:
    raster = build_bands("red", "green", "blue", None)

    written = raster.gs.write_rgb("b3", "b0", "b1")

    assert written.gs.rgb == ("b3", "b0", "b1")
    # b2 drew blue and no longer does, so it must not compose a stale colour.
    assert "colorinterp" not in written.b2.attrs


def test_write_rgb_leaves_its_source_alone() -> None:
    raster = build_bands(None, None, None)

    raster.gs.write_rgb("b0", "b1", "b2")

    assert all("colorinterp" not in raster[name].attrs for name in raster.data_vars)


def test_write_rgb_refuses_a_name_the_raster_does_not_carry() -> None:
    with pytest.raises(KeyError, match="not data variables"):
        build_bands(None, None).gs.write_rgb("b0", "b1", "b2")


def test_write_rgb_refuses_one_variable_on_two_channels() -> None:
    with pytest.raises(ValueError, match="cannot draw two channels"):
        build_bands(None, None).gs.write_rgb("b0", "b0", "b1")
