from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo import CRS

from geosave.geodata import (
    attrs,
    open_raster,
)
from geosave.geodata.warnings import DroppedAttrsWarning
from tests.conftest import build_dataset
from tests.geodata.conftest import build_raster, flat, geobox
from tests.geodata.core.raster.conftest import optical

UTM = "EPSG:32633"
DAYS = ("20250601", "20250602")


def test_to_dataarray_keeps_what_every_band_states_alike() -> None:
    box = geobox()
    built = build_dataset(
        {
            "B04": (("y", "x"), np.zeros(box.shape, "uint16")),
            "B03": (("y", "x"), np.zeros(box.shape, "uint16")),
        },
        box,
    ).gs.write_scale_offset(1e-4)
    built = built.assign(
        B04=built.B04.assign_attrs(long_name="red"),
        B03=built.B03.assign_attrs(long_name="green"),
    )

    with pytest.warns(DroppedAttrsWarning, match="long_name"):
        stacked = built.gs.to_dataarray()

    assert stacked.band.values.tolist() == ["B04", "B03"]
    assert stacked.attrs == {"scale_factor": pytest.approx(1e-4), "add_offset": 0}


@pytest.mark.parametrize("attributes", [{"scale_factor": 0.1}, {"_FillValue": 255}])
def test_stacking_refuses_conflicting_packing_without_reading_pixels(attributes):
    import dask.array as da
    from dask.delayed import delayed

    @delayed
    def pixels():
        raise AssertionError("packing validation must not compute pixels")

    values = da.from_delayed(pixels(), shape=(2, 2), dtype="uint16")
    source = xr.Dataset({"red": (("y", "x"), values), "nir": (("y", "x"), values)})
    source.nir.attrs.update(attributes)

    with pytest.raises(ValueError, match="matching CF packing"):
        source.gs.to_dataarray()


def test_stacking_accepts_native_attr_policy():
    source = build_raster()
    source.red.attrs["long_name"] = "red"
    source.nir.attrs["long_name"] = "infrared"

    stacked = source.gs.to_dataarray(combine_attrs="override")

    assert stacked.attrs["long_name"] == "red"


def test_nodata_reads_back_what_was_written() -> None:
    box = geobox()
    pixels = np.zeros(box.shape, "uint16")
    built = build_dataset(
        {"red": (("y", "x"), pixels), "nir": (("y", "x"), pixels)},
        box,
    ).gs.write_nodata(0, target="red")

    assert built.red.gs.nodata == 0
    assert built.nir.gs.nodata is None
    assert built.gs.nodata == {"red": 0, "nir": None}


def test_raster_statistics_tabulate_each_variable_as_its_band_does() -> None:
    box = geobox()
    pixels = np.arange(64, dtype="uint16").reshape(box.shape)
    built = build_dataset(
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
    built = build_dataset({"red": (("y", "x"), np.ones(geobox().shape))}, geobox())
    built.attrs = {"title": "S2"}
    built.x.attrs["crs"] = "EPSG:4326"

    written = built.gs.write_crs()

    assert written.attrs == {"title": "S2"}
    assert CRS(written.x.attrs["crs"]) == CRS(UTM)
    assert written.x.attrs["axis"] == "X"


def test_write_crs_keeps_a_one_pixel_grid() -> None:
    built = build_dataset(
        {"red": (("y", "x"), np.ones((1, 1)))},
        geobox(shape=(1, 1)),
    )

    assert built.gs.write_crs().odc.geobox == built.odc.geobox


def _classes() -> xr.Dataset:
    box = geobox()
    codes = np.zeros(box.shape, "uint8")
    return build_dataset(
        {"label": (("y", "x"), codes), "other": (("y", "x"), codes)}, box
    )


NAMES = {0: "water", 1: "trees"}
COLORS = {0: "#419bdf", 1: "#397d49"}


def test_write_legend_names_and_colours_each_class() -> None:
    labelled = _classes().gs.write_legend(NAMES, COLORS, target="label")

    legend = labelled.label.gs.legend
    assert legend.classes == NAMES
    assert legend.color_map == {0: "#419BDF", 1: "#397D49"}
    assert labelled.other.gs.legend is None


def test_write_legend_defaults_to_every_variable() -> None:
    labelled = _classes().gs.write_legend(NAMES)

    assert all(labelled[name].gs.legend.classes == NAMES for name in ("label", "other"))


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
    unplaced = build_dataset(
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
    timeless = build_dataset(
        {"dem": (("y", "x"), np.zeros((8, 8), "uint16"))}, geobox()
    )
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
    conflicting = build_dataset(
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


def build_bands(*colours: str | None) -> xr.Dataset:
    """Bands whose names do not hint at their declared color interpretation."""
    pixels = xr.DataArray(np.ones((2, 2), "uint8"), dims=("y", "x"))
    raster = xr.Dataset({f"b{band}": pixels for band, _ in enumerate(colours)})
    for band, colour in enumerate(colours):
        if colour is not None:
            raster[f"b{band}"].attrs |= attrs.GDALBand(colorinterp=colour).to_attrs()
    return raster


def test_rgb_follows_what_bands_draw_not_their_order() -> None:
    assert build_bands("blue", "red", "green").gs.rgb == ("b1", "b2", "b0")


def test_rgb_takes_the_first_of_two_bands_drawing_one_colour() -> None:
    assert build_bands("red", "red", "green", "blue").gs.rgb == ("b0", "b2", "b3")


def test_rgb_is_none_where_no_band_draws_blue() -> None:
    # A raster whose bands state no blue channel composes no colour of its own.
    assert build_bands("gray", "red", "green").gs.rgb is None


def test_a_plot_without_bands_refuses_a_raster_drawing_no_colour() -> None:
    with pytest.raises(ValueError, match="write_rgb"):
        build_bands("gray", "red", "green").gs.plot()


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


def test_properties_survive_per_band_catalog_publication(tmp_path):
    from geosave.geodata.stac.raster import stamp_properties_coordinate

    ds = stamp_properties_coordinate(
        build_raster(times=2), [{"eo:cloud_cover": 12.0}, {"eo:cloud_cover": 65.0}]
    )
    catalog = ds.gs.to_catalog(tmp_path, format_paths="{time:%Y%m%d}_{variable}.tif")
    items = list(catalog.gs.to_items())
    # Two band files per date are one Item each date, with that date's properties.
    assert [item.properties["eo:cloud_cover"] for item in items] == [12.0, 65.0]
    assert [sorted(item.assets) for item in items] == [["nir", "red"], ["nir", "red"]]


@pytest.mark.parametrize(
    ("times", "split_vars", "path_format", "expected"),
    [
        (0, False, "scene.tif", ["scene.tif"]),
        (0, True, "s2_{variable}.tif", ["s2_red.tif", "s2_nir.tif"]),
        (0, True, "scene/{variable}.tif", ["scene/red.tif", "scene/nir.tif"]),
        (2, False, "scene_{time:%Y%m%d}.tif", [f"scene_{day}.tif" for day in DAYS]),
        (
            2,
            True,
            "{time:%Y%m%d}/{variable}.tif",
            [
                "20250601/red.tif",
                "20250601/nir.tif",
                "20250602/red.tif",
                "20250602/nir.tif",
            ],
        ),
    ],
    ids=["one-file", "flat-bands", "folder-bands", "dates", "dates-and-bands"],
)
def test_explicit_splits_and_paths_preserve_pixels(
    tmp_path: Path, times: int, split_vars: bool, path_format: str, expected: list[str]
) -> None:
    written = build_raster(times=times)

    files = written.gs.to_cog(
        tmp_path,
        split_asset=("variable",) if split_vars else (),
        format_paths=path_format,
    )

    assert [Path(f) for f in files] == [tmp_path / name for name in expected]
    for path in files:
        with open_raster(path) as restored:
            assert restored.gs.geobox == written.gs.geobox
            for name in restored.data_vars:
                source = written[name]
                if times:
                    source = source.sel(time=restored.time.values)
                np.testing.assert_array_equal(
                    restored[name].squeeze(), source.squeeze()
                )


def test_place_is_a_callers_field_not_a_built_in(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lat"):
        flat(
            build_raster().gs.to_cog(
                tmp_path, split_asset=(), format_paths="{lat:.3f}.tif"
            )
        )


def test_several_dates_without_time_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collide"):
        flat(
            build_raster(times=2).gs.to_cog(
                tmp_path, split_asset=(), format_paths="scene.tif"
            )
        )
    assert not list(tmp_path.iterdir())


def test_a_partial_date_that_collides_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collid"):
        flat(
            build_raster(times=2).gs.to_cog(
                tmp_path, split_asset=(), format_paths="{time:%Y}.tif"
            )
        )


def test_time_on_an_undated_raster_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unavailable labels.*time"):
        flat(
            build_raster().gs.to_cog(
                tmp_path, split_asset=(), format_paths="{time:%Y}.tif"
            )
        )


def test_sub_second_instants_render_through_strftime(tmp_path: Path) -> None:
    instant = np.datetime64("2025-06-01T10:30:31.123456", "ns")
    written = build_raster(times=1).assign_coords(time=[instant])

    (file,) = flat(
        written.gs.to_cog(
            tmp_path, split_asset=(), format_paths="{time:%Y%m%dT%H%M%S_%f}.tif"
        )
    )

    assert file.name == "20250601T103031_123456.tif"
    assert open_raster(file).time.values == instant


def test_variable_attrs_reach_every_file(tmp_path: Path) -> None:
    source = build_raster(times=2)
    written = source.assign(
        {
            name: source[name].assign_attrs(units="reflectance")
            for name in ("red", "nir")
        }
    )

    files = written.gs.to_cog(
        tmp_path, split_asset=(), format_paths="{time:%Y%m%d}.tif"
    )

    for path in files:
        assert open_raster(path).red.attrs["units"] == "reflectance"


def test_an_existing_file_refuses_without_overwrite(tmp_path: Path) -> None:
    written = build_raster(times=1)
    flat(written.gs.to_cog(tmp_path, split_asset=(), format_paths="scene.tif"))

    with pytest.raises(FileExistsError):
        flat(written.gs.to_cog(tmp_path, split_asset=(), format_paths="scene.tif"))
    assert flat(
        written.gs.to_cog(
            tmp_path, split_asset=(), format_paths="scene.tif", overwrite=True
        )
    )


def test_a_raster_spanning_more_than_time_refuses(tmp_path: Path) -> None:
    spanning = build_raster(times=2).expand_dims({"model": ["a", "b"]})

    with pytest.raises(ValueError, match="a COG holds one instant of one grid"):
        flat(
            spanning.gs.to_cog(
                tmp_path, split_asset=(), format_paths="{time:%Y%m%d}.tif"
            )
        )


def test_remote_paths_keep_their_scheme(bucket: str) -> None:
    files = flat(
        build_raster(times=2).gs.to_cog(
            bucket, format_paths="{time:%Y%m%d}/{variable}.tif"
        )
    )

    assert [str(f) for f in files] == [
        f"{bucket}/20250601/red.tif",
        f"{bucket}/20250601/nir.tif",
        f"{bucket}/20250602/red.tif",
        f"{bucket}/20250602/nir.tif",
    ]


def test_each_file_states_its_band_and_date(tmp_path: Path) -> None:
    scene = build_raster(times=2)

    files = flat(
        scene.gs.to_cog(tmp_path, format_paths="s_{time:%Y%m%d}_{variable}.tif")
    )

    headers = [open_raster(f) for f in files]
    assert [h.gs.variables for h in headers] == [("red",), ("nir",), ("red",), ("nir",)]
    assert len({str(f) for f in files}) == 4
    assert headers[0].gs.times[0] == pd.Timestamp(scene.time.values[0])
    assert headers[0].gs.acdd is None


def test_a_scalar_time_fills_the_time_field(tmp_path: Path) -> None:
    # A scalar time coordinate is one date, as one step of a time axis is.
    scene = build_raster(times=2).isel(time=1)

    files = flat(
        scene.gs.to_cog(tmp_path, split_asset=(), format_paths="s_{time:%Y%m%d}.tif")
    )

    assert [file.name for file in files] == ["s_20250602.tif"]
    restored = open_raster(files[0])
    assert pd.Timestamp(restored.time.values) == pd.Timestamp("2025-06-02")


def test_a_relative_template_returns_absolute_paths(scene, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (path, _) = flat(
        scene.gs.to_cog("out", split_asset=(), format_paths="s_{time:%Y%m%d}.tif")
    )

    assert Path(path).is_absolute()
    assert Path(path) == tmp_path / "out" / "s_20250601.tif"


def test_native_deferred_cog_inventory_waits_for_pixels(tmp_path):
    from dask import delayed
    from dask.array import from_delayed
    from dask.delayed import Delayed
    from upath import UPath
    from tests.geodata.conftest import build_raster

    calls = []
    raster = build_raster(times=0)

    values = raster.red.values

    @delayed
    def pixels():
        calls.append(True)
        return values

    raster = raster.assign(
        red=(
            raster.red.dims,
            from_delayed(pixels(), shape=raster.red.shape, dtype=raster.red.dtype),
        )
    )
    pending = raster.gs.to_cog(tmp_path, format_paths="{variable}.tif", compute=False)
    assert isinstance(pending, Delayed)
    assert calls == []
    assert list(tmp_path.iterdir()) == []
    paths = pending.compute()
    assert isinstance(paths, list)
    assert all(isinstance(path, UPath) and path.exists() for path in paths)
    assert calls == [True]


@pytest.mark.parametrize(
    ("method", "backend", "suffix"),
    [
        ("to_cog", "geotiff", ".tif"),
        ("to_zarr", "zarr", ".zarr"),
        ("to_netcdf", "netcdf", ".nc"),
    ],
)
def test_deferred_formats_receive_lazy_pixels(
    tmp_path, monkeypatch, method, backend, suffix
):
    from geosave.geodata.io.formats import geotiff
    from geosave.geodata.io.formats import netcdf
    from geosave.geodata.io.formats import zarr

    source = build_raster().chunk({"y": 1, "x": 1})
    module = {"geotiff": geotiff, "zarr": zarr, "netcdf": netcdf}[backend]
    # The step that runs when the delayed result computes, and receives the pixels.
    name = "write_cog" if backend == "geotiff" else "_write"
    write = getattr(module, name)
    chunks = []

    def save(raster, destination, *args, **options):
        chunks.append(raster.red.chunks)
        assert raster.red.chunks == source.red.chunks
        return write(raster, destination, *args, **options)

    monkeypatch.setattr(module, name, save)
    if method == "to_cog":
        pending = source.gs.to_cog(
            tmp_path, split_asset=(), format_paths="raster.tif", compute=False
        )
    else:
        pending = getattr(source.gs, method)(
            tmp_path / f"raster{suffix}", compute=False
        )
    assert chunks == []
    saved = pending.compute()
    saved = saved[0] if method == "to_cog" else saved
    assert chunks == [source.red.chunks]
    with open_raster(saved) as restored:
        np.testing.assert_array_equal(restored.red, source.red)


@pytest.mark.parametrize("kind", ["dataset", "array", "tree"])
def test_explicit_templates_round_trip_under_a_literal_parent(tmp_path, kind):
    source = build_raster(times=2).chunk()
    parent = tmp_path / "{literal}"
    if kind == "array":
        source = source.red
    if kind == "tree":
        source = xr.DataTree.from_dict({"s2": source})
        template = "{asset}_{time:%Y%m%d}.tif"
        files = source.gs.to_cog(parent, format_paths=template)
    else:
        template = "{time:%Y%m%d}_{variable}.tif"
        files = source.gs.to_cog(parent, format_paths=template)
    assert files[0] == parent / (
        "s2/red_20250601.tif" if kind == "tree" else "20250601_red.tif"
    )
    with open_raster(files[0]) as restored:
        np.testing.assert_array_equal(
            restored.red, build_raster(times=2).red.isel(time=0)
        )
        assert restored.gs.geobox == build_raster().gs.geobox


def test_an_exact_destination_does_not_format_braces(tmp_path):
    source = build_raster()
    destination = tmp_path / "scene_{variable}.nc"
    saved = source.gs.to_netcdf(destination)
    assert saved == destination
    with open_raster(saved) as restored:
        np.testing.assert_array_equal(restored.red, source.red)
        assert list(restored.data_vars) == ["red", "nir"]


class RecordingArray:
    """A 2-D array recording the shape of every slice it serves."""

    def __init__(self, values: np.ndarray, served_shapes: list) -> None:
        self.values = values
        self.served_shapes = served_shapes
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim

    def __getitem__(self, key):
        served = self.values[key]
        self.served_shapes.append(served.shape)
        return served


def create_mercator_grid(pixel_size: float, size: int = 1024):
    from odc.geo.geobox import GeoBox

    return GeoBox.from_bbox(
        (0, 0, size * 10.0, size * 10.0), crs="EPSG:3857", resolution=pixel_size
    )


def test_read_pixels_samples_before_computing():
    import dask.array

    served_shapes = []
    values = np.arange(1024 * 1024, dtype="float32").reshape(1024, 1024)
    band = dask.array.from_array(RecordingArray(values, served_shapes), chunks=256)
    scene = build_dataset({"red": (("y", "x"), band)}, create_mercator_grid(10.0))
    # Dask probes the array with an empty slice when it is built
    served_shapes.clear()
    view = create_mercator_grid(80.0)

    pixels = scene.gs.read_pixels(view)

    assert pixels.odc.geobox == view
    served_pixels = 0
    for shape in served_shapes:
        served_pixels += shape[0] * shape[1]
    assert served_pixels <= 128 * 128


def test_read_pixels_outside_returns_nan_without_computing():
    import dask.array

    served_shapes = []
    values = np.ones((1024, 1024), dtype="float32")
    band = dask.array.from_array(RecordingArray(values, served_shapes), chunks=256)
    scene = build_dataset({"red": (("y", "x"), band)}, create_mercator_grid(10.0))
    # Dask probes the array with an empty slice when it is built
    served_shapes.clear()
    from odc.geo.geobox import GeoBox

    view = GeoBox.from_bbox(
        (50000, 50000, 60000, 60000), crs="EPSG:3857", resolution=100.0
    )

    pixels = scene.gs.read_pixels(view)

    assert pixels.odc.geobox == view
    assert np.isnan(pixels["red"].values).all()
    assert served_shapes == []


def test_read_pixels_keeps_band_identity_and_nodata():
    values = np.full((64, 64), 3, dtype="uint8")
    values[:4, :4] = 255
    scene = build_dataset(
        {"cover": (("y", "x"), values)}, create_mercator_grid(10.0, 64), nodata=255
    )
    scene = scene.gs.write_legend({3: "trees"}, {3: "#397d49"}, target="cover")

    pixels = scene.gs.read_pixels(create_mercator_grid(10.0, 64))

    assert list(pixels.data_vars) == ["cover"]
    assert pixels["cover"].dtype == np.float32
    assert pixels["cover"].gs.legend == scene["cover"].gs.legend
    assert np.isnan(pixels["cover"].values[:4, :4]).all()


def test_read_pixels_states_the_grid_it_reads_onto():
    from odc.geo.geobox import GeoBox

    grid = GeoBox.from_bbox(
        (500000, 9000000, 500640, 9000640), crs="EPSG:32749", resolution=10
    )
    scene = build_dataset({"red": (("y", "x"), np.ones((64, 64), "float32"))}, grid)
    view = GeoBox.from_bbox(
        grid.extent.to_crs("EPSG:3857").boundingbox, crs="EPSG:3857", shape=(32, 32)
    )

    pixels = scene.gs.read_pixels(view)

    assert pixels.odc.geobox == view


@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("axes", [("latitude", "longitude"), ("north", "east")])
def test_read_pixels_native_axes_preserve_time_metadata_and_axis_order(outside, axes):
    import dask.array
    from odc.geo.geobox import GeoBox

    served_shapes = []
    values = np.arange(2 * 64 * 64, dtype="float32").reshape(2, 64, 64)
    band = dask.array.from_array(
        RecordingArray(values, served_shapes), chunks=(1, 16, 16)
    )
    scene = (
        build_dataset(
            {"red": (("time", "y", "x"), band)},
            create_mercator_grid(10.0, 64),
            time=pd.date_range("2025-01-01", periods=2),
        )
        .rename(y=axes[0], x=axes[1])
        .transpose(axes[1], "time", axes[0])
    )
    scene = scene.assign_coords(sensor=("time", ["a", "b"]), mission="example")
    scene.attrs["title"] = "Two observations"
    scene.red.attrs["long_name"] = "Red reflectance"
    served_shapes.clear()
    bounds = (10000, 10000, 10160, 10160) if outside else (0, 0, 160, 160)
    view = GeoBox.from_bbox(bounds, crs="EPSG:3857", resolution=10)

    pixels = scene.gs.read_pixels(view)

    assert pixels.odc.geobox == view
    assert pixels.red.dims == ("x", "time", "y")
    xr.testing.assert_identical(pixels.time.variable, scene.time.variable)
    xr.testing.assert_identical(pixels.sensor.variable, scene.sensor.variable)
    xr.testing.assert_identical(pixels.mission.variable, scene.mission.variable)
    assert pixels.attrs["title"] == "Two observations"
    assert pixels.red.attrs["long_name"] == "Red reflectance"
    if outside:
        assert np.isnan(pixels.red).all()
        assert served_shapes == []
    else:
        np.testing.assert_array_equal(
            pixels.red.values, values[:, -16:, :16].transpose(2, 0, 1)
        )
        assert sum(np.prod(shape) for shape in served_shapes) <= 2 * 16 * 16


def test_read_pixels_rejects_variables_with_different_spatial_axis_names():
    grid = create_mercator_grid(10.0, 16)
    red = build_dataset({"red": (("y", "x"), np.ones((16, 16)))}, grid)
    nir = red.rename(red="nir", y="latitude", x="longitude")
    scene = xr.merge([red, nir])

    with pytest.raises(ValueError, match="spatial dimensions"):
        scene.gs.read_pixels(grid)


def test_to_dataarray_casts_before_stacking_without_reading_pixels() -> None:
    import dask.array as da
    from dask import delayed

    reads = []

    @delayed
    def pixels():
        reads.append(1)
        return np.array([[2**53 + 1]], dtype="uint64")

    source = xr.Dataset(
        {
            "large": (
                ("y", "x"),
                da.from_delayed(pixels(), shape=(1, 1), dtype="uint64"),
            ),
            "small": (("y", "x"), np.array([[1]], dtype="int64")),
        }
    )
    stacked = source.gs.to_dataarray(dtype="uint64")

    assert isinstance(stacked.data, da.Array)
    assert reads == []
    assert stacked.band.values.tolist() == ["large", "small"]
    np.testing.assert_array_equal(
        stacked.values[:, 0, 0], np.array([2**53 + 1, 1], dtype="uint64")
    )
    np.testing.assert_array_equal(source.gs.to_numpy(dtype="uint64"), stacked.values)


def test_to_dataarray_preserves_native_variable_names_and_order() -> None:
    source = xr.Dataset(
        {
            8: (("y", "x"), [[8]]),
            4: (("y", "x"), [[4]]),
        }
    )

    stacked = source.gs.to_dataarray()

    np.testing.assert_array_equal(stacked.band.values, [8, 4])
    np.testing.assert_array_equal(stacked.values[:, 0, 0], [8, 4])


def test_time_bounds_survive_array_conversion() -> None:
    from tests.geodata.conftest import assign_cells, build_raster

    dataset = build_raster(times=2)
    bounds = np.array(
        [["2025-06-01", "2025-06-02"], ["2025-06-11", "2025-06-12"]],
        dtype="datetime64[ns]",
    )
    dataset = assign_cells(dataset, bounds)

    restored = dataset.gs.to_dataarray().gs.to_dataset(dim="band")

    np.testing.assert_array_equal(restored.start_datetime.values, bounds[:, 0])
    np.testing.assert_array_equal(restored.end_datetime.values, bounds[:, 1])
    np.testing.assert_array_equal(restored.red, dataset.red)


def test_affine_validation_guarantees_one_geobox() -> None:
    from odc.geo.gcp import GCPGeoBox, GCPMapping
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import multipoint

    from geosave.geodata.errors import RasterValidationError

    box = geobox()
    pixels = np.zeros(box.shape, "uint16")
    scene = build_dataset({"red": (("y", "x"), pixels)}, box)
    scene.gs.validate(georeferenced=True, affine=True)
    scene.red.gs.validate(affine=True)
    assert isinstance(scene.gs.geobox, GeoBox)

    unplaced = xr.Dataset({"red": (("y", "x"), pixels)})
    with pytest.raises(RasterValidationError, match="pixel placement"):
        unplaced.gs.validate(affine=True)

    points = multipoint([(0, 0), (4, 0), (0, 4), (4, 4)], None)
    world = multipoint([(10, 20), (14, 20), (10, 16), (14, 16)], "EPSG:4326")
    surveyed = xr.DataArray(np.ones((4, 4)), dims=("y", "x")).gs.write_grid(
        GCPGeoBox((4, 4), GCPMapping(points, world))
    )
    surveyed.gs.validate(georeferenced=True)
    with pytest.raises(RasterValidationError, match="ground control points"):
        surveyed.gs.validate(affine=True)


def test_expand_rasters_keeps_every_variable_coordinate_and_time_bounds():
    import dask.array as da

    times = np.array(["2025-01-15", "2025-02-15"], dtype="datetime64[ns]")
    scene = xr.Dataset(
        {
            "red": (("time", "y", "x"), da.ones((2, 2, 2), chunks=1)),
            "dem": (("y", "x"), da.zeros((2, 2), chunks=1)),
        },
        coords={"time": times},
        attrs={"title": "Monthly observations"},
    )
    from tests.geodata.conftest import assign_cells

    scene = assign_cells(
        scene, [("2025-01-01", "2025-02-01"), ("2025-02-01", "2025-03-01")]
    )
    slices = scene.gs.expand_rasters("time")
    assert len(slices) == 2
    for position, piece in enumerate(slices):
        xr.testing.assert_identical(piece, scene.isel(time=position))
        assert list(piece.data_vars) == ["red", "dem"]
        assert piece.attrs == {"title": "Monthly observations"}
        assert piece.start_datetime.values == np.datetime64(f"2025-0{position + 1}-01")
        assert piece.red.chunks is not None
