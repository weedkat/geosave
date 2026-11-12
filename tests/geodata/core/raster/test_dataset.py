from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geosave.geodata import (
    attrs,
    open_raster,
)
from geosave.geodata.warnings import DroppedAttrsWarning
from tests.conftest import build_dataset
from tests.geodata.conftest import build_raster, geobox
from tests.geodata.core.raster.conftest import optical

import dask.array as da
from dask.callbacks import Callback

from shapely import from_wkt
from odc.geo.geom import Geometry


UTM = "EPSG:32633"


def test_to_array_keeps_what_every_band_states_alike() -> None:
    box = geobox()
    built = build_dataset(
        {
            "B04": (("y", "x"), np.zeros(box.shape, "uint16")),
            "B03": (("y", "x"), np.zeros(box.shape, "uint16")),
        },
        box,
    ).gs.write_packing(attrs.CFPacking(scale_factor=1e-4))
    built = built.assign(
        B04=built.B04.assign_attrs(long_name="red"),
        B03=built.B03.assign_attrs(long_name="green"),
    )

    with pytest.warns(DroppedAttrsWarning, match="long_name"):
        stacked = built.gs.to_array()

    assert stacked.band.values.tolist() == ["B04", "B03"]
    assert stacked.gs.packing.scale_factor == pytest.approx(1e-4)
    assert stacked.gs.packing.add_offset == 0
    assert "long_name" not in stacked.attrs


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

    with pytest.raises(ValueError, match="CF packing"):
        source.gs.to_array()


def test_stacking_accepts_native_attr_policy():
    source = build_raster()
    source.red.attrs["long_name"] = "red"
    source.nir.attrs["long_name"] = "infrared"

    stacked = source.gs.to_array(combine_attrs="override")

    assert stacked.attrs["long_name"] == "red"


@pytest.mark.parametrize(
    "left,right",
    [
        ({"units": "K"}, {"units": "m"}),
        ({"units": "K"}, {}),
        (
            {"flag_values": [0, 1], "flag_meanings": "water land"},
            {"flag_values": [1, 2], "flag_meanings": "water land"},
        ),
        (
            {"flag_values": [0, 1], "flag_meanings": "water land"},
            {"flag_values": [0, 1], "flag_meanings": "land water"},
        ),
        ({"valid_range": [0, 10]}, {"valid_range": [0, 20]}),
    ],
    ids=["units", "missing-units", "class-codes", "class-meanings", "valid-range"],
)
def test_stacking_rejects_different_pixel_meaning_before_reading(left, right):
    from dask.callbacks import Callback

    source = build_raster().chunk()
    source.red.attrs.update(left)
    source.nir.attrs.update(right)
    with Callback(pretask=lambda *_: pytest.fail("validation read pixels")):
        # Overriding descriptive attrs must not bypass pixel compatibility.
        with pytest.raises(ValueError, match="different.*(packing|flags|units)"):
            source.gs.to_array(combine_attrs="override")


@pytest.mark.parametrize(
    "policy", ["drop", "override", lambda *_args, **_kwargs: {"units": "wrong"}]
)
def test_stacking_attrs_policy_cannot_change_cf_meaning(policy, tmp_path):
    source = build_raster(packed=True).chunk()
    for array in source.data_vars.values():
        array.attrs["units"] = "1"
    before = source.copy(deep=True)
    image = source.gs.to_array(combine_attrs=policy)
    restored = image.gs.to_dataset()

    assert image.attrs["units"] == "1"
    assert image.attrs["scale_factor"] == source.red.attrs["scale_factor"]
    assert image.attrs["_FillValue"] == 0
    xr.testing.assert_allclose(xr.decode_cf(restored), xr.decode_cf(source))
    assert restored.gs.geobox == source.gs.geobox
    xr.testing.assert_identical(source, before)
    path = tmp_path / "bands.nc"
    restored.to_netcdf(path)
    with xr.open_dataset(path, decode_coords="all") as opened:
        xr.testing.assert_allclose(opened, xr.decode_cf(source, decode_coords="all"))


def test_nodata_reads_back_what_was_written() -> None:
    box = geobox()
    pixels = np.zeros(box.shape, "uint16")
    built = build_dataset(
        {"red": (("y", "x"), pixels), "nir": (("y", "x"), pixels)},
        box,
    ).gs.write_nodata(0, target=["red"])

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


def _classes() -> xr.Dataset:
    box = geobox()
    codes = np.zeros(box.shape, "uint8")
    return build_dataset(
        {"label": (("y", "x"), codes), "other": (("y", "x"), codes)}, box
    )


NAMES = {0: "water", 1: "trees"}
COLORS = {0: "#419bdf", 1: "#397d49"}


def test_write_legend_names_and_colours_each_class() -> None:
    labelled = _classes().gs.write_flags(NAMES, COLORS, target=["label"])

    legend = labelled.label.gs.flags
    assert legend.classes == NAMES
    assert labelled.label.gs.display.color_map == {0: "#419BDF", 1: "#397D49"}
    assert labelled.other.gs.flags is None


def test_write_legend_defaults_to_every_variable() -> None:
    labelled = _classes().gs.write_flags(NAMES)

    assert all(labelled[name].gs.flags.classes == NAMES for name in ("label", "other"))


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

    with pytest.raises(ValueError, match="spans"):
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

    with pytest.raises(ValueError):
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
    """Bands whose variable names do not hint at their spectral common names."""
    pixels = xr.DataArray(np.ones((2, 2), "uint8"), dims=("y", "x"))
    raster = xr.Dataset({f"b{band}": pixels for band, _ in enumerate(colours)})
    for band, colour in enumerate(colours):
        if colour is not None:
            raster[f"b{band}"].attrs |= attrs.EOBand(common_name=colour).to_attrs()
    return raster


def test_rgb_follows_what_bands_draw_not_their_order() -> None:
    assert build_bands("blue", "red", "green").gs.rgb == ("b1", "b2", "b0")


def test_rgb_takes_the_first_of_two_bands_drawing_one_colour() -> None:
    assert build_bands("red", "red", "green", "blue").gs.rgb == ("b0", "b2", "b3")


def test_rgb_is_none_where_no_band_draws_blue() -> None:
    # A raster whose bands state no blue channel composes no colour of its own.
    assert build_bands(None, "red", "green").gs.rgb is None


def test_a_plot_without_bands_refuses_a_raster_drawing_no_colour() -> None:
    with pytest.raises(ValueError, match="write_rgb"):
        build_bands(None, "red", "green").gs.plot()


def test_write_rgb_repoints_the_channels_it_replaces() -> None:
    raster = build_bands("red", "green", "blue", None)

    written = raster.gs.write_rgb("b3", "b0", "b1")

    assert written.gs.rgb == ("b3", "b0", "b1")
    # An explicit display overrides spectral roles without rewriting them.
    assert written.b2.gs.spectral.common_name == "blue"


def test_write_rgb_leaves_its_source_alone() -> None:
    raster = build_bands(None, None, None)

    raster.gs.write_rgb("b0", "b1", "b2")

    assert raster.gs.rgb is None
    assert attrs.Display.from_attrs(raster.attrs) is None


def test_write_rgb_refuses_a_name_the_raster_does_not_carry() -> None:
    with pytest.raises(KeyError, match="not data variables"):
        build_bands(None, None).gs.write_rgb("b0", "b1", "b2")


def test_write_rgb_refuses_one_variable_on_two_channels() -> None:
    with pytest.raises(ValueError, match="cannot draw two channels"):
        build_bands(None, None).gs.write_rgb("b0", "b0", "b1")


@pytest.mark.parametrize(
    ("method", "suffix"), [("to_zarr", ".zarr"), ("to_netcdf", ".nc")]
)
def test_deferred_storage_waits_for_compute(tmp_path, method, suffix):
    source = build_raster().chunk({"y": 1, "x": 1})
    tasks = []
    destination = tmp_path / f"raster{suffix}"
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        pending = getattr(source.gs, method)(destination, compute=False)
    assert tasks == []
    assert not destination.exists()
    saved = pending.compute()
    with open_raster(saved) as restored:
        xr.testing.assert_equal(restored, source)


def test_an_exact_destination_does_not_format_braces(tmp_path):
    source = build_raster()
    destination = tmp_path / "scene_{variable}.nc"
    saved = source.gs.to_netcdf(destination)
    assert saved == destination
    with open_raster(saved) as restored:
        np.testing.assert_array_equal(restored.red, source.red)
        assert list(restored.data_vars) == ["red", "nir"]


def test_to_array_casts_before_stacking_without_reading_pixels() -> None:
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
    stacked = source.gs.to_array(dtype="uint64")

    assert isinstance(stacked.data, da.Array)
    assert reads == []
    assert stacked.band.values.tolist() == ["large", "small"]
    np.testing.assert_array_equal(
        stacked.values[:, 0, 0], np.array([2**53 + 1, 1], dtype="uint64")
    )
    np.testing.assert_array_equal(source.gs.to_numpy(dtype="uint64"), stacked.values)


def test_to_array_preserves_native_variable_names_and_order() -> None:
    source = xr.Dataset(
        {
            8: (("y", "x"), [[8]]),
            4: (("y", "x"), [[4]]),
        }
    )

    stacked = source.gs.to_array()

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

    restored = dataset.gs.to_array().gs.to_dataset(dim="band")

    np.testing.assert_array_equal(restored.start_datetime.values, bounds[:, 0])
    np.testing.assert_array_equal(restored.end_datetime.values, bounds[:, 1])
    np.testing.assert_array_equal(restored.red, dataset.red)


def _split_scene() -> xr.Dataset:
    """Two passes of red and nir with periods, item properties and attrs."""
    from tests.geodata.conftest import assign_cells
    from geosave.geodata.stac.raster import stamp_properties_coordinate

    scene = build_raster(times=2)
    days = []
    for value in scene.time.values:
        day = value.astype("datetime64[D]")
        days.append((day, day + np.timedelta64(1, "D")))
    scene = assign_cells(scene, days)
    scene = stamp_properties_coordinate(
        scene, [{"eo:cloud_cover": 10.0}, {"eo:cloud_cover": 40.0}]
    )
    return scene.assign_attrs(title="Two passes")


def _item_ids(scene: xr.Dataset) -> list[str]:
    return list(pd.DatetimeIndex(scene.time.values).strftime("%Y%m%dT%H%M%S"))


def test_a_scene_expands_into_one_item_per_time_and_one_leaf_per_variable():
    scene = _split_scene()

    tree = scene.gs.expand_stac()

    first, second = _item_ids(scene)
    assert [leaf.path for leaf in tree.leaves] == [
        f"/{first}/red",
        f"/{first}/nir",
        f"/{second}/red",
        f"/{second}/nir",
    ]
    tree.gs.validate(stac_pattern=True)


def test_a_variable_without_time_belongs_to_no_item():
    scene = _split_scene()
    scene = scene.assign(height=scene.red.isel(time=0, drop=True))

    with pytest.raises(ValueError, match="'height' has no 'time'.*own collection"):
        scene.gs.expand_stac()


def test_coordinates_go_where_their_dims_say():
    scene = _split_scene()

    tree = scene.gs.expand_stac()

    top = tree.to_dataset(inherit=False)
    item = tree[_item_ids(scene)[1]].to_dataset(inherit=False)
    assert set(top.coords) == {"y", "x", "spatial_ref"}
    assert top.attrs["title"] == "Two passes"
    assert set(item.coords) == {
        "time",
        "start_datetime",
        "end_datetime",
        "stac_properties",
    }
    assert item.time.values == scene.time.values[1]
    for leaf in tree.leaves:
        assert not leaf.to_dataset(inherit=False).coords
        assert not leaf.attrs


def test_the_expansion_needs_no_spatial_names():
    scene = _split_scene().rename(y="latitude", x="longitude")

    tree = scene.gs.expand_stac()

    assert {"latitude", "longitude"} <= set(tree.to_dataset(inherit=False).coords)


def test_ids_follow_format_id_over_the_observations_scalars():
    scene = _split_scene().assign_coords(tile=("time", ["33UUP", "33UUQ"]))

    tree = scene.gs.expand_stac(format_id="S2_{tile}_{time:%Y%m%d}")

    days = pd.DatetimeIndex(scene.time.values).strftime("%Y%m%d")
    assert list(tree.children) == [f"S2_33UUP_{days[0]}", f"S2_33UUQ_{days[1]}"]


def test_ids_that_format_alike_are_refused():
    with pytest.raises(ValueError, match="format to one id"):
        _split_scene().gs.expand_stac(format_id="same")


def test_collections_compose_with_xarrays_constructor():
    scene = _split_scene()

    tree = xr.DataTree(
        children={"s2": scene.gs.expand_stac(), "s1": scene.gs.expand_stac()}
    )

    tree.gs.validate(stac_pattern=True)
    assert tree["s2"].to_dataset(inherit=False).attrs["title"] == "Two passes"


def test_expanding_then_combining_gives_the_identical_scene():
    scene = _split_scene()

    restored = scene.gs.expand_stac().gs.combine_stac()

    xr.testing.assert_identical(restored, scene)


def test_combining_then_expanding_gives_the_identical_tree():
    tree = _split_scene().gs.expand_stac()

    again = tree.gs.combine_stac().gs.expand_stac()

    xr.testing.assert_identical(again, tree)


def test_every_variable_can_share_one_asset():
    scene = _split_scene()

    tree = scene.gs.expand_stac(asset="image")

    first, second = _item_ids(scene)
    assert [leaf.path for leaf in tree.leaves] == [
        f"/{first}/image",
        f"/{second}/image",
    ]
    assert list(tree[f"{first}/image"].data_vars) == ["red", "nir"]
    xr.testing.assert_identical(tree.gs.combine_stac(), scene)


def test_a_timeless_dataset_is_one_item_named_by_format_id():
    scene = build_raster()

    with pytest.raises(ValueError, match="format_id needs 'time'"):
        scene.gs.expand_stac()
    tree = scene.gs.expand_stac(format_id="dem")
    assert [leaf.path for leaf in tree.leaves] == ["/dem/red", "/dem/nir"]


def test_a_round_trip_keeps_packing_encoding():
    scene = build_raster(times=2, packed=True)

    restored = scene.gs.expand_stac().gs.combine_stac()

    assert restored.red.encoding.get("scale_factor") == scene.red.encoding.get(
        "scale_factor"
    )
    assert restored.red.attrs == scene.red.attrs


def test_splitting_reads_no_pixels():
    import dask
    import dask.array as da

    from tests.geodata.conftest import _refuse

    scene = _split_scene()
    for name in scene.data_vars:
        scene[name] = scene[name].copy(data=da.from_array(scene[name].values))

    with dask.config.set(scheduler=_refuse):
        tree = scene.gs.expand_stac()

    for leaf in tree.leaves:
        assert hasattr(leaf.to_dataset()[leaf.name].data, "dask")


def test_decode_matches_native_cf_and_is_idempotent():
    source = build_raster(packed=True).chunk({"x": 1, "y": 1})
    source.red.attrs.update(scale_factor=0.0001, add_offset=-0.1)
    expected = xr.decode_cf(source, decode_coords=False)

    with Callback(pretask=lambda *_: pytest.fail("conversion read pixels")):
        decoded = source.gs.mask_and_scale()

    xr.testing.assert_allclose(decoded, expected)
    assert decoded.red.dtype == expected.red.dtype
    assert isinstance(decoded.red.data, da.Array)
    assert decoded.gs.geobox == source.gs.geobox
    assert "scale_factor" not in decoded.red.attrs
    assert decoded.red.encoding["scale_factor"] == 0.0001
    assert decoded.red.encoding["dtype"] == source.red.dtype
    assert decoded.red.encoding["_FillValue"] == 0
    xr.testing.assert_identical(decoded.gs.mask_and_scale(), decoded)
    assert source.red.attrs["_FillValue"] == 0
    assert "scale_factor" not in source.red.encoding


def test_dataset_decodes_coordinate_packing_and_inherited_time_bounds():
    raw = xr.Dataset(
        {
            "band": (
                ("time", "x"),
                np.array([[0, 10]], dtype="int16"),
                {"_FillValue": 0, "scale_factor": 0.1},
            ),
            "time_bounds": (("time", "vertices"), [[0, 1]]),
        },
        coords={
            "time": (
                "time",
                [0],
                {"units": "days since 2025-01-01", "bounds": "time_bounds"},
            ),
            "x": ("x", [0, 10], {"scale_factor": 0.1, "add_offset": 100.0}),
        },
    )
    before = raw.copy(deep=True)
    decoded = raw.gs.mask_and_scale()

    np.testing.assert_array_equal(decoded.x, [100, 101])
    np.testing.assert_allclose(decoded.band, [[np.nan, 1]], equal_nan=True)
    assert "time_bounds" in decoded.coords
    np.testing.assert_array_equal(
        decoded.time_bounds,
        np.array([["2025-01-01", "2025-01-02"]], dtype="datetime64[ns]"),
    )
    assert decoded.time.encoding["bounds"] == "time_bounds"
    xr.testing.assert_identical(raw, before)


def test_nodata_only_offset_only_and_missing_codes_use_native_decoding():
    source = xr.Dataset(
        {
            "mask": ("x", np.array([0, 7], "uint16"), {"_FillValue": 0}),
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


@pytest.fixture
def city():
    return Geometry(from_wkt("POLYGON ((13 52, 14 52, 13 53, 13 52))"), "EPSG:4326")


def test_extent_survives_band_selection_without_changing_pixels(city):
    source = build_raster().chunk()
    written = source.gs.write_extent(city)

    assert written.red.data is source.red.data
    assert "geospatial_bounds" not in source.attrs
    assert "geospatial_bounds" not in source.red.attrs
    assert written.gs.geobox == source.gs.geobox
    assert written.gs.region.extent == city
    assert written.red.gs.region.extent == city
    assert written.red.isel(x=slice(0, 2)).gs.region.extent == city

    cleared = written.gs.write_extent(None)
    assert cleared.gs.region.extent == source.gs.geobox.extent
    assert cleared.red.gs.region.extent == source.gs.geobox.extent
    assert written.gs.region.extent == city


@pytest.mark.parametrize("kind", ["array", "dataset", "tree"])
@pytest.mark.parametrize("method", ["to_item", "to_catalog"])
def test_publication_refuses_before_pixels_or_files(kind, method, tmp_path):
    source = build_raster().chunk()
    if kind == "array":
        source = source.red
    elif kind == "tree":
        source = xr.DataTree.from_dict({"scene": source})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        with pytest.raises(NotImplementedError, match="being rebuilt"):
            getattr(source.gs, method)(tmp_path / "publication")
    assert tasks == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["array", "dataset"])
def test_packing_keeps_support_and_lazy_pixels(kind):
    source = build_raster(times=0, packed=True).chunk()
    source["x_bounds"] = (
        ("x", "vertices"),
        np.column_stack([source.x - 5, source.x + 5]),
    )
    source.x.attrs["bounds"] = "x_bounds"
    if kind == "array":
        source = source.red
    original = source.copy(deep=True)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        decoded = source.gs.mask_and_scale()
    assert tasks == []
    restored = decoded
    expected = source
    if kind != "array":
        xr.testing.assert_identical(
            restored.x_bounds.variable, expected.x_bounds.variable
        )
        assert "x_bounds" in restored.coords
        assert restored.x.encoding["bounds"] == "x_bounds"
        restored, expected = restored.red, expected.red
    assert isinstance(restored.data, da.Array)
    np.testing.assert_allclose(restored, expected.gs.mask_and_scale(), equal_nan=True)
    assert restored.gs.geobox == expected.gs.geobox
    xr.testing.assert_identical(source, original)
