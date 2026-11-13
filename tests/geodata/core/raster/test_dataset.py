from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geosave.geodata.core.raster.datatree import stac_catalog
from geosave.geodata import (
    attrs,
    open_raster,
)
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

    stacked = built.gs.to_array()

    assert stacked.band.values.tolist() == ["B04", "B03"]
    assert stacked.name is None
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

    box = geobox(shape=(2, 2))
    values = da.from_delayed(pixels(), shape=box.shape, dtype="uint16")
    source = build_dataset(
        {"red": (("y", "x"), values), "nir": (("y", "x"), values)}, box
    )
    source.nir.attrs.update(attributes)

    with pytest.raises(ValueError, match="CF packing"):
        source.gs.to_array()


@pytest.mark.parametrize(
    "left,right",
    [
        ({"units": "K"}, {"units": "m"}),
        (
            {"flag_values": [0, 1], "flag_meanings": "water land"},
            {"flag_values": [1, 2], "flag_meanings": "water land"},
        ),
    ],
    ids=["units", "class-codes"],
)
def test_stacking_rejects_different_pixel_meaning_before_reading(left, right):
    from dask.callbacks import Callback

    source = build_raster().chunk()
    source.red.attrs.update(left)
    source.nir.attrs.update(right)
    with Callback(pretask=lambda *_: pytest.fail("validation read pixels")):
        with pytest.raises(ValueError, match="different.*(packing|flags|units)"):
            source.gs.to_array()


def test_nodata_reads_back_what_was_written() -> None:
    box = geobox()
    pixels = np.zeros(box.shape, "uint16")
    built = build_dataset(
        {"red": (("y", "x"), pixels), "nir": (("y", "x"), pixels)},
        box,
    ).gs.write_nodata(0, target=["red"])

    assert built.red.gs.nodata == 0
    assert built.nir.gs.nodata is None


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
    assert legend.color_hints == ["419BDF", "397D49"]
    assert labelled.other.gs.flags is None


@pytest.mark.parametrize(
    "method,value",
    [
        ("write_nodata", 255),
        ("write_flags", NAMES),
        ("write_packing", attrs.CFPacking(scale_factor=0.1)),
        ("write_spectral", "red"),
    ],
)
def test_metadata_target_rejects_a_bare_name_without_reading(method, value):
    source = _classes().chunk()
    with Callback(pretask=lambda *_: pytest.fail("metadata selection read pixels")):
        with pytest.raises(TypeError, match=r"target=\['label'\]"):
            getattr(source.gs, method)(value, target="label")


def test_empty_metadata_target_leaves_every_variable_unchanged():
    source = _classes()
    written = source.gs.write_nodata(255, target=[])

    xr.testing.assert_identical(written, source)


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


def test_a_target_that_is_not_a_data_variable_is_refused() -> None:
    with pytest.raises(KeyError, match="blue"):
        _bands().gs.write_nodata(0, target=["blue"])


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


def test_stacking_broadcasts_a_static_band_along_time() -> None:
    source = optical()
    # A static band of the same pixel meaning, such as a reference image.
    mixed = source.assign(static=source.B04.isel(time=0, drop=True))

    stacked = mixed.gs.to_numpy()

    assert stacked.shape == (2, len(mixed.data_vars), 8, 8)


def test_stacking_promotes_dtypes_as_numpy_does() -> None:
    source = optical()
    mixed = source.assign(ndvi=source.B04.astype("float32"))

    assert mixed.gs.to_numpy().dtype == np.float32


def test_to_numpy_casts_variables_onto_one_dtype() -> None:
    source = optical()
    mixed = source.assign(ndvi=source.B04.astype("float32"))

    assert mixed.gs.to_numpy(dtype="float32").dtype == np.dtype("float32")


def test_to_tensor_keeps_the_pixel_dtype_and_the_to_numpy_layout(torch) -> None:
    tensor = optical().gs.to_tensor()

    assert tensor.dtype is torch.uint16
    assert tuple(tensor.shape) == (2, 2, 8, 8)
    assert tensor.is_contiguous()
    torch.testing.assert_close(tensor, torch.tensor(optical().gs.to_numpy()))


def build_bands(*colours: str | None) -> xr.Dataset:
    """Bands whose variable names do not hint at their spectral common names."""
    pixels = xr.DataArray(np.ones((2, 2), "uint8"), dims=("y", "x"))
    raster = xr.Dataset({f"b{band}": pixels for band, _ in enumerate(colours)})
    for band, colour in enumerate(colours):
        if colour is not None:
            raster[f"b{band}"].attrs |= attrs.EOBand(common_name=colour).to_attrs()
    return raster


def test_rgb_is_none_where_no_composite_is_stated() -> None:
    # Band colours alone state no composite; loading or write_rgb states it.
    assert build_bands("blue", "red", "green").gs.rgb is None


def test_rgb_is_none_once_a_named_variable_is_dropped() -> None:
    written = build_bands("red", "green", "blue").gs.write_rgb("b0", "b1", "b2")

    assert written[["b0", "b1"]].gs.rgb is None


def test_a_plot_without_bands_refuses_a_raster_drawing_no_colour() -> None:
    with pytest.raises(ValueError, match="write_rgb"):
        build_bands(None, "red", "green").gs.plot()


def test_write_rgb_repoints_the_channels_it_replaces() -> None:
    raster = build_bands("red", "green", "blue", None)

    written = raster.gs.write_rgb("b3", "b0", "b1")

    assert written.gs.rgb == ("b3", "b0", "b1")
    # An explicit display overrides spectral roles without rewriting them.
    assert written.b2.gs.spectral.common_name == "blue"


def test_write_rgb_refuses_a_name_the_raster_does_not_carry() -> None:
    with pytest.raises(KeyError, match="b2"):
        build_bands(None, None).gs.write_rgb("b0", "b1", "b2")


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

    scene = build_raster(times=2)
    days = []
    for value in scene.time.values:
        day = value.astype("datetime64[D]")
        days.append((day, day + np.timedelta64(1, "D")))
    scene = assign_cells(scene, days)
    properties = [{"eo:cloud_cover": 10.0}, {"eo:cloud_cover": 40.0}]
    scene = scene.assign_coords(stac_properties=("time", properties))
    return scene.assign_attrs(title="Two passes")


def _item_ids(scene: xr.Dataset) -> list[str]:
    ids = []
    for label in pd.DatetimeIndex(scene.time.values):
        ids.append(f"{label:%Y%m%dT%H%M%S}_49LEL_-9.0465_111.0001")
    return ids


def test_a_scene_expands_into_one_item_per_time_and_one_leaf_per_variable():
    scene = _split_scene()

    tree = scene.gs.expand()

    first, second = _item_ids(scene)
    # A Dataset stating no Collection lands under the default collection.
    assert [leaf.path for leaf in tree.leaves] == [
        f"/default/{first}/red",
        f"/default/{first}/nir",
        f"/default/{second}/red",
        f"/default/{second}/nir",
    ]
    tree.gs.validate(stac_catalog)


def test_a_variable_without_time_is_left_out():
    scene = _split_scene()
    scene = scene.assign(height=scene.red.isel(time=0, drop=True))

    tree = scene.gs.expand()

    for leaf in tree.leaves:
        assert leaf.name != "height"


def test_coordinates_and_attrs_go_to_their_stac_level():
    collection = attrs.StacCollection(collection={"id": "s2"})
    scene = _split_scene().assign_attrs(collection.to_attrs())

    tree = scene.gs.expand()

    # Catalog: nothing of its own.
    assert not tree.to_dataset(inherit=False).variables
    assert not tree.attrs

    # Collection: the STAC Collection alone, named by its id.
    top = tree["s2"].to_dataset(inherit=False)
    assert not top.coords
    assert top.attrs == collection.to_attrs()

    # Item: its coordinates along time, and the Dataset's other attrs.
    item = tree["s2"][_item_ids(scene)[1]].to_dataset(inherit=False)
    assert set(item.coords) == {
        "time",
        "start_datetime",
        "end_datetime",
        "stac_properties",
    }
    assert item.time.values == scene.time.values[1]
    assert item.attrs == {"title": "Two passes"}

    # Asset: one variable with its own attrs and its grid.
    for leaf in tree.leaves:
        asset = leaf.to_dataset(inherit=False)
        assert set(asset.coords) == {"y", "x", "spatial_ref"}
        assert asset.spatial_ref.attrs == scene.spatial_ref.attrs
        assert not asset.attrs


def test_items_are_named_by_their_start_and_place():
    tree = _split_scene().gs.expand()

    assert list(tree["default"].children) == [
        "20250601T000000_49LEL_-9.0465_111.0001",
        "20250602T000000_49LEL_-9.0465_111.0001",
    ]


def test_items_are_named_by_a_given_id_format():
    tree = _split_scene().gs.expand(id_format="s2_{start:%Y%m%d}_{mgrs}")

    assert list(tree["default"].children) == ["s2_20250601_49LEL", "s2_20250602_49LEL"]


def test_two_items_formatting_to_one_id_are_refused():
    with pytest.raises(ValueError, match="two Items format to the id"):
        _split_scene().gs.expand(id_format="{mgrs}")


def test_expanding_then_combining_gives_the_identical_scene():
    scene = _split_scene()

    restored = scene.gs.expand()["default"].gs.combine()

    xr.testing.assert_identical(restored, scene)


def test_combine_refuses_an_asset_whose_packing_changes_between_items():
    from geosave.geodata.errors import RasterValidationError

    scene = build_raster(times=2, packed=True)
    tree = scene.gs.expand()
    first, second = tree["default"].children
    tree[f"default/{second}/red"]["red"].attrs["add_offset"] = -0.1

    with pytest.raises(RasterValidationError, match="another CF packing"):
        tree["default"].gs.combine()


def test_a_dataset_without_the_item_dim_is_refused():
    with pytest.raises(KeyError, match="time"):
        build_raster().gs.expand()


def test_a_dim_without_time_needs_an_id_format_it_can_fill():
    scene = (
        _split_scene().rename(time="date").drop_vars(["start_datetime", "end_datetime"])
    )

    # Ids are formatted from the region, whose period only `time` states.
    with pytest.raises(ValueError, match="no 'start'"):
        scene.gs.expand("date")


def test_a_round_trip_keeps_packing_encoding():
    scene = build_raster(times=2, packed=True)

    restored = scene.gs.expand()["default"].gs.combine()

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
        tree = scene.gs.expand()

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
