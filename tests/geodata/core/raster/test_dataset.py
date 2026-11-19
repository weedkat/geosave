from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
import xarray as xr

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


def test_to_tree_keeps_group_metadata_and_lazy_variable_identity():
    source = build_raster(times=2).chunk()
    source.attrs = {"title": "Observation", "rgb": ["red", "nir", "red"]}
    source.red.attrs["units"] = "1"
    source.red.encoding["dtype"] = "uint16"
    before = source.copy(deep=False)

    with Callback(pretask=lambda *_: pytest.fail("grouping read pixels")):
        tree = source.gs.to_tree(name="s2", groups={"image": ["nir"], "red": ["red"]})
        plain = source.gs.to_tree()

    assert list(tree.children) == ["s2"]
    assert list(tree["s2"].children) == ["image", "red"]
    assert tree["s2"].attrs == source.attrs
    assert not tree["s2"].data_vars
    assert not tree["s2"].coords
    assert tree["s2/image"].attrs == {}
    assert tree["s2/red"]["red"].data is source.red.data
    xr.testing.assert_identical(tree["s2/red"].gs.to_dataset(), source[["red"]])
    assert tree["s2/red"]["red"].encoding == source.red.encoding
    xr.testing.assert_identical(plain["ds"].gs.to_dataset(), source)
    xr.testing.assert_identical(source, before)


@pytest.mark.parametrize(
    "groups,message",
    [
        ({}, "missing"),
        ({"image": ["red"]}, "missing"),
        ({"image": ["red", "nir"], "mask": ["red"]}, "more than once"),
        ({"image": ["red", "red", "nir"]}, "more than once"),
        ({"image": ["missing"]}, "Unknown variable"),
        ({"image": []}, "nonempty"),
        ({"image": "red"}, "variable-name list"),
        ({"nested/image": ["red", "nir"]}, "child name"),
    ],
)
def test_to_tree_requires_an_exclusive_complete_partition(groups, message):
    source = build_raster().chunk()

    with Callback(pretask=lambda *_: pytest.fail("validation read pixels")):
        with pytest.raises(ValueError, match=message):
            source.gs.to_tree(groups=groups)


@pytest.mark.parametrize("name", ["", ".", "..", "nested/image"])
def test_to_tree_requires_one_source_node_name(name):
    with pytest.raises(ValueError, match="child name"):
        build_raster().gs.to_tree(name)


def test_to_tree_preserves_auxiliary_coordinates_and_references():
    source = build_raster(times=2).chunk()
    starts = source.time.values
    bounds = np.stack([starts, starts + np.timedelta64(1, "D")], axis=1)
    source = source.assign_coords(time_bounds=(("time", "bounds"), bounds))
    source.time.attrs["bounds"] = "time_bounds"
    source.time_bounds.encoding["dtype"] = "int64"

    with Callback(pretask=lambda *_: pytest.fail("grouping read pixels")):
        tree = source.gs.to_tree(groups={"nir": ["nir"], "red": ["red"]})
        raster = tree["ds"].gs.merge().gs.to_dataset()

    assert list(raster.data_vars) == ["nir", "red"]
    xr.testing.assert_identical(raster, source)
    assert raster.time_bounds.encoding == source.time_bounds.encoding
    assert raster.red.data is source.red.data


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


def test_stacking_requires_explicit_broadcast_for_static_channels() -> None:
    source = optical()
    # A static band of the same pixel meaning, such as a reference image.
    mixed = source.assign(static=source.B04.isel(time=0, drop=True))

    with pytest.raises(ValueError, match="dimensions"):
        mixed.gs.to_numpy()
    prepared = mixed.broadcast_like(source)
    assert prepared.gs.to_numpy().shape == (2, len(mixed.data_vars), 8, 8)


def test_stacking_requires_explicit_dtype_for_mixed_channels() -> None:
    source = optical()
    mixed = source.assign(ndvi=source.B04.astype("float32"))

    with pytest.raises(ValueError, match="dtype"):
        mixed.gs.to_numpy()


def test_to_numpy_casts_variables_onto_one_dtype() -> None:
    source = optical()
    mixed = source.assign(ndvi=source.B04.astype("float32"))

    assert mixed.gs.to_numpy(dtype="float32").dtype == np.dtype("float32")


def test_to_tensor_keeps_the_pixel_dtype_and_the_to_numpy_layout(torch) -> None:
    tensor, context = optical().gs.to_tensor(grid=True, context=["centroid"])

    assert list(context) == ["transform", "crs", "centroid"]

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
    scene = scene.gs.write_stac_property("eo:cloud_cover", [10.0, 40.0])
    return scene.assign_attrs(title="Two passes")


def _item_ids(scene: xr.Dataset) -> list[str]:
    ids = []
    for label in pd.DatetimeIndex(scene.time.values):
        ids.append(f"{label:%Y%m%dT%H%M%S}_49LEL_-9.0465_111.0001")
    return ids


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


def test_one_item_reads_its_property_values_and_declared_extensions():
    from geosave.geodata import attrs

    schema = "https://stac-extensions.github.io/eo/v1.1.0/schema.json"
    scene = build_raster(times=2)
    scene = scene.gs.write_stac_property("eo:cloud_cover", [12.5, float("nan")])
    scene = scene.gs.write_stac_property("platform", "sentinel-2b")
    scene = attrs.stamp_attrs(scene, attrs.StacCollection(extensions=[schema]))

    assert scene.isel(time=0).gs.read_stac_properties() == {
        "eo:cloud_cover": 12.5,
        "platform": "sentinel-2b",
    }
    # A missing value is left out.
    assert scene.isel(time=1).gs.read_stac_properties() == {"platform": "sentinel-2b"}
    with pytest.raises(ValueError):
        scene.gs.read_stac_properties()
    assert scene.gs.stac_extensions == [schema]
    assert build_raster().gs.stac_extensions == []


def test_model_channels_allow_distinct_units_and_require_explicit_dtype():
    source = build_raster()
    source.red.attrs["units"] = "m"
    source.nir.attrs["units"] = "K"
    pixels = source.gs.to_numpy()
    np.testing.assert_array_equal(pixels[0], source.red.values)
    source["nir"] = source.nir.astype("float32")
    with pytest.raises(ValueError, match="dtype"):
        source.gs.to_numpy()
    assert source.gs.to_numpy(dtype="float32").dtype == np.float32
    source["nir"] = source.nir.expand_dims(layer=[1])
    with pytest.raises(ValueError, match="dimensions"):
        source.gs.to_numpy(dtype="float32")


def test_model_stacking_rejects_empty_or_already_named_channel_axis():
    source = build_raster()
    with pytest.raises(ValueError, match="at least one"):
        source.drop_vars(list(source.data_vars)).gs.to_numpy()
    with pytest.raises(ValueError, match="band dimension"):
        source.expand_dims(band=[0]).gs.to_numpy()


def build_observation():
    grid = geobox(shape=(4, 4))
    product = build_dataset(
        {
            "landcover": (("y", "x"), np.ones((4, 4), dtype="uint8")),
            "probability": (("y", "x"), np.full((4, 4), 0.5, dtype="float32")),
        },
        grid,
        time=np.datetime64("2025-06-01", "ns"),
    )
    product.landcover.attrs.update(
        flag_values=np.array([1, 2], "uint8"), flag_meanings="forest water"
    )
    return product


def test_publish_item_writes_one_file_per_variable_by_default(tmp_path):
    from geosave.geodata import stac

    product = build_observation()

    item = product.gs.publish_item(tmp_path, id="forest-20250601")

    assert list(item.assets) == ["landcover", "probability"]
    assert (tmp_path / "forest-20250601/landcover.tif").exists()
    assert item.datetime is not None
    assert stac.create_catalog([item])["id"].tolist() == ["forest-20250601"]
    restored = open_raster(item.assets["landcover"].href, mask_and_scale=False)
    assert restored.gs.geobox == product.gs.geobox


def test_publish_item_groups_variables_and_reads_marked_properties(tmp_path):
    source = build_raster(times=1).isel(time=0)
    source = source.gs.write_stac_property("eo:cloud_cover", 12.5)

    item = source.gs.publish_item(
        tmp_path, id="scene", assets={"image": ["red", "nir"]}
    )

    assert list(item.assets) == ["image"]
    assert len(item.assets["image"].extra_fields["raster:bands"]) == 2
    assert item.properties["eo:cloud_cover"] == 12.5


def test_asset_on_another_grid_joins_the_published_item(tmp_path):
    from geosave.geodata import stac

    product = build_observation()
    coarse = build_dataset(
        {"scl": (("y", "x"), np.full((2, 2), 4, dtype="uint8"))},
        product.gs.geobox.zoom_out(2),
        time=product.time.values,
    )

    item = product.gs.publish_item(
        tmp_path, id="scene", assets={"landcover": ["landcover"]}
    )
    item = stac.publish_item(item, assets={"scl": coarse}, root=tmp_path)

    assert list(item.assets) == ["landcover", "scl"]
    assert item.assets["scl"].extra_fields["proj:shape"] == [2, 2]
    assert item.assets["landcover"].extra_fields["proj:shape"] == [4, 4]


def test_publish_item_refuses_a_time_series(tmp_path):
    with pytest.raises(ValueError):
        build_raster(times=2).gs.publish_item(tmp_path, id="series")
    assert not list(tmp_path.rglob("*.tif"))
