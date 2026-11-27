from __future__ import annotations


import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geosave.geodata import open_raster_tree
from geosave.geodata.errors import RasterValidationError
from tests.conftest import build_tree
from geosave.geodata.io import storage
from tests.geodata.conftest import (
    _refuse,
    build_raster,
)
from tests.conftest import place

from shapely import from_wkt
from odc.geo.geom import Geometry
from shapely import Polygon


def test_to_dataset_inherits_coordinates_and_attrs_without_reading_pixels():
    scene = build_raster(times=2).chunk()
    scene.red.attrs["long_name"] = "Red"
    scene.red.encoding["dtype"] = "uint16"

    shared = xr.Dataset(coords=scene.coords).assign_coords(platform="sentinel-2")
    shared.platform.attrs["stac_property"] = "platform"
    shared.attrs = {"license": "root-license", "institution": "Survey"}

    pixels = scene[["red"]].drop_vars(list(scene.coords))
    pixels.attrs = {"title": "Reflectance"}
    pixels.encoding["source"] = "memory"

    tree = xr.DataTree.from_dict(
        {
            "/": shared,
            "s2": xr.Dataset(attrs={"license": "CC-BY-4.0", "title": "Source"}),
            "s2/image": pixels,
        }
    )
    node = tree["s2/image"]
    before = tree.copy(deep=False)

    with dask.config.set(scheduler=_refuse):
        raster = node.gs.to_dataset()

    assert raster.attrs == {
        "license": "CC-BY-4.0",
        "institution": "Survey",
        "title": "Reflectance",
    }
    assert raster.gs.geobox == scene.gs.geobox
    xr.testing.assert_identical(raster.time.variable, scene.time.variable)
    assert raster.platform.item() == "sentinel-2"
    assert raster.platform.attrs == {"stac_property": "platform"}
    assert raster.red.data is node["red"].data
    assert raster.red.attrs == scene.red.attrs
    assert raster.red.encoding == scene.red.encoding
    assert raster.encoding == pixels.encoding

    assert node.to_dataset(inherit="all_coords").attrs == {"title": "Reflectance"}
    xr.testing.assert_identical(tree, before)


def test_to_dataset_resolves_parent_updates_on_each_call():
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"license": "old"}),
            "image": build_raster(),
        }
    )
    node = tree["image"]
    first = node.gs.to_dataset()

    tree.attrs["license"] = "new"
    second = node.gs.to_dataset()

    assert first.attrs["license"] == "old"
    assert second.attrs["license"] == "new"

    second.attrs["license"] = "export-only"
    assert tree.attrs["license"] == "new"
    assert "license" not in node.attrs


def test_to_dataset_only_materializes_the_selected_node():
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                coords={"platform": "sentinel-2"}, attrs={"title": "Survey"}
            ),
            "image": build_raster(),
        }
    )

    root = tree.gs.to_dataset()

    assert not root.data_vars
    assert root.platform.item() == "sentinel-2"
    assert root.attrs == {"title": "Survey"}


def test_to_dataset_supports_a_root_raster():
    scene = build_raster().chunk()
    tree = xr.DataTree(scene)

    with dask.config.set(scheduler=_refuse):
        raster = tree.gs.to_dataset()

    xr.testing.assert_identical(raster, scene)
    assert raster.red.data is tree["red"].data


def test_grouped_dataset_merges_back_without_reading_or_casting_pixels():
    source = build_raster(times=2).chunk()
    source.attrs = {"title": "Source"}
    source["nir"] = source.nir.astype("float32")
    source.red.attrs["units"] = "1"
    source.nir.attrs["units"] = "m"
    source.red.encoding["scale_factor"] = 0.1
    tree = source.gs.to_tree("s2", groups={"red": ["red"], "nir": ["nir"]})
    before = tree.copy(deep=False)

    with dask.config.set(scheduler=_refuse):
        merged = tree["s2"].gs.merge()

    assert merged.name == "s2"
    assert not merged.children
    xr.testing.assert_identical(merged.gs.to_dataset(), source)
    assert merged["red"].data is source.red.data
    assert merged["nir"].dtype == source.nir.dtype
    assert merged["red"].encoding == source.red.encoding
    xr.testing.assert_identical(tree, before)


def test_group_existing_leaf_preserves_parent_metadata_and_variable_order():
    source = build_raster(times=2).chunk()
    source.red.encoding["dtype"] = "uint16"
    tree = xr.DataTree.from_dict(
        {"/": xr.Dataset(attrs={"license": "CC0"}), "s2": source}
    )
    before = tree.copy(deep=False)

    with dask.config.set(scheduler=_refuse):
        split = tree["s2"].gs.group({"nir": ["nir"], "red": ["red"]})
        rebuilt = split.gs.merge().gs.to_dataset()

    assert split.name == "s2"
    assert split.parent is None
    assert list(split.children) == ["nir", "red"]
    assert split.attrs == {"license": "CC0"}
    assert split["red"].attrs == {}
    assert not split.coords
    assert set(source.coords) <= set(split["red"].coords)
    assert split["red"]["red"].data is source.red.data
    assert rebuilt.red.encoding == source.red.encoding
    xr.testing.assert_identical(rebuilt, source.assign_attrs(license="CC0"))
    xr.testing.assert_identical(tree, before)


def test_group_and_split_have_separate_call_contracts():
    leaf = build_raster(times=2).gs.to_tree()["ds"]

    with pytest.raises(TypeError, match="dim"):
        leaf.gs.split()
    with pytest.raises(TypeError, match="groups"):
        leaf.gs.split(groups={"image": ["red", "nir"]})

    split = leaf.gs.split("time", names=["first", "second"])
    assert list(split["time"].children) == ["first", "second"]


def test_dimension_split_and_concat_preserve_repeated_positions_and_static_data():
    source = build_raster(times=2).chunk()
    source = source.assign_coords(cloud=("time", [10, 20]))
    source = source.assign_coords(time=source.time.isel(time=[0, 0]).values)
    source["height"] = source.red.isel(time=0, drop=True)
    source.attrs["title"] = "Observation series"
    source.red.encoding["dtype"] = "uint16"
    root = source.gs.to_tree("s2")

    with dask.config.set(scheduler=_refuse):
        split = root["s2"].gs.split(dim="time", names=["first", "second"])

    assert list(split.children) == ["static", "time"]
    assert not split.coords
    assert not split["time"].coords
    assert "time" not in split["static"].gs.to_dataset().coords
    assert "cloud" not in split["static"].gs.to_dataset().coords
    assert list(split["time"].children) == ["first", "second"]
    assert split["time/first"].gs.to_dataset().time.ndim == 0
    assert split["time/second"].gs.to_dataset().cloud.item() == 20
    assert set(split["time/first"].data_vars) == {"red", "nir"}

    with dask.config.set(scheduler=_refuse):
        split["time"] = split["time"].gs.concat(dim="time")
        rebuilt = split.gs.merge().gs.to_dataset()

    xr.testing.assert_identical(rebuilt, source)
    assert rebuilt.height.dims == ("y", "x")
    assert rebuilt.red.encoding == source.red.encoding
    assert root["s2"].gs.to_dataset().red.data is source.red.data


def test_ungroup_preserves_nested_metadata_coordinates_and_sibling_order():
    source = build_raster().chunk()
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"license": "CC0"}),
            "s2": xr.Dataset(
                coords={"time": np.datetime64("2025-06-01", "ns")},
                attrs={"title": "Source"},
            ),
            "s2/image": xr.Dataset(attrs={"title": "Image"}),
            "s2/image/red": source[["red"]],
            "s2/nir": source[["nir"]],
            "terrain": source[["red"]].rename(red="height"),
        }
    )
    expected = tree["s2/image/red"].gs.to_dataset()
    before = tree.copy(deep=False)

    with dask.config.set(scheduler=_refuse):
        result = tree.gs.ungroup("s2")

    assert list(result.children) == ["image", "nir", "terrain"]
    xr.testing.assert_identical(result["image/red"].gs.to_dataset(), expected)
    assert result["nir"].gs.to_dataset().attrs["title"] == "Source"
    assert "title" not in result["terrain"].gs.to_dataset().attrs
    assert "time" not in result["terrain"].gs.to_dataset().coords
    assert result["image/red"]["red"].data is source.red.data
    result.attrs["license"] = "updated"
    assert result["nir"].gs.to_dataset().attrs["license"] == "updated"
    xr.testing.assert_identical(tree, before)


@pytest.mark.parametrize("difference", ["name", "grid", "time", "scalar"])
def test_merge_rejects_incompatible_children_without_reading_pixels(difference):
    source = build_raster(times=2).chunk()
    left = source[["red"]]
    right = source[["nir"]]
    if difference == "name":
        right = right.rename(nir="red")
    elif difference == "grid":
        right = right.assign_coords(x=right.x + 1)
    elif difference == "time":
        right = right.assign_coords(time=right.time + np.timedelta64(1, "D"))
    else:
        left = left.assign_coords(platform="left")
        right = right.assign_coords(platform="right")

    tree = xr.DataTree.from_dict({"left": left, "right": right})

    with dask.config.set(scheduler=_refuse):
        with pytest.raises(ValueError):
            tree.gs.merge(combine_attrs="drop_conflicts")


def test_merge_uses_native_attribute_policy_and_requires_immediate_rasters():
    source = build_raster()
    tree = source.gs.to_tree("s2", groups={"red": ["red"], "nir": ["nir"]})
    tree.attrs["license"] = "CC0"
    tree["s2/red"].attrs["title"] = "Red"
    tree["s2/nir"].attrs["title"] = "NIR"

    with pytest.raises(ValueError, match="immediate raster-leaf"):
        tree.gs.merge()
    with pytest.raises(ValueError, match="immediate raster-leaf"):
        tree["s2/red"].gs.merge()
    with pytest.raises(ValueError, match="combine_attrs"):
        tree["s2"].gs.merge()

    result = tree["s2"].gs.merge(combine_attrs="drop_conflicts")

    assert result.attrs == {"license": "CC0"}
    assert set(result.data_vars) == {"red", "nir"}


def test_ungroup_rejects_collisions_and_only_accepts_a_direct_group():
    source = build_raster()
    tree = xr.DataTree.from_dict({"s2/red": source, "red": source})
    before = tree.copy(deep=False)

    with pytest.raises(ValueError, match="already exist"):
        tree.gs.ungroup("s2")
    with pytest.raises(ValueError, match="must be a group"):
        tree.gs.ungroup("red")
    with pytest.raises(KeyError):
        tree.gs.ungroup("s2/red")

    xr.testing.assert_identical(tree, before)


def test_ungroup_selected_subtree_preserves_external_coordinates_and_attrs():
    source = build_raster().chunk()
    coordinates = xr.Dataset(
        coords={"platform": "sentinel-2"}, attrs={"license": "CC0"}
    )
    tree = xr.DataTree.from_dict({"/": coordinates, "area/s2/image": source})
    expected = tree["area/s2/image"].gs.to_dataset()

    with dask.config.set(scheduler=_refuse):
        result = tree["area"].gs.ungroup("s2")

    assert result.name == "area"
    assert result.parent is None
    xr.testing.assert_identical(result["image"].gs.to_dataset(), expected)
    assert result["image"]["red"].data is source.red.data


@pytest.mark.parametrize("method,suffix", [("to_netcdf", ".nc"), ("to_zarr", ".zarr")])
def test_grouped_tree_store_preserves_metadata_ownership_and_merge(
    tmp_path, method, suffix
):
    source = build_raster(times=2)
    source.attrs = {"title": "Observation"}
    source.red.attrs["units"] = "1"
    tree = source.gs.to_tree("s2", groups={"red": ["red"], "nir": ["nir"]})

    path = getattr(tree.gs, method)(tmp_path / f"grouped{suffix}")

    with open_raster_tree(path) as restored:
        assert restored["s2"].attrs == source.attrs
        assert restored["s2/red"].attrs == {}
        raster = restored["s2"].gs.merge().gs.to_dataset()
        xr.testing.assert_identical(raster, source)
        assert raster.gs.geobox == source.gs.geobox


def build_stac_layout():
    """One collection, one Item, two assets: a STAC tree layout."""
    scene = build_raster()
    return xr.DataTree.from_dict(
        {
            "s2/item-a": xr.Dataset(coords={"time": np.datetime64("2025-06-01", "ns")}),
            "s2/item-a/red": scene[["red"]],
            "s2/item-a/nir": scene[["nir"]],
        }
    )


def test_georeferenced_validation_checks_each_node():
    from geosave.geodata.errors import RasterValidationError

    scene = build_raster()
    tree = xr.DataTree.from_dict({"placed": scene})
    tree.gs.validate()

    pixels = xr.Dataset({"red": (("y", "x"), np.ones((2, 2)))})
    tree["nested/pixels"] = xr.DataTree(pixels)

    with pytest.raises(RasterValidationError) as caught:
        tree.gs.validate()

    assert caught.value.path == "nested/pixels"


def test_tree_has_independent_rasters_without_single_raster_properties():
    from odc.geo.geobox import GeoBox
    from geosave.geodata.core.raster.base import GeoRaster

    projected = build_raster(times=1)
    geographic = build_raster(times=2).pipe(
        place, GeoBox.from_bbox((111, -9, 111.01, -8.99), crs="EPSG:4326", shape=(2, 2))
    )
    tree = build_tree({"projected": projected, "geographic": geographic})
    tree.gs.validate()

    assert not isinstance(tree.gs, GeoRaster)
    for name in ("crs", "transform", "region"):
        assert not hasattr(tree.gs, name)
    # The shared grid and timeline are readable only when the leaves share them.
    with pytest.raises(RasterValidationError, match="GeoBoxes"):
        tree.gs.geobox
    with pytest.raises(RasterValidationError, match="GeoBoxes"):
        tree.gs.spatial_dims
    with pytest.raises(ValueError, match="one timeline"):
        tree.gs.times

    assert tree["projected"].gs.to_dataset().gs.geobox == projected.gs.geobox
    assert tree["geographic"].gs.to_dataset().gs.geobox == geographic.gs.geobox


def test_a_tree_timespan_covers_every_dated_leaf_on_its_own_timeline():
    early = build_raster(times=2)
    late = build_raster(times=1).assign_coords(time=[pd.Timestamp("2031-01-01")])
    static = build_raster()
    tree = build_tree({"early": early, "late": late, "static": static})

    start, _ = early.gs.timespan
    assert tree.gs.timespan == (start, pd.Timestamp("2031-01-01").to_pydatetime())
    assert build_tree({"static": static}).gs.timespan is None


def test_a_tree_timespan_reads_the_periods_its_leaves_state():
    monthly = build_raster(times=1).gs.write_time_bounds("MS")
    tree = build_tree({"image": monthly})

    assert tree.gs.timespan == monthly.gs.timespan


@pytest.mark.integration
def test_a_tree_round_trips_through_a_upath_carrying_its_options(remote) -> None:
    prefix, options = remote
    folder = storage.create_upath(f"{prefix}/s0", options)
    sample = build_tree(
        {"optical": build_raster(times=2), "label": build_raster(times=1)}
    )

    saved = sample.gs.to_zarr(folder / "scene.zarr")

    with open_raster_tree(saved) as opened:
        assert tuple(opened.children) == ("optical", "label")


@pytest.mark.parametrize("method", ["to_netcdf", "to_zarr"])
def test_a_deferred_write_computes_to_the_eager_result(tmp_path, method):
    from dask.delayed import Delayed

    from dask.callbacks import Callback

    source = build_raster().chunk({"y": 1, "x": 1})
    tree = xr.DataTree.from_dict({"image": source, "label": source})
    suffix = ".nc" if method == "to_netcdf" else ".zarr"
    tasks = []
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        pending = getattr(tree.gs, method)(tmp_path / f"scene{suffix}", compute=False)
    assert tasks == []
    assert isinstance(pending, Delayed)
    assert list(tmp_path.iterdir()) == []

    saved = pending.compute()

    assert saved == tmp_path / f"scene{suffix}"
    with open_raster_tree(saved) as restored:
        np.testing.assert_array_equal(
            restored["label"].to_dataset().red,
            tree["label"].to_dataset().red,
        )


def build_stac_catalog():
    """Two Items of one collection, each with a 10 m and a 20 m asset, plus a timeless DEM."""
    from tests.conftest import build_dataset
    from tests.geodata.conftest import geobox as make_geobox

    fine = make_geobox(shape=(4, 4))
    coarse = fine.zoom_out(2)
    nodes = {"s2": xr.Dataset(attrs={"title": "s2"})}
    for day, value in (("2025-06-01", 1.0), ("2025-06-06", 2.0)):
        when = np.datetime64(day, "ns")
        nodes[f"s2/{day}"] = xr.Dataset(coords={"time": when})
        nodes[f"s2/{day}/B04"] = build_dataset(
            {"B04": (("y", "x"), np.full((4, 4), value))}, fine
        )
        nodes[f"s2/{day}/B11"] = build_dataset(
            {"B11": (("y", "x"), np.full((2, 2), value))}, coarse
        )
    height = build_dataset({"height": (("y", "x"), np.zeros((4, 4)))}, fine)
    nodes["dem/cop/height"] = height
    return xr.DataTree.from_dict(nodes)


def test_an_item_merges_its_assets_and_keeps_its_time_scalar():
    tree = build_stac_catalog()
    grid = tree["s2/2025-06-01/B04"].gs.to_dataset().gs.geobox

    item = tree.gs.reproject(grid)["s2/2025-06-01"]
    assets = []
    for leaf in item.leaves:
        assets.append(leaf.to_dataset(inherit="all_coords"))
    dataset = xr.merge(assets, join="exact")

    assert set(dataset.data_vars) == {"B04", "B11"}
    assert dataset.time.ndim == 0
    assert dataset.gs.geobox == grid


def test_reproject_moves_every_leaf_lazily():
    tree = build_stac_catalog().chunk()
    grid = tree["s2/2025-06-01/B11"].gs.to_dataset().gs.geobox

    with dask.config.set(scheduler=_refuse):
        moved = tree.gs.reproject(grid)

    for leaf in moved.leaves:
        assert leaf.to_dataset(inherit="all_coords").gs.geobox == grid
    assert moved["s2"].attrs == {"title": "s2"}


def test_composite_reduces_each_dated_node_on_its_own_timeline_lazily():
    optical = build_raster(times=3).astype("float32")
    radar = optical.isel(time=[0, 2]).assign_coords(
        time=pd.to_datetime(["2025-06-02", "2025-07-02"])
    )
    labels = build_raster()[["red"]].rename(red="cover")
    tree = xr.DataTree.from_dict(
        {"s2": optical.chunk(), "s1": radar.chunk(), "labels": labels}
    )

    with dask.config.set(scheduler=_refuse):
        monthly = tree.gs.composite("MS", "median")

    assert (
        monthly["s2"].time.values.tolist()
        == pd.to_datetime(["2025-06-01"]).values.tolist()
    )
    assert monthly["s1"].time.size == 2
    xr.testing.assert_identical(monthly["labels"].to_dataset(), labels)
    expected = optical.gs.composite("MS", "median")
    xr.testing.assert_allclose(monthly["s2"].to_dataset().compute(), expected.compute())


def test_mask_and_scale_decodes_each_packed_node_lazily():
    packed = build_raster(times=2, packed=True).chunk()
    codes = build_raster(times=2)[["red"]].rename(red="scl").chunk()
    tree = xr.DataTree.from_dict({"s2/reflectance": packed, "s2/quality": codes})
    tree["s2"].attrs["title"] = "s2"

    with dask.config.set(scheduler=_refuse):
        decoded = tree.gs.mask_and_scale()

    expected = packed.gs.mask_and_scale()
    xr.testing.assert_identical(decoded["s2/reflectance"].to_dataset(), expected)
    assert decoded["s2/quality"].scl.dtype == "uint16"
    assert decoded["s2"].attrs == {"title": "s2"}
    monthly = decoded.gs.composite("MS", "median")
    assert monthly["s2/reflectance"].red.dtype == expected.red.dtype


def test_composite_refuses_what_dataset_composite_refuses():
    packed = xr.DataTree.from_dict({"s2": build_raster(times=2, packed=True)})
    with pytest.raises(ValueError, match="mask_and_scale"):
        packed.gs.composite("MS", "median")

    shared = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"time": build_raster(times=2).time}),
            "s2": build_raster(times=2).drop_vars("time"),
        }
    )
    with pytest.raises(ValueError, match="inherits `time`"):
        shared.gs.composite("MS", "median")


def test_a_reprojected_tree_chips_every_leaf_by_one_window():
    tree = build_stac_catalog()
    moved = tree.gs.reproject(tree["s2/2025-06-01/B04"].gs.to_dataset().gs.geobox)

    chip = moved.isel(y=slice(0, 2), x=slice(0, 2))

    for leaf in chip.leaves:
        assert leaf.to_dataset(inherit="all_coords")[leaf.name].shape == (2, 2)
    times = chip["s2/2025-06-06"].to_dataset(inherit=False).time.values
    assert times == np.datetime64("2025-06-06")


@pytest.fixture
def city():
    return Geometry(from_wkt("POLYGON ((13 52, 14 52, 13 53, 13 52))"), "EPSG:4326")


def test_a_dataset_unions_variable_extents(city):
    left = build_raster()[["red"]].gs.write_extent(city)
    other = Geometry(from_wkt("POLYGON ((15 52, 16 52, 15 53, 15 52))"), "EPSG:4326")
    right = build_raster()[["nir"]].gs.write_extent(other.to_crs("EPSG:3857"))
    dataset = xr.Dataset({"red": left.red, "nir": right.nir})
    expected = city.union(other)
    assert dataset.gs.region.extent.geom.equals_exact(expected.geom, 1e-8)


def test_tree_requires_raster_leaves_and_accepts_a_root_raster():
    from geosave.geodata.errors import RasterValidationError

    scene = build_raster()
    xr.DataTree(scene).gs.validate()
    tree = xr.DataTree.from_dict({"group": scene, "group/child": xr.Dataset()})

    with pytest.raises(RasterValidationError, match="leaves") as error:
        tree.gs.validate()
    assert error.value.path == "group"


def test_empty_tree_is_not_a_raster_tree():
    from geosave.geodata.errors import RasterValidationError

    with pytest.raises(RasterValidationError, match="no raster variables"):
        xr.DataTree().gs.validate()


def test_a_tree_converts_to_tensors_that_carry_their_grid():
    torch = pytest.importorskip("torch")
    from affine import Affine
    from odc.geo.geobox import GeoBox

    scene = build_raster(times=2).astype("float32")
    mask = (build_raster()[["red"]] > 1500).astype("uint8").rename(red="cover")
    tree = xr.DataTree.from_dict({"image": scene, "label": mask})

    pixels, context = tree.gs.to_tensor(context={"transform": "transform", "crs": "crs"})

    assert list(pixels) == ["image", "label"]
    assert list(context) == ["transform", "crs"]
    assert pixels["image"].shape == (2, 2, 2, 2)
    assert pixels["image"].dtype == torch.float32
    assert pixels["label"].shape == (1, 2, 2)
    assert pixels["label"].dtype == torch.uint8
    assert context["crs"].item() == 32749
    grid = GeoBox(
        pixels["image"].shape[-2:],
        Affine(*context["transform"].tolist()),
        int(context["crs"]),
    )
    assert grid == scene.gs.geobox

    batch = torch.utils.data.default_collate([context, context])
    assert batch["transform"].shape == (2, 6)
    assert batch["crs"].tolist() == [32749, 32749]


def test_context_names_add_their_readers_tensors():
    torch = pytest.importorskip("torch")
    from geosave.geodata import CONTEXT_READERS

    scene = build_raster(times=2)
    tree = xr.DataTree.from_dict({"image": scene})

    _, context = tree.gs.to_tensor(context={"year": "year", "centroid": "centroid"})

    assert list(context) == ["year", "centroid"]
    torch.testing.assert_close(
        context["centroid"], torch.from_numpy(CONTEXT_READERS["centroid"](tree))
    )
    with pytest.raises(KeyError, match="is not one of"):
        tree.gs.to_tensor(context={"elevation": "elevation"})


def test_a_tree_reads_pixels_by_node_path_and_context_by_key():
    from geosave.geodata import CONTEXT_READERS

    scene = build_raster(times=2).astype("float32")
    mask = (build_raster()[["red"]] > 1500).astype("uint8").rename(red="cover")
    tree = xr.DataTree.from_dict({"image": scene, "label": mask})

    pixels, context = tree.gs.to_numpy(context={"latlon": "centroid", "week": "week"})

    assert list(pixels) == ["image", "label"]
    assert list(context) == ["latlon", "week"]
    image = scene.gs.to_numpy()
    label = mask.gs.to_numpy()
    np.testing.assert_array_equal(pixels["image"], image)
    np.testing.assert_array_equal(pixels["label"], label)
    np.testing.assert_array_equal(context["latlon"], CONTEXT_READERS["centroid"](tree))
    assert context["week"].shape == (2,)


def test_a_tree_sample_reads_its_nodes_in_one_computation():
    scene = build_raster(times=2).chunk()
    tree = xr.DataTree.from_dict({"image": scene, "brightness": scene.mean("time")})
    computations = []

    def count(*args, **kwargs):
        computations.append(args)
        return dask.get(*args, **kwargs)

    with dask.config.set(scheduler=count):
        pixels = tree.gs.to_numpy()

    assert len(computations) == 1
    assert pixels["brightness"].shape == (2, 2, 2)


CUSTOM_CRS = "+proj=utm +zone=49 +south +ellps=GRS80 +towgs84=1,2,3 +units=m"


def test_tensors_need_one_grid_only_for_shared_spatial_context():
    pytest.importorskip("torch")
    from geosave.geodata.errors import RasterValidationError

    scene = build_raster()
    coarse = scene.coarsen(x=2, y=2).mean()
    mixed = xr.DataTree.from_dict({"image": scene, "coarse": coarse})
    pixels = mixed.gs.to_tensor()
    assert pixels["image"].shape[-2:] == (2, 2)
    assert pixels["coarse"].shape[-2:] == (1, 1)
    assert list(pixels) == ["image", "coarse"]
    with dask.config.set(scheduler=_refuse):
        with pytest.raises(RasterValidationError, match="GeoBoxes"):
            mixed.chunk().gs.to_tensor(context={"transform": "transform", "crs": "crs"})
        with pytest.raises(RasterValidationError, match="GeoBoxes"):
            mixed.chunk().gs.to_tensor(context={"centroid": "centroid"})
    with pytest.raises(RasterValidationError):
        xr.DataTree().gs.to_tensor()
    with pytest.raises(ValueError, match="EPSG code"):
        tree = xr.DataTree.from_dict({"image": build_raster(crs=CUSTOM_CRS)})
        tree.gs.to_tensor(context={"transform": "transform", "crs": "crs"})


def test_dimension_split_names_default_to_labels_and_duplicates_require_names():
    source = build_raster(times=2).chunk()
    leaf = source.gs.to_tree()["ds"]
    with dask.config.set(scheduler=_refuse):
        split = leaf.gs.split(dim="time")
    assert list(split["time"].children) == [str(value) for value in source.time.values]
    duplicated = source.assign_coords(time=source.time.isel(time=[0, 0]).values)
    with pytest.raises(ValueError, match="one unique child name"):
        duplicated.gs.to_tree()["ds"].gs.split(dim="time")


@pytest.mark.parametrize("names", [["one"], ["a", "a"], ["a/b", "c"], ["", "c"]])
def test_dimension_split_rejects_invalid_names(names):
    with pytest.raises(ValueError):
        build_raster(times=2).gs.to_tree()["ds"].gs.split(dim="time", names=names)


@pytest.mark.parametrize(
    "change",
    ["dtype", "units", "flags", "packing", "encoding", "unsigned", "variables"],
)
def test_concat_refuses_representation_changes_even_when_attrs_can_drop(change):
    source = build_raster(times=2).chunk()
    split = source.gs.to_tree()["ds"].gs.split(dim="time", names=["a", "b"])
    other = split["time/b"].gs.to_dataset()
    if change == "dtype":
        other["red"] = other.red.astype("float32")
    elif change == "units":
        other.red.attrs["units"] = "m"
    elif change == "flags":
        other.red.attrs.update(flag_values=[1, 2], flag_meanings="land water")
    elif change == "packing":
        other.red.attrs["scale_factor"] = 0.1
    elif change == "unsigned":
        other.red.attrs["_Unsigned"] = "true"
    elif change == "encoding":
        other.red.encoding["dtype"] = "uint8"
    else:
        other = other.drop_vars("red")
    split["time/b"] = xr.DataTree(other)
    with dask.config.set(scheduler=_refuse), pytest.raises(ValueError):
        split["time"].gs.concat("time", combine_attrs="drop_conflicts")


@pytest.mark.parametrize("operation", ["crop", "crop_to_grid", "reproject"])
@pytest.mark.parametrize("subtree", [False, True])
def test_spatial_transforms_keep_leaf_grids_and_group_metadata(operation, subtree):
    source = build_raster(times=2).chunk()
    root = xr.Dataset(
        coords={"time": source.time.variable, "platform": "sentinel-2"},
        attrs={"license": "CC0"},
    )
    pixels = source.drop_vars("time")
    tree = xr.DataTree.from_dict(
        {"/": root, "s2": xr.Dataset(attrs={"title": "S2"}), "s2/image": pixels}
    )
    before = tree.copy(deep=False)
    target = source.gs.geobox[:1, :1]
    selected = tree["s2"] if subtree else tree
    with dask.config.set(scheduler=_refuse):
        if operation == "crop":
            result = selected.gs.crop(target.extent, mask=False)
        else:
            result = getattr(selected.gs, operation)(target)
    leaf = result["image"] if subtree else result["s2/image"]
    raster = leaf.gs.to_dataset()
    assert raster.gs.geobox == target
    assert raster.attrs == {"license": "CC0", "title": "S2"}
    assert leaf.attrs == {}
    assert "x" not in result.coords and "spatial_ref" not in result.coords
    assert "time" in result.coords
    assert raster.platform.item() == "sentinel-2"
    xr.testing.assert_identical(result.platform, tree.platform)
    np.testing.assert_array_equal(raster.red.values, source.red.values[..., :1, :1])
    xr.testing.assert_identical(tree, before)


@pytest.mark.parametrize("coordinates", [["x", "y"], ["spatial_ref"]])
def test_validation_requires_grid_coordinates_on_each_leaf(coordinates):
    source = build_raster().chunk()
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={key: source[key].variable for key in coordinates}),
            "image": source.drop_vars(coordinates),
        }
    )

    with dask.config.set(scheduler=_refuse), pytest.raises(RasterValidationError):
        tree.gs.validate()


def test_crop_to_grid_clips_shared_extent_at_its_owner():
    source = build_raster().chunk()
    source = source.gs.write_extent(source.gs.geobox.extent)
    shared = xr.Dataset(
        coords={"platform": "sentinel-2"}, attrs={**source.attrs, "license": "CC0"}
    )
    pixels = source.copy(deep=False)
    pixels.attrs = {}
    tree = xr.DataTree.from_dict({"/": shared, "image": pixels})
    before = tree.copy(deep=False)
    target = source.gs.geobox[:1, :1]
    cropped = tree.gs.crop_to_grid(target)
    assert (
        cropped["image"].gs.to_dataset().gs.region.extent
        == source.gs.crop_to_grid(target).gs.region.extent
    )
    assert cropped["image"].attrs == {}
    assert cropped.attrs["license"] == "CC0"
    assert cropped.attrs["extent"] == source.gs.crop_to_grid(target).attrs["extent"]
    xr.testing.assert_identical(cropped.platform, tree.platform)
    xr.testing.assert_identical(tree, before)


def test_restructuring_attrs_policy_cannot_drop_pixel_interpretation():
    source = build_raster(times=2, packed=True)
    source.red.attrs["units"] = "1"
    split = source.gs.to_tree()["ds"].gs.split(dim="time")
    joined = split["time"].gs.concat("time", combine_attrs="drop").gs.to_dataset()
    for key in source.data_vars:
        for field in ("scale_factor", "add_offset", "_FillValue", "units"):
            assert joined[key].attrs.get(field) == source[key].attrs.get(field)
    groups = source.gs.to_tree(groups={"red": ["red"], "nir": ["nir"]})
    merged = groups["ds"].gs.merge(combine_attrs="drop").gs.to_dataset()
    assert merged.red.attrs == source.red.attrs


def test_concat_policy_combines_dataset_attrs_and_keeps_first_variable_attrs():
    source = build_raster(times=2, packed=True)
    split = source.gs.to_tree()["ds"].gs.split(dim="time", names=["a", "b"])
    split["time/a"].attrs["title"] = "first"
    split["time/b"].attrs["title"] = "second"
    split["time/b/red"].attrs["long_name"] = "Red, second slice"

    joined = split["time"].gs.concat("time", combine_attrs="drop_conflicts")
    raster = joined.gs.to_dataset()

    assert "title" not in raster.attrs
    assert raster.red.attrs == source.red.attrs
    assert (
        raster.red.attrs["scale_factor"].dtype == source.red.attrs["scale_factor"].dtype
    )


def test_tree_chunk_heights_are_shared_by_its_raster_leaves():
    raster = build_raster()
    tree = xr.DataTree.from_dict({"/image": raster, "/label": raster[["red"]]})

    assert tree.gs.chunk_heights is None
    assert tree.chunk(y=1).gs.chunk_heights == (1, 1)

    mixed = xr.DataTree.from_dict(
        {"/image": raster.chunk(y=1), "/label": raster[["red"]].chunk(y=2)}
    )
    with pytest.raises(RasterValidationError, match="chunked 2 ways"):
        mixed.gs.chunk_heights


def test_tree_spatial_dims_are_those_of_the_shared_geobox():
    raster = build_raster()
    tree = xr.DataTree.from_dict({"/image": raster, "/label": raster[["red"]]})

    assert tree.gs.spatial_dims == ("y", "x")


def test_tree_extent_is_the_union_of_its_leaves():
    raster = build_raster()
    grid = raster.gs.geobox.extent
    corners = grid.exterior.coords
    lower = Geometry(Polygon(corners[:3]), grid.crs)
    upper = Geometry(Polygon([corners[0], corners[2], corners[3]]), grid.crs)
    tree = xr.DataTree.from_dict(
        {
            "image": raster.gs.write_extent(lower),
            "label": raster[["red"]].gs.write_extent(upper),
        }
    )

    assert tree.gs.extent.area == pytest.approx(grid.area)
    assert xr.DataTree.from_dict({"image": raster}).gs.extent is None
