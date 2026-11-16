"""Native grouping and selection preserve opened raster assets."""

import pytest
import xarray as xr
from dask.callbacks import Callback

from geosave.geodata import attrs
from geosave.geodata.stac import RasterStac
from tests.geodata.conftest import assign_cells, build_raster, create_test_collection


def test_grouped_assets_preserve_native_datasets_without_execution():
    scene = build_raster(times=2, packed=True).chunk({"time": 1})
    scene["height"] = scene.red.isel(time=0, drop=True)
    scene = scene.gs.write_stac_property("eo:cloud_cover", [10.0, 40.0])
    scene = scene.gs.write_stac_property("platform", "sentinel-2b")
    scene = assign_cells(
        scene,
        [
            ("2025-06-01", "2025-06-02"),
            ("2025-06-02", "2025-06-03"),
        ],
    )
    scene = scene.rename({"eo:cloud_cover": "cloud"})
    scene.attrs["title"] = "Source title"
    collection = create_test_collection("optical", description="Optical")
    scene = attrs.stamp_attrs(
        scene,
        attrs.StacCollection(
            collection=collection.to_dict(), extensions=["example:extension"]
        ),
    )
    scene.red.encoding["dtype"] = "uint16"
    before = scene.copy(deep=False)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        result = RasterStac.from_dataset(scene, id_format="{start:%Y%m%d}")
    assert tasks == []
    assert result.collection.id == "optical"
    assert list(result.tree.children) == ["20250601", "20250602"]
    for index, item in enumerate(result.tree.children.values()):
        assert not item.data_vars
        assert set(item.coords) == {
            "time",
            "start_datetime",
            "end_datetime",
            "cloud",
            "platform",
        }
        assert item["cloud"].attrs["stac_property"] == "eo:cloud_cover"
        assert item["platform"].item() == "sentinel-2b"
        expected_item_attrs = dict(scene.attrs)
        expected_item_attrs.pop("stac_collection")
        assert item.attrs == expected_item_attrs
        assert set(item.children) == {"red", "nir", "height"}
        for name, node in item.children.items():
            asset = node.to_dataset(inherit="all_coords")
            local = node.to_dataset(inherit=False)
            assert not (set(local.coords) & set(item.coords))
            assert set(local.coords) == {"x", "y", "spatial_ref"}
            assert local.attrs == {}
            expected = scene.isel(time=index)[[name]].copy(deep=False)
            expected.attrs = {}
            assert asset.gs.geobox == scene.gs.geobox
            assert asset[name].encoding == expected[name].encoding
            assert asset[name].chunks is not None
            xr.testing.assert_identical(asset.compute(), expected.compute())
    xr.testing.assert_identical(scene, before)


def test_missing_item_dimension_uses_native_groupby_error():
    with pytest.raises(KeyError):
        RasterStac.from_dataset(build_raster())


def test_custom_item_dimension():
    scene = build_raster(times=2).rename_dims(time="observation")
    result = RasterStac.from_dataset(
        scene, item_dim="observation", id_format="{start:%Y%m%d}"
    )
    assert len(result.tree.children) == 2


def test_duplicate_output_ids_do_not_overwrite_items():
    with pytest.raises(ValueError, match="Duplicate Item id"):
        RasterStac.from_dataset(build_raster(times=2), id_format="same")


def build_grid(shape, names, dtype):
    """Build two dated observations of `names` over one 40 m extent."""
    import numpy as np
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    geobox = GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_040.0, 9_000_040.0),
        crs="EPSG:32749",
        shape=shape,
        tight=True,
    )
    coords = dict(xr_coords(geobox, always_yx=True))
    coords["time"] = ("time", np.array(["2025-06-01", "2025-06-02"], "datetime64[ns]"))
    block = np.arange(2 * shape[0] * shape[1], dtype=dtype).reshape(2, *shape) + 1
    variables = {}
    for name in names:
        variables[name] = (("time", "y", "x"), block, {"_FillValue": 0})
    raster = xr.Dataset(variables, coords=coords).chunk({"time": 1})
    raster = raster.gs.write_stac_property("eo:cloud_cover", [24.001125, 40.0])
    raster = raster.gs.write_stac_property("platform", "sentinel-2b")
    collection = create_test_collection("optical", description="Optical")
    return attrs.stamp_attrs(
        raster, attrs.StacCollection(collection=collection.to_dict())
    )


def build_two_grids():
    """Stage `red`/`nir` at 10 m and `scl` at 20 m over one extent and two dates."""
    fine = build_grid((4, 4), ["red", "nir"], "uint16")
    for name in fine.data_vars:
        fine[name].attrs.update(scale_factor=0.0001, add_offset=0.0)
    coarse = build_grid((2, 2), ["scl"], "uint8")
    return fine, coarse


def test_merge_adds_assets_on_another_grid_to_shared_items():
    fine, coarse = build_two_grids()
    staged = RasterStac.from_dataset(fine)

    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        merged = staged.merge(RasterStac.from_dataset(coarse))

    assert tasks == []
    assert list(merged.tree.children) == list(staged.tree.children)
    assert merged.collection.id == "optical"
    for item in merged.tree.children.values():
        assert list(item.children) == ["red", "nir", "scl"]
        assert dict(item["red"].sizes) == {"y": 4, "x": 4}
        assert dict(item["scl"].sizes) == {"y": 2, "x": 2}
    # The inputs keep their own assets.
    assert list(next(iter(staged.tree.children.values())).children) == ["red", "nir"]


def test_merge_adds_items_only_one_side_states():
    fine, _ = build_two_grids()
    first = RasterStac.from_dataset(fine.isel(time=[0]), id_format="{start:%Y%m%d}")
    second = RasterStac.from_dataset(fine.isel(time=[1]), id_format="{start:%Y%m%d}")

    assert list(first.merge(second).tree.children) == ["20250601", "20250602"]


def test_merge_refuses_conflicts():
    fine, coarse = build_two_grids()
    staged = RasterStac.from_dataset(fine)

    with pytest.raises(ValueError, match="Asset .*red.* is in both"):
        staged.merge(RasterStac.from_dataset(fine))

    relabelled = coarse.gs.write_stac_property("platform", "sentinel-2c")
    with pytest.raises(ValueError, match="Item .* differs"):
        staged.merge(RasterStac.from_dataset(relabelled))

    other = create_test_collection("radar", description="Radar")
    foreign = attrs.stamp_attrs(
        coarse, attrs.StacCollection(collection=other.to_dict())
    )
    with pytest.raises(ValueError, match="Collections differ"):
        staged.merge(RasterStac.from_dataset(foreign))


def test_to_cog_items_load_back_per_native_grid(tmp_path):
    from geosave.geodata import stac

    fine, coarse = build_two_grids()
    staged = RasterStac.from_dataset(fine, id_format="{start:%Y%m%d}").merge(
        RasterStac.from_dataset(coarse, id_format="{start:%Y%m%d}")
    )

    items = staged.to_cog(tmp_path)

    assert [item.id for item in items] == ["20250601", "20250602"]
    item = items[0]
    assert item.assets["red"].href == str(tmp_path / "20250601" / "red.tif")
    assert item.assets["red"].extra_fields["proj:shape"] == [4, 4]
    assert item.assets["scl"].extra_fields["proj:shape"] == [2, 2]
    assert item.properties["eo:cloud_cover"] == 24.001125
    assert item.properties["platform"] == "sentinel-2b"

    loaded = stac.load_native_datasets(
        items, chunks={}, with_properties=stac.list_properties(items)
    )
    assert [list(native.data_vars) for _, _, native in loaded] == [
        ["red", "nir"],
        ["scl"],
    ]
    sources = [fine, coarse]
    for (_, _, native), source in zip(loaded, sources, strict=True):
        assert native.gs.geobox == source.gs.geobox
        assert native["eo_cloud_cover"].values.tolist() == [24.001125, 40.0]
        assert native["platform"].values.tolist() == ["sentinel-2b", "sentinel-2b"]
        for name in native.data_vars:
            assert native[name].dtype == source[name].dtype
            assert (native[name].values == source[name].values).all()
    for name in ("red", "nir"):
        assert loaded[0][2][name].attrs["scale_factor"] == 0.0001


def test_to_cog_links_items_to_the_staged_collection(tmp_path):
    fine, _ = build_two_grids()
    staged = RasterStac.from_dataset(fine, id_format="{start:%Y%m%d}")

    items = staged.to_cog(tmp_path)

    for item in items:
        assert item.collection_id == "optical"
        assert item.get_collection() is staged.collection
    written = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.*"))
    assert written == [
        "20250601/nir.tif",
        "20250601/red.tif",
        "20250602/nir.tif",
        "20250602/red.tif",
    ]


def test_to_cog_without_a_collection_leaves_items_unlinked(tmp_path):
    items = RasterStac.from_dataset(
        build_raster(times=1), id_format="{start:%Y%m%d}"
    ).to_cog(tmp_path)

    assert items[0].collection_id is None
    assert items[0].links == []


def test_a_given_collection_replaces_the_one_the_dataset_states(tmp_path):
    source = create_test_collection("optical", description="Optical")
    scene = attrs.stamp_attrs(
        build_raster(times=1), attrs.StacCollection(collection=source.to_dict())
    )
    derived = create_test_collection("water", description="Water maps")

    staged = RasterStac.from_dataset(scene, collection=derived)

    assert staged.collection.id == "water"
    assert RasterStac.from_dataset(scene).collection.id == "optical"


def test_to_cog_refuses_existing_files_unless_overwriting(tmp_path):
    staged = RasterStac.from_dataset(build_raster(times=1), id_format="{start:%Y%m%d}")
    staged.to_cog(tmp_path)

    with pytest.raises(FileExistsError):
        staged.to_cog(tmp_path)
    assert len(staged.to_cog(tmp_path, overwrite=True)) == 1
