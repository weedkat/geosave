"""Bounded feasibility probes; does not implement proposed GeoSave APIs."""

from datetime import UTC, datetime
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import mkdtemp

from affine import Affine
from dask.callbacks import Callback
import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
import xarray as xr

import geosave.geodata as gs
from geosave.geodata import stac
from geosave.geodata.core.raster.rules.datatree import same_geobox, same_grid
from geosave.geodata.io.raster.geotiff import write_files
from geosave.geodata.io.storage import create_upath
from geosave.geodata.sampling import RasterChipper
from geosave.geodata.stac.extensions import classification
from pystac.extensions.raster import RasterExtension


ROOT = Path(mkdtemp(prefix="geosave-tree-contract-"))
RESULTS = {"root": str(ROOT)}
GRID = GeoBox((4, 4), Affine(10, 0, 500000, 0, -10, 9000040), "EPSG:32749")
COARSE = GeoBox((2, 2), Affine(20, 0, 500000, 0, -20, 9000040), "EPSG:32749")
DATES = np.array(["2025-06-01", "2025-06-02"], dtype="datetime64[ns]")


def raster(names, grid=GRID, dtype="uint16", times=None):
    coords = dict(xr_coords(grid, always_yx=True))
    shape = tuple(grid.shape)
    dims = ("y", "x")
    if times is not None:
        coords["time"] = ("time", times)
        shape = (len(times), *shape)
        dims = ("time", *dims)

    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape) + 1
    variables = {name: (dims, data.copy(), {"_FillValue": 0}) for name in names}
    return xr.Dataset(variables, coords=coords).chunk()


def refused(action):
    try:
        action()
    except Exception as error:
        return {"type": type(error).__name__, "message": str(error).splitlines()[0]}
    raise AssertionError("Expected rejection did not occur")


def structure_probe():
    image = raster(["red", "nir"], times=DATES)
    label = raster(["class"], dtype="uint8", times=DATES[1:])
    height = raster(["height"], dtype="float32")
    image.attrs["title"] = "Reflectance"
    label.attrs["title"] = "Reference classes"
    tasks = []

    with Callback(pretask=lambda *args: tasks.append(args[0])):
        root = xr.Dataset(
            coords=xr_coords(GRID, always_yx=True), attrs={"license": "CC0"}
        )
        tree = xr.DataTree.from_dict(
            {"/": root, "image": image, "label": label, "static": height}
        )
        tree.gs.validate(same_geobox)
        extracted = tree["image"].gs.to_dataset()

    assert not tasks
    assert extracted.red.data is image.red.data
    assert extracted.attrs == {"license": "CC0", "title": "Reflectance"}
    assert tree["label"].sizes["time"] == 1
    assert "time" not in tree["static"].to_dataset().dims

    exact = refused(
        lambda: xr.merge([image, label], join="exact", compat="no_conflicts")
    )
    outer = xr.merge([image, label], join="outer", compat="no_conflicts")
    duplicate = refused(
        lambda: xr.merge([height, height + 1], join="exact", compat="no_conflicts")
    )
    inconsistent = refused(
        lambda: xr.DataTree.from_dict({"/": root, "coarse": raster(["scl"], COARSE)})
    )

    RESULTS["structure"] = {
        "tasks": len(tasks),
        "shares_pixels": True,
        "label_times_before": 1,
        "label_times_after_outer_merge": outer["class"].sizes["time"],
        "label_dtype_before": str(label["class"].dtype),
        "label_dtype_after_outer_merge": str(outer["class"].dtype),
        "exact_merge": exact,
        "duplicate_name_merge": duplicate,
        "root_coordinates_reject_other_grid": inconsistent,
    }
    return tree


def general_tree_probe():
    image = raster(["red", "nir"])
    scl = raster(["scl"], COARSE, "uint8")
    tasks = []

    with Callback(pretask=lambda *args: tasks.append(args[0])):
        tree = xr.DataTree.from_dict(
            {
                "/": xr.Dataset(attrs={"license": "CC0"}),
                "scene": xr.Dataset(attrs={"platform": "study"}),
                "scene/image": image,
                "scene/scl": scl,
            }
        )
        tree.gs.validate()
        extracted = tree["scene/scl"].gs.to_dataset()
        cropped = tree.gs.crop(GRID[:2, :2].extent)
        aligned = tree.gs.reproject(GRID)
        aligned.gs.validate(same_geobox)

    assert not tasks
    assert extracted.scl.data is scl.scl.data
    assert extracted.attrs == {"license": "CC0", "platform": "study"}
    assert tree["scene/scl"].gs.to_dataset().gs.geobox == COARSE

    pixels = {}
    for node in tree.leaves:
        if node.data_vars:
            pixels[node.path], context = node.gs.to_dataset().gs.to_tensor()
            assert context == {}

    assert pixels["/scene/image"].shape == (2, 4, 4)
    assert pixels["/scene/scl"].shape == (1, 2, 2)
    expected = np.repeat(np.repeat(scl.scl.values, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(
        aligned["scene/scl"].gs.to_dataset().scl.values, expected
    )
    assert dict(cropped["scene/image"].sizes) == {"y": 2, "x": 2}
    assert dict(cropped["scene/scl"].sizes) == {"y": 1, "x": 1}

    RESULTS["general_tree"] = {
        "planning_tasks": len(tasks),
        "effective_attrs": extracted.attrs,
        "original_grid_preserved": True,
        "leaf_tensor_shapes": {
            name: list(value.shape) for name, value in pixels.items()
        },
        "crop_shapes": {node.path: dict(node.sizes) for node in cropped.leaves},
        "explicit_alignment": "matches expected nearest-neighbour pixels",
    }


def processing_probe(tree):
    parent_coordinate_failure = refused(
        lambda: RasterChipper([tree], size=2, mode="drop")[0]
    )
    # Continue with equivalent leaf-local coordinates, the current design default.
    local_nodes = {"/": xr.Dataset(attrs=tree.attrs)}
    local_nodes.update(
        {name: node.gs.to_dataset() for name, node in tree.children.items()}
    )
    tree = xr.DataTree.from_dict(local_nodes)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        chipper = RasterChipper([tree], size=2, stride=2, mode="drop")
        chip = chipper[0]
    assert not tasks

    pixels, context = chip.gs.to_tensor(grid=True)
    assert set(context) == {"transform", "crs"}
    temporal = refused(lambda: chip.gs.to_tensor(context=["time"]))
    sample = refused(lambda: tree.gs.sample("D"))

    shifted = GeoBox((4, 4), Affine(10, 0, 500020, 0, -10, 9000040), "EPSG:32749")
    adjacent = xr.DataTree.from_dict({"a": raster(["a"]), "b": raster(["b"], shifted)})
    adjacent.gs.validate(same_grid)
    different_extent = refused(lambda: adjacent.gs.validate(same_geobox))
    RESULTS["processing"] = {
        "current_parent_coordinate_crop_failure": parent_coordinate_failure,
        "chipping_tasks": len(tasks),
        "tensor_shapes": {name: list(value.shape) for name, value in pixels.items()},
        "shared_grid_context": list(context),
        "shared_time_context": temporal,
        "temporal_sampling": sample,
        "same_lattice_different_extent": different_extent,
    }


def persistence_probe(tree):
    path = ROOT / "analysis.zarr"
    tree.gs.to_zarr(path)
    with gs.open_raster_tree(path) as restored:
        restored.gs.validate(same_geobox)
        assert tuple(restored.children) == tuple(tree.children)
        assert restored.attrs["license"] == "CC0"
        for name in tree.children:
            original = tree[name].gs.to_dataset()
            actual = restored[name].gs.to_dataset()
            assert actual.attrs == original.attrs
            assert actual.sizes == original.sizes
            for variable in original.data_vars:
                np.testing.assert_array_equal(
                    actual[variable].values, original[variable].values
                )
    RESULTS["zarr"] = {"groups_attrs_dimensions_pixels": "preserved"}


def publication_inputs():
    fine = raster(["red", "nir"], times=DATES)
    fine.attrs["title"] = "Analysis group title"
    for name in fine.data_vars:
        fine[name].attrs.update(units="1", scale_factor=0.0001, add_offset=0.0)
    coarse = raster(["scl"], COARSE, "uint8", DATES)
    coarse.scl.attrs.update(
        flag_values=[1, 2, 3, 4], flag_meanings="one two three four"
    )
    # Restrict codes to the declared four classes for both dates.
    coarse["scl"] = (coarse.scl - 1) % 4 + 1
    coarse.scl.attrs = {
        "_FillValue": 0,
        "flag_values": [1, 2, 3, 4],
        "flag_meanings": "one two three four",
    }
    static = raster(["height"], dtype="float32")

    assets_by_item = {}
    files = {create_upath(ROOT / "height.tif"): static}
    for index in range(2):
        item_id = f"scene-{index}"
        assets = {
            "image": fine.isel(time=index),
            "scl": coarse.isel(time=index),
            "height": static,
        }
        assets_by_item[item_id] = assets
        for key in ("image", "scl"):
            files[create_upath(ROOT / f"{item_id}-{key}.tif")] = assets[key]

    return fine, coarse, static, assets_by_item, files


def describe_items(assets_by_item):
    items = []
    for index, (item_id, assets) in enumerate(assets_by_item.items()):
        instant = datetime(2025, 6, index + 1, tzinfo=UTC)
        region = gs.GeoRegion(GRID.extent, timespan=(instant, instant))
        item = stac.StacItem.from_region(
            region, id=item_id, properties={"platform": "study"}
        )
        for key, dataset in assets.items():
            path = ROOT / ("height.tif" if key == "height" else f"{item_id}-{key}.tif")
            asset = item.add_raster_asset(key, dataset, str(path), with_eo=True)
            bands = RasterExtension.ext(asset).bands
            for band, variable in zip(bands, dataset.data_vars.values(), strict=True):
                classification.write_band(asset, band, variable.gs.flags)
            RasterExtension.ext(asset).bands = bands
        items.append(item)

    return items


def publish_probe():
    fine, coarse, static, assets_by_item, files = publication_inputs()
    written = write_files(files)
    assert len(written) == 5

    for path, source in files.items():
        with gs.open_raster(path, mask_and_scale=False) as actual:
            assert actual.gs.geobox == source.gs.geobox
            for name in source.data_vars:
                assert actual[name].dtype == source[name].dtype
                np.testing.assert_array_equal(actual[name].values, source[name].values)

    items = describe_items(assets_by_item)

    rows = stac.create_catalog(items)
    catalog_path = ROOT / "catalog.parquet"
    rows.gs.to_geoparquet(catalog_path)
    reopened = gs.open_vector(catalog_path)
    restored_items = reopened.gs.to_items()
    assert [item.id for item in restored_items] == ["scene-0", "scene-1"]
    assert (
        restored_items[0].assets["height"].href
        == restored_items[1].assets["height"].href
    )

    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        loaded = reopened.gs.to_dataset(
            geobox=GRID, chunks={}, groupby="id", resampling="nearest"
        )
        reconstructed = xr.DataTree.from_dict(
            {"image": loaded[["red", "nir"]], "scl": loaded[["scl"]]}
        )
        reconstructed.gs.validate(same_geobox)
    assert not tasks
    assert loaded.gs.geobox == GRID
    for name in ("red", "nir"):
        np.testing.assert_array_equal(loaded[name].values, fine[name].values)
    expected_scl = np.repeat(np.repeat(coarse.scl.values, 2, axis=1), 2, axis=2)
    np.testing.assert_array_equal(loaded.scl.values, expected_scl)
    assert loaded.red.attrs["scale_factor"] == 0.0001
    assert loaded.scl.attrs["flag_meanings"] == "one two three four"
    assert loaded.height.dims == ("time", "y", "x")
    assert "title" not in loaded.attrs

    native = stac.load_native_datasets(restored_items, chunks={}, groupby="id")
    assert len(native) == 2
    assert native[0][2].gs.geobox == GRID
    assert native[1][2].gs.geobox == COARSE
    np.testing.assert_array_equal(native[1][2].scl.values, coarse.scl.values)
    RESULTS["publication"] = {
        "items": len(items),
        "files": len(written),
        "shared_static_href": True,
        "planning_tasks": len(tasks),
        "loaded_variables": list(loaded.data_vars),
        "static_source_dims": list(static.height.dims),
        "static_loaded_dims": list(loaded.height.dims),
        "source_group_title_recovered": "title" in loaded.attrs,
        "native_shapes": [list(ds.gs.geobox.shape) for _, _, ds in native],
        "source_values_packing_flags_and_requested_grid": "checked",
        "hierarchy": "explicit caller grouping; not recovered from STAC",
        "metadata_wiring": "with_eo=True plus existing classification.write_band",
        "direct_asset_grid_dtype_values": "preserved for all five files",
    }
    return items


def staging_layout_probe():
    image = raster(["red", "nir"], times=DATES)
    height = raster(["height"], COARSE, "float32")
    tasks = []

    with Callback(pretask=lambda *args: tasks.append(args[0])):
        tree = xr.DataTree.from_dict(
            {
                "/": xr.Dataset(attrs={"license": "CC0"}),
                "scene": xr.Dataset(attrs={"title": "Source Dataset"}),
                "scene/static": height,
                "scene/time/2025-06-01": image.isel(time=0),
                "scene/time/2025-06-02": image.isel(time=1),
            }
        )
        tree.gs.validate()
        static = tree["scene/static"].gs.to_dataset()
        items = []
        for index, label in enumerate(("2025-06-01", "2025-06-02")):
            snapshot = tree[f"scene/time/{label}"].gs.to_dataset()
            assert snapshot.attrs == {"license": "CC0", "title": "Source Dataset"}
            np.testing.assert_array_equal(snapshot.time.values, DATES[index])
            instant = datetime(2025, 6, index + 1, tzinfo=UTC)
            region = gs.GeoRegion(GRID.extent, timespan=(instant, instant))
            item = stac.StacItem.from_region(region, id=f"explicit-{index}")
            item.add_raster_asset("image", snapshot, str(ROOT / f"image-{index}.tif"))
            item.add_raster_asset("elevation", static, str(ROOT / "shared-height.tif"))
            items.append(item)

    assert not tasks
    assert static.height.dims == ("y", "x")
    assert static.gs.geobox == COARSE
    assert "time" not in static.coords
    assert [item.id for item in items] == ["explicit-0", "explicit-1"]
    assert items[0].assets["elevation"].href == items[1].assets["elevation"].href
    assert all("title" not in item.properties for item in items)
    assert items[0].assets["image"].extra_fields["proj:shape"] == [4, 4]
    assert items[0].assets["elevation"].extra_fields["proj:shape"] == [2, 2]

    RESULTS["staging_layout"] = {
        "tasks": len(tasks),
        "item_ids": [item.id for item in items],
        "assets": list(items[0].assets),
        "static_dims": list(static.height.dims),
        "shared_static_href": True,
        "inherited_source_attrs": snapshot.attrs,
        "mode": "metadata staging only; no files written by this probe",
    }


def native_grid_probe():
    # Same CRS/resolution does not identify a source pixel lattice.
    shifted = GeoBox((4, 4), Affine(10, 0, 500005, 0, -10, 9000040), "EPSG:32749")
    datasets = [raster(["value"]), raster(["value"], shifted)]
    files = {create_upath(ROOT / f"shift-{i}.tif"): ds for i, ds in enumerate(datasets)}
    write_files(files)
    items = []
    for i, (path, ds) in enumerate(files.items()):
        instant = datetime(2025, 7, i + 1, tzinfo=UTC)
        region = gs.GeoRegion(ds.gs.geobox.extent, timespan=(instant, instant))
        item = stac.StacItem.from_region(region, id=f"shift-{i}")
        item.add_raster_asset("value", ds, str(path), with_eo=True)
        items.append(item)
    groups = stac.load_native_datasets(items, chunks={}, groupby="id")
    assert len(groups) == 1
    output = groups[0][2]
    assert output.gs.geobox != shifted
    output.compute()
    RESULTS["native_grid_limit"] = {
        "input_grids": 2,
        "output_groups": len(groups),
        "output_shape": list(output.gs.geobox.shape),
        "source_transform_x": [ds.gs.geobox.transform.c for ds in datasets],
        "output_transform_x": output.gs.geobox.transform.c,
        "preserves_both_source_grids": False,
    }


if __name__ == "__main__":
    RESULTS["versions"] = {
        name: version(name)
        for name in ("xarray", "odc-geo", "odc-stac", "dask", "pystac")
    }
    tree = structure_probe()
    print("PASS structure", flush=True)
    for name, action in [
        ("general_tree", general_tree_probe),
        ("processing", lambda: processing_probe(tree)),
        ("zarr", lambda: persistence_probe(tree)),
        ("publication", publish_probe),
        ("staging_layout", staging_layout_probe),
        ("native_grid_limit", native_grid_probe),
    ]:
        try:
            action()
            print(f"PASS {name}", flush=True)
        except Exception as error:
            RESULTS[name + "_failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            (ROOT / "results.json").write_text(json.dumps(RESULTS, indent=2))
            print(f"Results: {ROOT / 'results.json'}", flush=True)
            raise
    (ROOT / "results.json").write_text(json.dumps(RESULTS, indent=2))
    print(json.dumps(RESULTS, indent=2))
