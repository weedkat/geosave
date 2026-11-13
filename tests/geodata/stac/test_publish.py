"""Publication splits rasters into Items and assets; asset reading keeps file pixels."""

from __future__ import annotations

from tests.conftest import write_cogs

from tests.conftest import combine_item


from datetime import UTC, datetime

import dask
import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin

from geosave.geodata.stac.load import load_datatree
from tests.conftest import build_dataset
from tests.geodata.conftest import (
    build_raster,
    create_file_items,
    item_table,
)


@pytest.fixture
def item(tmp_path):
    path = tmp_path / "image.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32749",
        transform=from_origin(500_000, 9_000_020, 10, 10),
        nodata=0,
    ) as tiff:
        tiff.write(np.stack([np.full((2, 2), value, "uint8") for value in (1, 2, 3)]))

    entry = next(
        create_file_items(
            {"visual": path},
            id_format="scene",
            datetime=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )
    # This fixture deliberately publishes no band identities.
    entry.assets["visual"].extra_fields.pop("eo:bands", None)
    return entry


_REBUILDING = pytest.mark.skip(
    reason="STAC publication is being rebuilt on stac_property coordinates"
)


def test_unlisted_bands_use_the_asset_key_without_reading_pixels(item):
    original = item.to_dict()

    raster = combine_item(load_datatree([item]))

    assert list(raster.data_vars) == ["visual.1", "visual.2", "visual.3"]
    assert dask.is_dask_collection(raster["visual.3"])
    assert raster["visual.3"].gs.nodata == 0
    assert int(raster["visual.3"].max()) == 3
    assert item.to_dict() == original


def test_native_composite_load_retains_bounds(tmp_path):

    # Scenes on June 1 and 2, so the composite spans both.
    ds = build_raster(times=2).gs.mask_and_scale().gs.composite("MS", "median")
    original = next(
        create_file_items(write_cogs(ds, tmp_path, format_paths="original.tif"))
    )
    loaded = combine_item(load_datatree([original]))
    rebuilt = next(
        create_file_items(write_cogs(loaded, tmp_path, format_paths="rebuilt.tif"))
    )
    assert original.datetime is None
    assert rebuilt.datetime == original.datetime
    assert rebuilt.properties["start_datetime"] == original.properties["start_datetime"]
    assert rebuilt.properties["end_datetime"] == original.properties["end_datetime"]


@pytest.fixture
def catalog(tmp_path):
    """A dated optical collection and a timeless DEM on one grid, catalogued."""
    from tests.geodata.conftest import write_collection_cogs

    files = write_collection_cogs(
        {"optical": build_raster(times=2), "dem": build_raster()[["red"]]}, tmp_path
    )
    return item_table(files, prefix="c")


def write_mixed_scene(tmp_path, crs_by_name=None):
    """Write a red band at 10 m and a swir band at 20 m as one Item's assets."""
    from odc.geo.geobox import GeoBox

    crs_by_name = crs_by_name or {}
    paths = {}
    for name, resolution in [("red", 10), ("swir", 20)]:
        crs = crs_by_name.get(name, "EPSG:32749")
        grid = GeoBox.from_bbox(
            (500000, 9000000, 500640, 9000640), crs=crs, resolution=resolution
        )
        raster = build_dataset(
            {name: (("y", "x"), np.full(grid.shape, resolution, "uint16"))},
            grid,
            time=np.datetime64("2024-03-05"),
        )
        paths[name] = write_cogs(raster, tmp_path, format_paths=f"{name}.tif")[0]
    return next(iter(create_file_items(paths, id_format="mixed", collection="s2")))


@_REBUILDING
def test_a_mixed_resolution_row_opens_on_its_native_grids(tmp_path):
    entry = write_mixed_scene(tmp_path)

    scene = load_datatree([entry])

    item = scene[f"s2/{entry.id}"]
    assert [leaf.name for leaf in item.leaves] == ["red", "swir"]
    assert item["red"].to_dataset().odc.geobox.shape == (64, 64)
    assert item["swir"].to_dataset().odc.geobox.shape == (32, 32)
    assert int(item["swir"].swir.max()) == 20


def write_png_item(tmp_path, stated_shape=(48, 64)):
    """Write a PNG, which holds no grid, and an Item whose asset states one."""
    from odc.geo.geobox import GeoBox

    from geosave.geodata.stac.extensions import projection

    path = tmp_path / "label.png"
    with rasterio.open(
        path, "w", driver="PNG", width=64, height=48, count=1, dtype="uint8"
    ) as png:
        png.write(np.arange(64 * 48, dtype="uint8").reshape(1, 48, 64))
    entry = pystac.Item("label", None, None, datetime(2024, 3, 5, tzinfo=UTC), {})
    entry.add_asset("label", pystac.Asset(str(path), roles=["data"]))
    rows, columns = stated_shape
    grid = GeoBox.from_bbox(
        (500000, 9000000, 500000 + columns * 10, 9000000 + rows * 10),
        crs="EPSG:32749",
        resolution=10,
    )
    projection.write_asset(entry.assets["label"], grid)
    return entry, grid


def build_item_node():
    """The first Item node of an expanded two-date scene."""
    tree = build_raster(times=2, packed=True).gs.expand()
    return next(iter(tree["default"].children.values()))


@_REBUILDING
def test_an_item_node_becomes_an_item_with_its_time_and_footprint():
    from geosave.geodata.stac.publish import StacItem

    node = build_item_node()

    item = StacItem.from_datatree(node, collection="default")

    assert item.id == node.name
    assert item.collection_id == "default"
    assert item.datetime is not None
    assert item.datetime.isoformat() == "2025-06-01T00:00:00+00:00"
    assert item.geometry is not None
    assert len(item.geometry["coordinates"][0]) == 17
    assert item.assets == {}


@_REBUILDING
def test_an_item_takes_the_id_properties_and_geometry_it_is_given():
    from shapely.geometry import box, mapping

    from geosave.geodata.stac.publish import StacItem

    footprint = mapping(box(111, -9.1, 111.1, -9))

    item = StacItem.from_datatree(
        build_item_node(),
        id="scene",
        properties={"platform": "s2b"},
        geometry=footprint,
    )

    assert item.id == "scene"
    assert item.properties["platform"] == "s2b"
    assert item.bbox == [111, -9.1, 111.1, -9]


@_REBUILDING
def test_a_raster_asset_declares_the_extensions_it_states():
    from geosave.geodata.stac.publish import StacItem

    node = build_item_node()
    item = StacItem.from_datatree(node)
    red = node["red"].to_dataset(inherit="all_coords")

    asset = item.add_raster_asset("red", red, "/data/red.tif")
    bare = item.add_raster_asset(
        "plain",
        red,
        "/data/plain.tif",
        roles=["overview"],
        with_proj=False,
        with_raster=False,
    )

    assert asset.extra_fields["proj:code"] == "EPSG:32749"
    assert asset.extra_fields["raster:bands"][0]["scale"] == 0.0001
    assert asset.roles == ["data"]
    assert "proj:code" not in bare.extra_fields
    assert bare.roles == ["overview"]
    assert any("projection" in uri for uri in item.stac_extensions)
    assert any("raster" in uri for uri in item.stac_extensions)


@_REBUILDING
def test_a_cloned_item_stays_a_stac_item():
    from geosave.geodata.stac.publish import StacItem

    item = StacItem.from_datatree(build_item_node())

    assert type(item.clone()) is StacItem
    assert type(StacItem.from_dict(item.to_dict())) is StacItem


@_REBUILDING
def test_a_published_scene_loads_back_from_its_stored_table(tmp_path):
    from geosave import open_vector
    from tests.geodata.conftest import assign_cells

    scene = build_raster(times=2, packed=True)
    days = []
    for value in scene.time.values:
        days.append((value, value + np.timedelta64(1, "D")))
    scene = assign_cells(scene, days)
    scene["red"].attrs.update(long_name="Red", units="1")

    rows = scene.gs.expand().gs.to_catalog(tmp_path / "out")
    stored = open_vector(rows.gs.to_geoparquet(tmp_path / "catalog.parquet"))
    restored = stored.gs.to_datatree()["default"].gs.combine()

    # Hrefs are stored relative to the table and resolve against it.
    assert stored.iloc[0]["assets"]["red"]["href"].startswith("./out/")
    assert stored.attrs["source"] == str(tmp_path / "catalog.parquet")
    for name in ("red", "nir"):
        np.testing.assert_array_equal(restored[name].values, scene[name].values)
    for name in ("time", "start_datetime", "end_datetime"):
        assert restored[name].equals(scene[name])
    assert restored.red.attrs["long_name"] == "Red"
    assert restored.red.attrs["units"] == "1"
    assert (
        np.float32(restored.red.attrs["scale_factor"])
        == scene.red.attrs["scale_factor"]
    )


@_REBUILDING
def test_tables_read_from_different_files_do_not_concatenate(tmp_path):
    from geosave import open_vector
    from geosave.geodata.stac.catalog import concat_catalogs

    tree = build_raster(times=1).gs.expand()
    first = open_vector(
        tree.gs.to_catalog(tmp_path / "a").gs.to_geoparquet(tmp_path / "a.parquet")
    )
    second = open_vector(
        tree.gs.to_catalog(tmp_path / "b").gs.to_geoparquet(tmp_path / "b.parquet")
    )

    assert concat_catalogs([first, first]).attrs["source"] == first.attrs["source"]
    with pytest.raises(ValueError, match="different files"):
        concat_catalogs([first, second])
