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

from geosave.geodata import (
    open_vector,
)
from geosave.geodata.stac.load import load_datatree
from geosave.geodata.stac.publish import create_stac_item
from geosave.geodata.stac.catalog import create_catalog
from tests.conftest import build_dataset
from tests.geodata.conftest import (
    _refuse,
    build_raster,
    create_file_items,
    create_test_collection,
    item_table,
    read_items,
)
from tests.geodata.stac.conftest import create_native_item


def test_a_catalog_loads_one_node_per_asset_whatever_the_asset_order(tmp_path):
    from affine import Affine
    from odc.geo.geobox import GeoBox

    base = Affine(10, 0, 500000, 0, -10, 9000020)
    grids = {
        "red": GeoBox((4, 4), base, "EPSG:32749"),
        "nir": GeoBox((4, 4), base, "EPSG:32749"),
        "swir": GeoBox((2, 2), base * Affine.scale(2), "EPSG:32749"),
    }
    sources = {}
    for value, (name, grid) in enumerate(grids.items(), start=1):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=1,
            dtype="uint16",
            crs=str(grid.crs),
            transform=grid.transform,
        ) as file:
            file.write(np.full(tuple(grid.shape), value, dtype="uint16"), 1)
        sources[name] = path
    entry = create_stac_item(
        sources,
        id="first",
        input_datetime=datetime(2025, 1, 1, tzinfo=UTC),
        with_proj=True,
        with_raster=True,
        with_eo=True,
    )
    entry.set_collection(create_test_collection("s2", description="S2"))
    second = entry.clone()
    second.id = "second"
    second.datetime = datetime(2025, 1, 2, tzinfo=UTC)
    second.assets = dict(reversed(list(second.assets.items())))

    tree = create_catalog([entry, second]).gs.to_datatree()
    # Each Item lists its assets as its own leaves, whatever their order.
    assert tree["s2/first"].isomorphic(tree["s2/second"])
    for item in ("first", "second"):
        for name, grid in grids.items():
            leaf = tree[f"s2/{item}/{name}"]
            assert leaf.gs.geobox == grid
            np.testing.assert_array_equal(
                leaf[name].values, np.full(grid.shape, list(grids).index(name) + 1)
            )


def test_native_paths_publish_actual_band_metadata(tmp_path):
    from tests.geodata.conftest import build_raster

    path = tmp_path / "image.tif"
    source = build_raster(times=1, packed=True)
    write_cogs(source, path.parent, format_paths=path.name)
    items = list(create_file_items([path], collection="s2"))
    assert len(items) == 1
    assert items[0].collection_id == "s2"
    assert set(items[0].assets) == {"image"}
    with create_catalog(items).iloc[0].gs.to_datatree() as tree:
        restored = combine_item(tree)
        assert restored.red.dtype == source.red.dtype
        assert (
            restored.red.gs.packing.scale_factor == source.red.gs.packing.scale_factor
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

    return create_stac_item(
        {"visual": path},
        id="scene",
        input_datetime=datetime(2025, 1, 1, tzinfo=UTC),
        with_proj=True,
        with_raster=True,
        with_eo=True,
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


@pytest.mark.integration
def test_an_item_table_of_cogs_loads_back(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)
    paths = write_cogs(
        source, prefix, format_paths="forest_{time:%Y%m%d}.tif", storage_options=options
    )
    # GDAL reads a remote TIFF with the keys in its environment, not fsspec options.
    rows = item_table(paths)

    saved = rows.gs.to_geoparquet(f"{prefix}/catalog.parquet", storage_options=options)

    rows = open_vector(saved, storage_options=options)
    assert rows.iloc[0]["assets"]["image"]["href"] == str(paths[0])
    with dask.config.set(scheduler=_refuse):
        restored = read_items(rows)
    np.testing.assert_array_equal(restored.red.values, source.red.values)
    restored.close()


def test_only_data_assets_load(tmp_path):
    entry, source = create_native_item(tmp_path)

    loaded = read_items(create_catalog([entry]))

    # The thumbnail is skipped; the data asset's bands keep the names it states.
    assert list(loaded.data_vars) == list(source.data_vars)
    loaded.close()


@pytest.fixture
def catalog(tmp_path):
    """A dated optical collection and a timeless DEM on one grid, catalogued."""
    from tests.conftest import build_tree
    from tests.geodata.conftest import write_collection_cogs

    tree = build_tree(
        {"optical": build_raster(times=2), "dem": build_raster()[["red"]]}
    )
    files = write_collection_cogs(tree, tmp_path)
    return item_table(files, prefix="c")


def test_collections_open_as_groups_in_first_appearance_order(catalog):
    tree = catalog.gs.to_datatree()

    assert tree.gs.groups == tuple(dict.fromkeys(catalog["collection"]))
    assert [leaf.name for leaf in tree.leaves] == ["image", "image", "red"]
    assert tree["optical"].gs.combine_stac().sizes["time"] == 2
    # The DEM Item is dated, so it loads as one time step, as stac.load_dataset does.
    assert tree["dem"].gs.combine_stac().sizes["time"] == 1
    assert tree["optical"].gs.combine_stac().red.chunks is not None


def test_an_item_without_a_collection_sits_under_default(catalog):
    rows = catalog.copy()
    rows["collection"] = None

    assert rows.gs.to_datatree().gs.groups == ("default",)


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


def test_an_asset_states_the_grid_a_png_lacks(tmp_path):
    entry, grid = write_png_item(tmp_path)

    scene = combine_item(load_datatree([entry]))

    assert scene.odc.geobox == grid


def test_create_stac_item_describes_saved_assets(tmp_path):
    import rasterio
    from geosave.geodata.io.formats.geotiff import write_cog
    from geosave.geodata.stac.publish import create_stac_item

    dataset = build_raster(times=1)
    paths = {}
    for name in dataset.data_vars:
        paths[name] = write_cog(dataset[[name]], tmp_path / f"{name}.tif")

    entry = create_stac_item(
        paths,
        id="scene",
        input_datetime=datetime(2025, 6, 1, tzinfo=UTC),
        with_proj=True,
        with_raster=True,
        with_eo=True,
    )

    assert set(entry.assets) == {"red", "nir"}
    assert entry.datetime == datetime(2025, 6, 1, tzinfo=UTC)
    for name, asset in entry.assets.items():
        with rasterio.open(paths[name]) as raster:
            assert asset.extra_fields["proj:shape"] == [raster.height, raster.width]
            assert (
                asset.extra_fields["raster:bands"][0]["data_type"] == raster.dtypes[0]
            )
            assert asset.extra_fields["eo:bands"][0]["name"] == name
    assert "proj:shape" not in entry.properties


def test_saved_item_preserves_nominal_time_from_interval_properties(tmp_path):
    from copy import deepcopy
    from geosave.geodata.io.formats.geotiff import write_cog
    from geosave.geodata.stac.publish import create_stac_item

    path = write_cog(build_raster(times=1), tmp_path / "composite.tif")
    properties = {
        "datetime": "2025-06-15T00:00:00Z",
        "start_datetime": "2025-06-01T00:00:00Z",
        "end_datetime": "2025-07-01T00:00:00Z",
        "platform": "s2a",
    }
    before = deepcopy(properties)
    entry = create_stac_item({"image": path}, id="composite", properties=properties)

    assert entry.datetime == datetime(2025, 6, 15, tzinfo=UTC)
    assert entry.to_dict()["properties"]["datetime"] == "2025-06-15T00:00:00Z"
    assert entry.properties["platform"] == "s2a"
    assert properties == before


def _scene():
    source = build_raster(times=2).chunk()
    return source.assign(dem=source.red.isel(time=0, drop=True).astype("float32"))
