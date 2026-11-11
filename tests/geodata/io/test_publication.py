"""Publication preserves pixels and preflights the complete batch."""

from pathlib import Path

import pytest
import rasterio
import numpy as np

from tests.geodata.conftest import build_raster


def test_publish_observations_as_items_and_cogs(tmp_path):
    dataset = build_raster(times=2)
    catalog = dataset.gs.to_catalog(tmp_path)

    items = catalog.gs.to_items()
    assert len(items) == 2
    for index, item in enumerate(items):
        assert set(item.assets) == {"red", "nir"}
        with rasterio.open(item.assets["red"].href) as raster:
            np.testing.assert_array_equal(raster.read(1), dataset.red.isel(time=index))
            assert raster.transform == dataset.gs.geobox.transform


def test_later_path_collision_fails_before_writing(tmp_path):
    dataset = build_raster(times=2)
    with pytest.raises(ValueError, match="collide"):
        dataset.gs.to_catalog(tmp_path, format_paths="{time:%Y}/{variable}.tif")
    assert not list(Path(tmp_path).rglob("*.tif"))


def test_cf_interval_and_collection_survive_publication_round_trip(tmp_path):
    import json
    import pyarrow.parquet as pq
    from geosave import open_vector
    from tests.geodata.conftest import create_test_collection

    from tests.geodata.conftest import assign_cells

    dataset = assign_cells(
        build_raster(times=1),
        [
            (np.datetime64("2025-06-01"), np.datetime64("2025-07-01")),
        ]
    )
    collection = create_test_collection("optical", description="Optical measurements")
    catalog = dataset.gs.to_catalog(tmp_path, collection=collection)
    path = catalog.gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = open_vector(path)
    item = restored.gs.to_items()[0]
    assert item.common_metadata.start_datetime.month == 6
    assert item.common_metadata.end_datetime.month == 7
    assert item.get_collection().extent.to_dict() == collection.extent.to_dict()
    assert Path(item.assets["red"].href).is_absolute()

    metadata = pq.read_metadata(path).metadata
    assert json.loads(metadata[b"geo"])["version"] == "1.1.0"
    assert json.loads(metadata[b"stac-geoparquet"])["version"] == "1.0.0"


def test_array_bands_and_tree_grids_remain_separate_assets(tmp_path):
    import xarray as xr

    dataset = build_raster(times=1)
    item = dataset.gs.to_dataarray().gs.to_item(tmp_path / "array")
    assert set(item.assets) == {"red", "nir"}

    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords={"time": dataset.time}),
            "fine": dataset.drop_vars("time"),
            "crop": dataset.isel(x=slice(0, 1)).drop_vars("time"),
        }
    )
    item = tree.gs.to_item(tmp_path / "tree")
    assert item.assets["fine/red"].extra_fields["proj:shape"] == [2, 2]
    assert item.assets["crop/red"].extra_fields["proj:shape"] == [2, 1]


def test_collection_links_are_preserved_without_inventing_document_urls(tmp_path):
    from geosave import open_vector
    from tests.geodata.conftest import create_test_collection

    dataset = build_raster(times=1)
    for name, href in (
        ("embedded", None),
        ("linked", "https://example.org/collection.json"),
    ):
        collection = create_test_collection(name, description="Measurements")
        collection.set_self_href(href)
        catalog = dataset.gs.to_catalog(tmp_path / name, collection=collection)
        path = catalog.gs.to_geoparquet(tmp_path / f"{name}.parquet")
        restored = open_vector(path)
        item = restored.gs.to_items()[0]

        assert item.get_collection().get_self_href() == href
        assert collection.get_self_href() == href
        assert item.collection_id == name


def test_explicit_item_id_overrides_format(tmp_path):
    dataset = build_raster(times=1)
    item = dataset.gs.to_item(tmp_path, id="observation", format_id="{missing}")
    assert item.id == "observation"


def test_static_variable_and_shared_metadata_accompany_each_observation(tmp_path):
    from tests.geodata.conftest import create_test_collection

    dataset = build_raster(times=2)
    dataset["dem"] = dataset.red.isel(time=0, drop=True).gs.write_nodata(np.uint16(0))
    dataset.dem.attrs["valid_min"] = np.uint16(0)
    collection = create_test_collection("observations", description="Measurements")
    dataset = dataset.assign_attrs(
        stac_collection=collection.to_dict(),
    )
    from geosave.geodata.stac.raster import stamp_properties_coordinate

    dataset = stamp_properties_coordinate(
        dataset, [{"platform": "example"}, {"platform": "example"}]
    )
    original = dataset.copy(deep=True)

    catalog = dataset.gs.to_catalog(tmp_path)

    for item in catalog.gs.to_items():
        assert item.collection_id == collection.id
        assert item.properties["platform"] == "example"
        assert "dem" in item.assets
        with rasterio.open(item.assets["dem"].href) as raster:
            assert raster.nodata == 0
            assert "_FillValue" not in raster.tags()
            assert raster.tags(1)["valid_min"] == "0"
            np.testing.assert_array_equal(raster.read(1), dataset.dem)

    import xarray as xr

    xr.testing.assert_identical(dataset, original)


def test_variable_and_band_labels_remain_distinct_in_saved_assets(tmp_path):
    import xarray as xr
    from geosave.geodata.io.rasterio import open_raster

    array = build_raster(times=1).gs.to_dataarray().rename("reflectance")
    dataset = array.to_dataset()
    original = dataset.copy(deep=True)
    item = dataset.gs.to_item(
        tmp_path,
        split_asset=("variable", "band"),
        format_paths="{id}/{variable}_{band}.tif",
    )

    assert set(item.assets) == {"reflectance_red", "reflectance_nir"}
    for band in ("red", "nir"):
        with open_raster(
            item.assets[f"reflectance_{band}"].href, chunks=None
        ) as restored:
            assert list(restored.data_vars) == ["reflectance"]
            np.testing.assert_array_equal(
                restored.reflectance, array.sel(band=band).squeeze("time")
            )
    xr.testing.assert_identical(dataset, original)
