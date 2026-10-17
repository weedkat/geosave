"""Observation publication and catalog metadata contracts."""

from datetime import UTC, datetime

import pystac
import pytest

from geosave.geodata import open_vector, stac
from geosave.geodata.core import GeoCatalog
from tests.geodata.conftest import build_raster


def entry(collection="optical"):
    item = pystac.Item(
        "one",
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        [0, 0, 1, 1],
        datetime(2025, 1, 1, tzinfo=UTC),
        {},
        collection=collection,
    )
    item.add_asset(
        "red", pystac.Asset("/tmp/red.tif", media_type=pystac.MediaType.GEOTIFF)
    )
    return item


def test_collection_survives_read_filter_rewrite(tmp_path):
    collection = stac.create_collection(
        "optical", description="Surface reflectance", license="CC-BY-4.0"
    )
    collection.extra_fields["custom:version"] = "v1"
    rows = stac.create_catalog([entry()], collections=[collection])
    saved = rows.gs.to_geoparquet(tmp_path / "first.parquet")
    restored = open_vector(saved)
    second = restored.iloc[:1].gs.to_geoparquet(tmp_path / "second.parquet")
    assert (
        open_vector(second).gs.collections["optical"].description
        == collection.description
    )
    assert (
        open_vector(second).gs.collections["optical"].extra_fields["custom:version"]
        == "v1"
    )
    assert collection.extent.spatial.bboxes == [[-180.0, -90.0, 180.0, 90.0]]


def test_concat_merges_collections_and_keeps_the_first_definition():
    optical = stac.create_collection("optical", description="Reflectance")
    label = stac.create_collection("label", description="Land cover")
    a = stac.create_catalog([entry()], collections=[optical])
    b = stac.create_catalog([entry("label")], collections=[label])
    assert set(GeoCatalog.concat([a, b]).gs.collections) == {"optical", "label"}
    optical.description = "Conflicting description"
    c = stac.create_catalog([entry()], collections=[optical])
    assert (
        GeoCatalog.concat([a, c]).gs.collections["optical"].description == "Reflectance"
    )
    assert (
        GeoCatalog.concat([c, a]).gs.collections["optical"].description
        == "Conflicting description"
    )


def test_missing_collection_is_not_fabricated(tmp_path):
    rows = stac.create_catalog([entry()])
    with pytest.raises(ValueError, match="missing.*optical"):
        rows.gs.to_geoparquet(tmp_path / "missing.parquet")


@pytest.mark.parametrize("compute", [True, False])
def test_properties_survive_per_band_cog_writing(tmp_path, compute):
    ds = build_raster(times=2).assign_coords({"eo:cloud_cover": ("time", [12.0, 65.0])})
    scenes = ds.gs.to_cog(str(tmp_path / "{time:%Y%m%d}_{band}.tif"), compute=compute)
    items = list(stac.create_items([scene.compute() for scene in scenes]))
    assert [item.properties["eo:cloud_cover"] for item in items] == [12.0, 65.0]
    assert any("eo/" in schema for schema in items[0].stac_extensions)


def test_json_overwrite_keeps_assets_and_unrelated_files(tmp_path):
    from geosave.geodata.io.vector import stac_json

    item = entry(None)
    stac_json.write([item], tmp_path)
    asset = tmp_path / "pixels.tif"
    asset.write_bytes(b"raster")
    notes = tmp_path / "notes.txt"
    notes.write_text("keep")
    stac_json.write([item], tmp_path, overwrite=True)
    assert asset.read_bytes() == b"raster"
    assert notes.read_text() == "keep"


def test_composite_keeps_nominal_time_and_bounds(tmp_path):
    from geosave.geodata.attrs import TimeSpec

    ds = build_raster(times=1)
    ds.time.attrs.update(TimeSpec.from_resample("MS").to_attrs())
    scene = ds.gs.to_cog(str(tmp_path / "composite.tif"))[0]
    item = next(stac.create_items(scene))
    assert item.datetime == datetime(2025, 6, 1, tzinfo=UTC)
    assert item.properties["start_datetime"].startswith("2025-06-01")
    assert item.properties["end_datetime"].startswith("2025-06-30")


def test_unsplit_multidate_store_cannot_be_one_observation(tmp_path):
    ds = build_raster(times=2)
    scenes = ds.gs.to_netcdf(tmp_path / "cube.nc")
    with pytest.raises(ValueError, match="split multiple dates"):
        list(stac.create_items(scenes))
    split = [
        ds.isel(time=index).gs.to_netcdf(tmp_path / f"date-{index}.nc")
        for index in range(2)
    ]
    assert len(list(stac.create_items(split))) == 2


@pytest.mark.parametrize("value", [-1.0, 101.0, float("inf")])
def test_invalid_cloud_cover_rejected(tmp_path, value):
    ds = build_raster(times=1).assign_coords({"eo:cloud_cover": value})
    scenes = ds.gs.to_cog(tmp_path / "scene.tif")
    with pytest.raises(ValueError, match="eo:cloud_cover"):
        list(stac.create_items(scenes))


def test_missing_and_dropped_properties_are_not_exported(tmp_path):
    ds = build_raster(times=1).assign_coords(
        {"eo:cloud_cover": float("nan"), "unknown:value": 12}
    )
    item = next(stac.create_items(ds.gs.to_cog(tmp_path / "missing.tif")))
    assert "eo:cloud_cover" not in item.properties
    assert "unknown:value" not in item.properties
    ds = ds.assign_coords({"eo:cloud_cover": 12}).drop_vars("eo:cloud_cover")
    item = next(stac.create_items(ds.gs.to_cog(tmp_path / "dropped.tif")))
    assert "eo:cloud_cover" not in item.properties


def test_property_rename_is_rejected():
    from geosave.geodata.stac.source import StacSourceConfig

    with pytest.raises(ValueError, match="exact STAC key"):
        StacSourceConfig(with_properties=[{"key": "eo:cloud_cover", "name": "cloud"}])


def test_core_band_fields_are_read():
    import xarray as xr
    from geosave.geodata.stac.translate import band

    asset = pystac.Asset(
        "/tmp/multi.tif",
        extra_fields={
            "bands": [
                {"name": "red", "eo:common_name": "red", "raster:scale": 0.01},
                {"name": "nir", "eo:common_name": "nir"},
            ]
        },
    )
    assert band.read_aliases(asset) == [("red", "red"), ("nir", "nir")]
    assert band.read_attrs(xr.DataArray(0), asset, 0)["scale_factor"] == 0.01


def test_native_collection_property_does_not_search():
    from unittest.mock import Mock
    from geosave.geodata.stac.source import StacSource

    collection = stac.create_collection("optical", description="Reflectance")
    client = Mock()
    client.collection.return_value = collection
    source = StacSource(client, collection="optical")
    assert source.collection is collection
    client.search_items.assert_not_called()


def test_concat_feeds_multicollection_stacks(tmp_path):
    from geosave.geodata import tree

    ds = build_raster(times=1)
    layers = tree({"optical": ds, "label": ds[["red"]].rename(red="label")})
    scenes = layers.gs.to_cog(str(tmp_path / "{collection}_{time:%Y%m%d}.tif"))
    catalogs = []
    for scene in scenes:
        collection = stac.create_collection(
            scene.collection, description=scene.collection
        )
        catalogs.append(
            stac.create_catalog(stac.create_items(scene), collections=[collection])
        )
    combined = GeoCatalog.concat(catalogs)
    samples = combined.gs.sample()
    assert len(samples) == 1
    assert set(samples.iloc[0].gs.items) == {"optical", "label"}
    loaded = samples.iloc[0].gs.to_xarray()
    assert set(loaded.children) == {"optical", "label"}
    assert loaded["optical"].dataset.red.chunks is not None


def test_item_id_and_geometry_do_not_depend_on_asset_order(tmp_path):
    from geosave.geodata.io.raster.file import Scene

    ds = build_raster(times=1)
    scene = ds.gs.to_cog(str(tmp_path / "{band}.tif"))[0]
    reversed_scene = Scene(
        scene.collection,
        dict(reversed(list(scene.written.items()))),
        coords=scene.coords,
    )
    first = next(stac.create_items(scene))
    second = next(stac.create_items(reversed_scene))
    assert (first.id, first.geometry) == (second.id, second.geometry)


def test_duplicate_items_rejected_by_both_writers(tmp_path):
    from geosave.geodata.io.vector import stac_json, stac_geoparquet
    from geosave.geodata.stac.translate.table import from_items

    item = entry(None)
    rows = from_items([item, item])
    with pytest.raises(ValueError, match="share the id"):
        stac_geoparquet.write(rows, tmp_path / "duplicate.parquet")
    with pytest.raises(ValueError, match="share the id"):
        stac_json.write([item, item], tmp_path / "json")


def test_concat_keeps_the_first_definition_even_after_filtering_rows():
    optical = stac.create_collection("optical", description="Old optical")
    label = stac.create_collection("label", description="Labels")
    rows = stac.create_catalog([entry(), entry("label")], collections=[optical, label])
    optical.description = "New optical"
    fresh = stac.create_catalog([entry()], collections=[optical])
    combined = GeoCatalog.concat([rows[rows.collection == "label"], fresh])
    assert combined.gs.collections["optical"].description == "Old optical"


def test_collection_links_and_nested_fields_survive_publication(tmp_path):
    collection = stac.create_collection(
        "optical", description="Reflectance", keywords=["optical"]
    )
    collection.add_link(pystac.Link("license", "https://example.com/license"))
    rows = stac.create_catalog([entry()], collections=[collection])
    collection.keywords.append("changed")
    assert rows.gs.collections["optical"].keywords == ["optical"]
    saved = rows.gs.to_geoparquet(tmp_path / "links.parquet")
    restored = open_vector(saved).gs.collections["optical"]
    assert restored.get_single_link("license").target == "https://example.com/license"


def test_source_properties_fuse_and_publish_through_geoparquet(local_source, tmp_path):
    from unittest.mock import Mock
    import numpy as np
    import xarray as xr

    source, anchor = local_source
    collection = source.collection
    first = source.search(anchor).gs.to_items()[0]
    first.properties["eo:cloud_cover"] = 20.0
    second = first.clone()
    second.id = "second"
    second.properties["eo:cloud_cover"] = 40.0
    client = Mock()
    client.collection.return_value = collection
    client.search_items.return_value = [first, second]
    source.client = client
    loaded = source.set_config(with_properties=["eo:cloud_cover"]).load(anchor)
    loaded = loaded.set_coords("eo:cloud_cover")
    assert loaded["eo:cloud_cover"].dims == ("time",)
    assert loaded["eo:cloud_cover"].item() == 30.0
    assert loaded.red.chunks is not None
    file = loaded.gs.to_netcdf(tmp_path / "property.nc")
    with xr.open_dataset(file.written["image"].path) as stored:
        np.testing.assert_equal(stored["eo:cloud_cover"].values, [30.0])
    scenes = loaded.gs.to_cog(tmp_path / "processed.tif")
    output = stac.create_collection("processed", description="Processed observations")
    rows = stac.create_catalog(
        stac.create_items(scenes, collection=output), collections=[output]
    )
    saved = rows.gs.to_geoparquet(tmp_path / "processed.parquet")
    assert open_vector(saved).gs.to_items()[0].properties["eo:cloud_cover"] == 30.0


@pytest.mark.parametrize("existing_catalog", [False, True])
def test_json_refuses_unrelated_destination_collision(tmp_path, existing_catalog):
    from geosave.geodata.io.vector import stac_json

    if existing_catalog:
        stac_json.write([entry(None)], tmp_path)
    second = entry(None)
    second.id = "second"
    destination = tmp_path / "second" / "second.json"
    destination.parent.mkdir()
    destination.write_text('{"user": "keep"}')
    with pytest.raises(FileExistsError, match="not owned"):
        stac_json.write([second], tmp_path, overwrite=True)
    assert destination.read_text() == '{"user": "keep"}'


def test_equal_explicit_interval_is_an_instant(tmp_path):
    ds = build_raster()
    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    item = next(
        stac.create_items(
            ds.gs.to_cog(tmp_path / "undated.tif"), datetime=(stamp, stamp)
        )
    )
    assert item.datetime == stamp


def test_native_composite_load_retains_bounds(tmp_path):
    from geosave.geodata.attrs import TimeSpec
    from geosave.geodata.io.raster.stac import open_raster

    ds = build_raster(times=1)
    ds.time.attrs.update(TimeSpec.from_resample("MS").to_attrs())
    original = next(stac.create_items(ds.gs.to_cog(tmp_path / "original.tif")))
    loaded = open_raster([original])
    rebuilt = next(stac.create_items(loaded.gs.to_cog(tmp_path / "rebuilt.tif")))
    assert rebuilt.datetime == original.datetime
    assert rebuilt.properties["start_datetime"] == original.properties["start_datetime"]
    assert rebuilt.properties["end_datetime"] == original.properties["end_datetime"]


def test_native_json_preserves_relative_descriptive_links(tmp_path):
    from geosave.geodata.io.vector import stac_json

    item = entry(None)
    item.set_self_href(str(tmp_path / "source" / "item.json"))
    item.add_link(pystac.Link("license", "./license.txt"))
    root = stac_json.write([item], tmp_path / "published")
    restored = next(pystac.Catalog.from_file(str(root)).get_items(recursive=True))
    assert restored.get_single_link("license").get_absolute_href() == str(
        tmp_path / "source" / "license.txt"
    )
    assert item.get_single_link("license").target == "./license.txt"


def test_stac_source_does_not_invent_composite_bounds(local_source, tmp_path):
    from unittest.mock import Mock

    source, anchor = local_source
    item = source.search(anchor).gs.to_items()[0]
    item.properties.update(
        start_datetime="2025-01-01T00:00:00Z", end_datetime="2025-01-31T23:59:59Z"
    )
    collection = source.collection
    source.client = Mock()
    source.client.search_items.return_value = [item]
    source.client.collection.return_value = collection
    loaded = source.load(anchor)
    rebuilt = next(stac.create_items(loaded.gs.to_cog(tmp_path / "composite.tif")))
    assert "start_datetime" not in loaded
    assert "end_datetime" not in loaded
    assert "start_datetime" not in rebuilt.properties
    assert "end_datetime" not in rebuilt.properties


def test_item_identity_uses_union_footprint_and_grid_fields_require_one_grid(tmp_path):
    from geosave.geodata.io.raster.file import RasterFile, Scene

    ds = build_raster(times=1).isel(time=0)
    shifted = ds.assign_coords(x=ds.x + 1000)
    files = {
        "first": RasterFile.from_xarray(tmp_path / "first.tif", ds),
        "second": RasterFile.from_xarray(tmp_path / "second.tif", shifted),
    }
    first = next(stac.create_items(Scene(None, files)))
    second = next(stac.create_items(Scene(None, dict(reversed(list(files.items()))))))
    assert (first.id, first.geometry) == (second.id, second.geometry)
    with pytest.raises(ValueError, match="common grid"):
        next(stac.create_items(Scene(None, files), id_format="{width}"))


def test_concat_refuses_mixing_item_and_sampling_rows():
    import geopandas as gpd

    rows = stac.create_catalog([entry(None)])
    sampling = gpd.GeoDataFrame(
        {"sources": ["{}"]}, geometry=rows.geometry, crs=rows.crs
    )
    with pytest.raises(ValueError, match="different kinds"):
        GeoCatalog.concat([rows, sampling])
