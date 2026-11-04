"""stac-geoparquet: Items to rows, to one file, and back to pixels."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pystac
import pytest
from pystac.extensions.raster import RasterExtension
from stac_geoparquet.arrow import stac_table_to_items

from geosave.geodata import GeoCatalog, attrs, open_vector
from geosave.geodata.io import stac as stac_io
from geosave.geodata.io.formats import stac_geoparquet as table
from geosave.geodata.stac.catalog import from_items
from tests.conftest import build_tree
from tests.geodata.conftest import (
    build_raster,
    cog_template,
    create_file_items,
    create_test_collection,
    item_table,
    read_items,
    tree_templates,
    write_collection_cogs,
)
from tests.geodata.stac.conftest import create_native_item, entry


@pytest.mark.parametrize("omitted", [[], ["type", "stac_version"]])
def test_upstream_tables_read_without_optional_item_columns(tmp_path, omitted):
    from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

    from geosave import GeoCatalog

    item = entry("optical")
    item.add_asset("image", pystac.Asset(href="image.tif"))
    collection = create_test_collection("optical", description="Optical imagery")
    arrow = parse_stac_items_to_arrow([item]).read_all().drop(omitted)
    path = tmp_path / "upstream.parquet"
    to_parquet(arrow, path, collections={"optical": collection.to_dict()})

    rows = open_vector(path)

    assert isinstance(rows.gs, GeoCatalog)
    # stac-geoparquet requires neither column; the table reads as written.
    assert set(omitted).isdisjoint(rows.columns)
    assert rows.gs.collections["optical"].description == "Optical imagery"
    assert rows.loc[0, "assets"]["image"]["href"] == str(tmp_path / "image.tif")
    restored = rows.iloc[0].gs.to_item()
    assert restored.id == item.id
    assert restored.to_dict()["type"] == "Feature"
    assert restored.to_dict()["stac_version"] == "1.1.0"


def _refuse(*args, **kwargs):
    raise AssertionError("the table computed pixels")


def _per_band(template: str) -> str:
    """Write one file per band, so each asset is one band named by its key."""
    return template.replace(".tif", "_{band}.tif")


def _cog_items(scene, root, name="forest", **options):
    dated = "time" in scene.dims
    template = _per_band(cog_template(name, dated=dated))
    id = f"{name}_{{start:%Y%m%dT%H%M%S}}" if dated else name
    return (
        item_table(scene.gs.to_cog(root, path_format=template), id=id, **options)
    ).gs.to_items()


def _stack_items(scene, label, root, name="s0"):
    sample = build_tree({"optical": scene, "label": label})
    templates = tree_templates(sample)
    saved = write_collection_cogs(
        sample, root / name, {group: _per_band(t) for group, t in templates.items()}
    )
    return (item_table(saved, prefix=name)).gs.to_items()


def _save(items, path, **options):
    options.setdefault(
        "collections",
        [
            create_test_collection(key, description=key)
            for key in dict.fromkeys(item.collection_id for item in items)
            if key is not None
        ],
    )
    return from_items(items).gs.to_geoparquet(path, **options)


def _collections():
    return [
        create_test_collection(name, description=f"{name} rows", license="CC-BY-4.0")
        for name in ("optical", "label")
    ]


def test_items_linking_to_an_api_root_convert_without_reading_it(
    scene, tmp_path
) -> None:
    import pystac

    # A searched Item links to its API's root; nothing here should fetch it.
    unreachable = "http://127.0.0.1:9/stac/catalog.json"
    items = _cog_items(scene, tmp_path)
    for item in items:
        item.add_link(pystac.Link(pystac.RelType.ROOT, unreachable))

    rows = from_items(items)
    path = rows.gs.to_geoparquet(tmp_path / "items.parquet")

    assert len(open_vector(path)) == len(items)


def test_a_written_table_is_stac_geoparquet(scene, tmp_path) -> None:
    path = _save(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    metadata = pq.read_metadata(path).metadata
    assert b"stac-geoparquet" in metadata
    assert b"geo" in metadata


def test_rows_load_back_as_the_raster_that_was_written(scene, tmp_path) -> None:
    path = _save(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    with dask.config.set(scheduler=_refuse):
        restored = read_items(open_vector(path))

    assert dict(restored.sizes) == {"time": 2, "y": 64, "x": 64}
    np.testing.assert_array_equal(restored.red.values, scene.red.values)
    restored.close()


def test_a_table_follows_its_assets_when_moved(scene, tmp_path) -> None:
    root = tmp_path / "dataset"
    _save(_cog_items(scene, root), root / "items.parquet")

    shutil.move(root, tmp_path / "moved")

    rows = open_vector(tmp_path / "moved" / "items.parquet")
    restored = read_items(rows)
    np.testing.assert_array_equal(restored.red.values, scene.red.values)
    restored.close()


def test_each_group_of_a_tree_loads_alone(scene, label, tmp_path) -> None:
    path = _save(_stack_items(scene, label, tmp_path), tmp_path / "items.parquet")
    rows = open_vector(path)

    optical = read_items(rows[rows["collection"] == "optical"])
    classes = read_items(rows[rows["collection"] == "label"])

    # Reading promises band names, not their order; a model selects by name.
    assert set(optical.gs.variables) == {"red", "nir"}
    assert classes.gs.variables == ("label",)
    optical.close()
    classes.close()


def test_collections_ride_in_the_table_with_the_extent_of_their_rows(
    scene, label, tmp_path
) -> None:
    given = _collections()
    path = _save(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=given,
    )

    stored = table.read_collections(path)

    assert sorted(stored) == ["label", "optical"]
    assert stored["optical"].license == "CC-BY-4.0"
    assert stored["optical"].extent.spatial.bboxes[0] == pytest.approx(
        tuple(scene.gs.geobox.extent.to_crs("EPSG:4326").boundingbox)
    )
    assert stored["optical"].extent.temporal.to_dict()["interval"] == [
        ["2025-06-01T00:00:00Z", "2025-06-11T00:00:00Z"]
    ]
    # The Collections handed in keep the open extent they came with.
    assert given[0].extent.spatial.bboxes == [[-180.0, -90.0, 180.0, 90.0]]


def test_a_collection_a_stac_api_returned_is_stored(scene, label, tmp_path) -> None:
    from pystac_client import Client, CollectionClient
    from pystac_client.stac_api_io import StacApiIO

    # What `StacClient.collection` returns: a Collection bound to its API.
    api = Client(id="api", description="A STAC API")
    api._stac_io = StacApiIO()
    remote = [
        CollectionClient.from_dict(given.to_dict(), root=api)
        for given in _collections()
    ]

    path = _save(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=remote,
    )

    assert table.read_collections(path)["optical"].description == "optical rows"


def test_only_the_collections_given_are_stored(scene, label, tmp_path) -> None:
    path = _save(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=_collections()[:1],
    )

    assert set(table.read_collections(path)) == {"optical"}


def test_a_batch_concatenated_onto_a_table_reads_back_whole(
    scene, label, tmp_path
) -> None:
    path = _save(
        _stack_items(scene, label, tmp_path, "s0"),
        tmp_path / "items.parquet",
        collections=_collections(),
    )
    more = from_items(_stack_items(scene, label, tmp_path, "s1"))

    grown = GeoCatalog.concat([open_vector(path), more])
    grown.gs.to_geoparquet(
        path,
        collections=table.read_collections(path).values(),
        overwrite=True,
    )

    rows = open_vector(path)
    assert sorted({id.split("/")[0] for id in rows["id"]}) == ["s0", "s1"]
    assert sorted(table.read_collections(path)) == ["label", "optical"]
    restored = read_items(
        rows[rows["id"].str.startswith("s1/") & (rows["collection"] == "optical")]
    )
    np.testing.assert_array_equal(restored.red.values, scene.red.values)
    restored.close()


def test_a_bbox_filter_reads_only_matching_rows(scene, tmp_path) -> None:
    path = _save(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    assert open_vector(path, bbox=(0.0, 0.0, 1.0, 1.0)).empty


def test_a_table_that_is_not_stac_is_returned_as_read(tmp_path) -> None:
    from shapely.geometry import Point

    plain = gpd.GeoDataFrame(
        [{"id": "a", "assets": {"label": {"href": "a.tif"}}, "quality": 0.97}],
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    path = plain.gs.to_geoparquet(tmp_path / "labels.parquet")

    rows = open_vector(path)

    assert rows.loc[0, "quality"] == 0.97
    assert rows.loc[0, "assets"]["label"]["href"] == "a.tif"
    assert type(rows.gs) is GeoCatalog


def test_a_vector_that_is_not_an_item_table_does_not_write_as_one(tmp_path) -> None:
    from shapely.geometry import Point

    plain = gpd.GeoDataFrame([{"id": "a"}], geometry=[Point(0, 0)], crs="EPSG:4326")

    with pytest.raises(ValueError, match="item table"):
        table.write_stac_geoparquet(plain, tmp_path / "plain.parquet")
    written = plain.gs.to_geoparquet(tmp_path / "plain.parquet")
    assert table.read_collections(written) == {}


def test_filtered_rows_do_not_gain_their_index_as_a_property(
    scene, label, tmp_path
) -> None:
    rows = from_items(_stack_items(scene, label, tmp_path), collections=_collections())

    (item,) = (rows[rows["collection"] == "label"]).gs.to_items()

    assert "__index_level_0__" not in item.properties


def test_selected_columns_of_an_item_table_read_as_selected(
    scene, label, tmp_path
) -> None:
    path = _save(_stack_items(scene, label, tmp_path), tmp_path / "c.parquet")

    rows = open_vector(path, columns=["id", "geometry", "stac_version"])

    assert sorted(rows.columns) == ["geometry", "id", "stac_version"]


def test_unused_collection_definitions_are_not_written(scene, tmp_path) -> None:
    rows = from_items(_cog_items(scene, tmp_path))
    path = rows.gs.to_geoparquet(
        tmp_path / "filtered.parquet", collections=_collections()
    )
    assert table.read_collections(path) == {}


def test_writing_leaves_the_rows_and_collections_unchanged(
    scene, label, tmp_path
) -> None:
    rows = from_items(_stack_items(scene, label, tmp_path), collections=_collections())
    given = _collections()
    before = (
        [item.to_dict() for item in rows.gs.to_items()],
        [each.to_dict() for each in given],
    )

    rows.gs.to_geoparquet(tmp_path / "items.parquet", collections=given)

    assert (
        [item.to_dict() for item in rows.gs.to_items()],
        [each.to_dict() for each in given],
    ) == before


def test_importing_core_and_io_loads_no_stac() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import geosave.geodata.core, geosave.geodata.io; "
        "assert not {'pystac', 'rustac', 'stac_geoparquet'} & set(sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_described_rasters_round_trip_through_the_table(tmp_path) -> None:
    optical = build_raster(times=2, packed=True)[["red"]]
    optical = optical.gs.write_spectral("red", center_wavelength=0.665, target="red")
    optical = optical.assign(red=optical.red.assign_attrs(units="1"))
    label = build_raster()[["nir"]].astype("uint8").rename(nir="label")
    label = label.gs.write_legend(
        {0: "background", 1: "forest"}, {1: "#00ff00"}, target="label"
    )
    when = datetime(2025, 6, 1, tzinfo=UTC)
    optical_scenes = optical.gs.to_cog(
        tmp_path, path_format="optical_{time:%Y%m%d}_{band}.tif"
    )
    (label_scene,) = label.gs.to_cog(tmp_path, path_format="{band}.tif")
    rows = from_items(
        [
            *create_file_items(optical_scenes, id_format="optical_{start:%Y%m%d}"),
            *create_file_items(label_scene, id_format="label", datetime=when),
        ]
    )

    saved = rows.gs.to_geoparquet(tmp_path / "items.parquet")

    restored = (open_vector(saved)).gs.to_items()
    red = stac_io.open_scene(restored[0]).gs.to_dataset().red
    label = stac_io.open_scene(restored[-1]).gs.to_dataset().label
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].data_type == "uint16"
    assert red.attrs["units"] == "1"
    assert red.gs.packing.scale_factor == pytest.approx(1e-4)
    assert red.gs.spectral.common_name == "red"
    assert red.gs.spectral.center_wavelength == 0.665
    assert red.gs.nodata == optical.red.gs.nodata
    legend = attrs.Legend.model_validate(label.attrs)
    assert legend.classes == {0: "background", 1: "forest"}


def test_collection_survives_read_filter_rewrite(tmp_path):
    collection = create_test_collection(
        "optical", description="Surface reflectance", license="CC-BY-4.0"
    )
    collection.extra_fields["custom:version"] = "v1"
    rows = from_items([entry()], collections=[collection])
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


def test_collection_links_and_nested_fields_survive_publication(tmp_path):
    collection = create_test_collection(
        "optical", description="Reflectance", keywords=["optical"]
    )
    collection.add_link(pystac.Link("license", "https://example.com/license"))
    rows = from_items([entry()], collections=[collection])
    collection.keywords.append("changed")
    assert rows.gs.collections["optical"].keywords == ["optical"]
    saved = rows.gs.to_geoparquet(tmp_path / "links.parquet")
    restored = open_vector(saved).gs.collections["optical"]
    assert restored.get_single_link("license").target == "https://example.com/license"


def test_a_written_table_retains_item_and_asset_properties(tmp_path):
    first, source = create_native_item(tmp_path, "first")
    second, _ = create_native_item(tmp_path, "second")
    bands = [
        {"name": name, "data_type": source[name].dtype.name}
        for name in source.gs.variables
    ]
    first.assets["reflectance"].extra_fields["bands"] = bands
    second.properties["new:property"] = 7
    second.assets["other"] = second.assets.pop("reflectance")

    saved = from_items([second, first]).gs.to_geoparquet(tmp_path / "items.parquet")

    restored = list(stac_table_to_items(pq.read_table(saved)))
    assert [entry["id"] for entry in restored] == ["second", "first"]
    assert restored[0]["properties"]["new:property"] == 7
    assert restored[1]["assets"]["reflectance"]["bands"] == bands


def test_a_table_keeps_its_bbox_through_read_and_rewrite(tmp_path):
    entry, _ = create_native_item(tmp_path)
    first = from_items([entry]).gs.to_geoparquet(tmp_path / "catalog.parquet")
    second = open_vector(first).gs.to_geoparquet(tmp_path / "again.parquet")

    for path in (first, second):
        (restored,) = stac_table_to_items(pq.read_table(path))
        assert restored["bbox"] == pytest.approx(entry.bbox)


def test_an_edited_geometry_changes_the_bbox_written(tmp_path):
    entry, _ = create_native_item(tmp_path)
    rows = from_items([entry])
    rows = rows.set_geometry(rows.geometry.translate(xoff=1.0))

    (restored,) = stac_table_to_items(
        pq.read_table(rows.gs.to_geoparquet(tmp_path / "catalog.parquet"))
    )

    assert restored["bbox"][0] == pytest.approx(entry.bbox[0] + 1.0)


@pytest.mark.parametrize("folder", ["dataset", "data set"])
def test_native_table_follows_its_assets_when_moved(tmp_path, folder):
    home = tmp_path / folder
    home.mkdir()
    entry, _ = create_native_item(home)
    saved = from_items([entry]).gs.to_geoparquet(home / "catalog.parquet")
    stored = pq.read_table(saved).column("assets")[0].as_py()
    assert stored["reflectance"]["href"] == "./scene.tif"
    assert stored["thumbnail"]["href"] == "https://example.com/thumb.jpg"

    moved = tmp_path / "moved"
    shutil.move(home, moved)

    href = open_vector(moved / "catalog.parquet").iloc[0].assets["reflectance"]["href"]
    assert href == str(moved / "scene.tif")
    assert Path(href).is_file()


def test_a_catalog_writes_stac_geoparquet_that_validates(rows, tmp_path) -> None:
    path = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")

    restored = (open_vector(path)).gs.to_items()
    assert [item.id for item in restored] == rows["id"].tolist()
    for item in restored:
        item.validate()


def test_a_property_date_given_as_datetime_becomes_iso_text():
    from datetime import UTC, datetime

    from geosave.geodata.stac.catalog import build_item

    fields = {
        "type": "Feature",
        "id": "a",
        "geometry": None,
        "bbox": None,
        "links": [],
        "assets": {},
        "stac_extensions": [],
        "properties": {"datetime": datetime(2024, 1, 5, tzinfo=UTC)},
    }

    assert build_item(fields).datetime == datetime(2024, 1, 5, tzinfo=UTC)
