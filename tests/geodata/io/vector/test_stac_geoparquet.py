"""stac-geoparquet: Items to rows, to one file, and back to pixels."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import dask
import numpy as np
import pyarrow.parquet as pq
import pystac
import pytest
from pystac.extensions.raster import RasterExtension
from stac_geoparquet.arrow import stac_table_to_items

from geosave.geodata import GeoVector, attrs, open_vector, stac, tree
from geosave.geodata.io.raster import stac as stac_io
from geosave.geodata.io.vector import stac_geoparquet as table
from geosave.geodata.stac import create_catalog, create_items
from geosave.geodata.stac.translate import table as stac_table
from tests.geodata.conftest import (
    build_raster,
    cog_template,
    item_table,
    read_items,
    tree_templates,
)
from tests.geodata.stac.conftest import create_native_item, entry


def _refuse(*args, **kwargs):
    raise AssertionError("the table computed pixels")


def _per_band(template: str) -> str:
    """Write one file per band, so each asset is one band named by its key."""
    return template.replace(".tif", "_{band}.tif")


def _cog_items(scene, root, name="forest", **options):
    dated = "time" in scene.dims
    template = _per_band(cog_template(root, name, dated=dated))
    id = f"{name}_{{start:%Y%m%dT%H%M%S}}" if dated else name
    return stac_table.to_items(item_table(scene.gs.to_cog(template), id=id, **options))


def _stack_items(scene, label, root, name="s0"):
    sample = tree({"optical": scene, "label": label})
    templates = tree_templates(sample, root / name)
    saved = sample.gs.to_cog({group: _per_band(t) for group, t in templates.items()})
    return stac_table.to_items(item_table(saved, prefix=name))


def _save(items, path, **options):
    options.setdefault(
        "collections",
        [
            stac.create_collection(key, description=key)
            for key in dict.fromkeys(item.collection_id for item in items)
            if key is not None
        ],
    )
    return stac_table.from_items(items).gs.to_geoparquet(path, **options)


def _collections():
    return [
        stac.create_collection(name, description=f"{name} rows", license="CC-BY-4.0")
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

    rows = stac_table.from_items(items)
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


def test_read_rows_carry_hrefs_that_open(scene, tmp_path) -> None:
    items = _cog_items(scene, tmp_path)
    path = _save(items, tmp_path / "catalog" / "items.parquet")

    rows = open_vector(path)

    assert rows["assets"][0]["red"]["href"] == items[0].assets["red"].href


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


def test_only_data_assets_load(scene, tmp_path) -> None:
    import pystac

    (item, *_) = _cog_items(scene, tmp_path)
    item.add_asset("thumbnail", pystac.Asset("preview.png", roles=["thumbnail"]))

    restored = read_items(stac_table.from_items([item]))

    assert set(restored.gs.variables) == {"red", "nir"}
    restored.close()


def test_an_existing_table_is_not_replaced_unasked(scene, tmp_path) -> None:
    items = _cog_items(scene, tmp_path)
    path = _save(items, tmp_path / "items.parquet")

    with pytest.raises(FileExistsError):
        _save(items, path)


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


def test_items_of_a_stored_collection_link_to_it(scene, label, tmp_path) -> None:
    path = _save(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=_collections(),
    )

    (item, *_) = stac_table.to_items(open_vector(path))

    assert [link.rel for link in item.links] == ["collection"]


def test_a_row_naming_an_absent_collection_is_rejected(scene, label, tmp_path) -> None:
    with pytest.raises(ValueError, match="missing.*label"):
        _save(
            _stack_items(scene, label, tmp_path),
            tmp_path / "items.parquet",
            collections=_collections()[:1],
        )


def test_a_table_without_collections_reads_none(scene, tmp_path) -> None:
    path = _save(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    assert table.read_collections(path) == {}


def test_a_batch_concatenated_onto_a_table_reads_back_whole(
    scene, label, tmp_path
) -> None:
    path = _save(
        _stack_items(scene, label, tmp_path, "s0"),
        tmp_path / "items.parquet",
        collections=_collections(),
    )
    more = stac_table.from_items(_stack_items(scene, label, tmp_path, "s1"))

    grown = GeoVector.concat([open_vector(path), more])
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


def test_a_rewrite_preserves_attached_collections(scene, label, tmp_path) -> None:
    path = _save(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=_collections(),
    )

    open_vector(path).gs.to_geoparquet(path, overwrite=True)

    stored = table.read_collections(path)
    assert sorted(stored) == ["label", "optical"]
    assert stored["optical"].description == "optical rows"


def test_a_bbox_filter_reads_only_matching_rows(scene, tmp_path) -> None:
    path = _save(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    assert open_vector(path, bbox=(0.0, 0.0, 1.0, 1.0)).empty


def test_a_table_that_is_not_stac_is_returned_as_read(tmp_path) -> None:
    from shapely.geometry import Point

    plain = GeoVector.from_geometry(
        Point(0, 0),
        properties={"id": "a", "assets": {"label": {"href": "a.tif"}}, "quality": 0.97},
    )
    path = plain.gs.to_geoparquet(tmp_path / "labels.parquet")

    rows = open_vector(path)

    assert rows.loc[0, "quality"] == 0.97
    assert rows.loc[0, "assets"]["label"]["href"] == "a.tif"


def test_an_item_table_writes_through_the_vector_accessor(
    scene, label, tmp_path
) -> None:
    rows = stac.create_catalog(
        _stack_items(scene, label, tmp_path), collections=_collections()
    )

    path = rows.gs.to_geoparquet(
        tmp_path / "catalog.parquet", collections=_collections()
    )

    assert sorted(table.read_collections(path)) == ["label", "optical"]
    assert open_vector(path)["id"].tolist() == rows["id"].tolist()
    with pytest.raises(FileExistsError):
        rows.gs.to_geoparquet(path)


def test_a_vector_that_is_not_an_item_table_does_not_write_as_one(tmp_path) -> None:
    from shapely.geometry import Point

    plain = GeoVector.from_geometry(Point(0, 0), properties={"id": "a"})

    with pytest.raises(ValueError, match="item table"):
        table.write(plain, tmp_path / "plain.parquet")
    written = plain.gs.to_geoparquet(tmp_path / "plain.parquet")
    assert table.read_collections(written) == {}


def test_open_vector_reads_an_item_table_with_hrefs_that_open(
    scene, label, tmp_path
) -> None:
    path = _save(_stack_items(scene, label, tmp_path), tmp_path / "catalog.parquet")
    shutil.move(tmp_path, tmp_path.with_name("moved"))
    moved = tmp_path.with_name("moved") / path.name

    rows = open_vector(moved)

    samples = rows.gs.sample()

    assert len(samples) == 1
    sample = samples.iloc[0].gs.to_xarray()
    np.testing.assert_array_equal(
        sample.gs.rasters["optical"].red.values, scene.red.values
    )


def test_a_stored_table_names_no_absolute_location(scene, label, tmp_path) -> None:
    items = _stack_items(scene, label, tmp_path)
    collections = [
        stac.create_collection(name, description=name) for name in ("optical", "label")
    ]
    path = _save(items, tmp_path / "catalog.parquet", collections=collections)

    stored = pq.read_table(path).to_pylist()

    assert str(tmp_path) not in str(stored)
    assert stored[0]["links"][0]["href"] == "./catalog.parquet"
    shutil.move(tmp_path, tmp_path.with_name("elsewhere"))
    moved = tmp_path.with_name("elsewhere") / "catalog.parquet"
    assert len(open_vector(moved)) == len(items)


def test_filtered_rows_do_not_gain_their_index_as_a_property(
    scene, label, tmp_path
) -> None:
    rows = stac.create_catalog(
        _stack_items(scene, label, tmp_path), collections=_collections()
    )

    (item,) = stac_table.to_items(rows[rows["collection"] == "label"])

    assert "__index_level_0__" not in item.properties


def test_a_table_rewritten_elsewhere_names_no_absolute_location(
    scene, label, tmp_path
) -> None:
    first = _save(
        _stack_items(scene, label, tmp_path / "a"),
        tmp_path / "a/catalog.parquet",
        collections=_collections(),
    )

    second = open_vector(first).gs.to_geoparquet(tmp_path / "a/copy.parquet")

    assert str(tmp_path) not in str(pq.read_table(second).to_pylist())


def test_a_plain_table_with_a_stac_version_column_reads_as_written(tmp_path) -> None:
    import geopandas as gpd
    from shapely.geometry import box

    plots = gpd.GeoDataFrame(
        {"stac_version": ["1.1.0"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )
    path = plots.gs.to_geoparquet(tmp_path / "plots.parquet")

    assert open_vector(path)["stac_version"].tolist() == ["1.1.0"]


def test_selected_columns_of_an_item_table_read_as_selected(
    scene, label, tmp_path
) -> None:
    path = _save(_stack_items(scene, label, tmp_path), tmp_path / "c.parquet")

    rows = open_vector(path, columns=["id", "geometry", "stac_version"])

    assert sorted(rows.columns) == ["geometry", "id", "stac_version"]


def test_reading_a_plain_table_does_not_import_stac(tmp_path) -> None:
    import subprocess
    import sys

    import geopandas as gpd
    from shapely.geometry import box

    plots = gpd.GeoDataFrame({"n": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    path = plots.gs.to_geoparquet(tmp_path / "plots.parquet")
    code = (
        "import sys; from geosave.geodata import open_vector; "
        f"open_vector({str(path)!r}); "
        "assert not {'geosave.geodata.stac', 'pystac', 'rustac', "
        "'stac_geoparquet'} & set(sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_unused_collection_definitions_are_not_written(scene, tmp_path) -> None:
    rows = stac_table.from_items(_cog_items(scene, tmp_path))
    path = rows.gs.to_geoparquet(
        tmp_path / "filtered.parquet", collections=_collections()
    )
    assert table.read_collections(path) == {}


def test_writing_leaves_the_rows_and_collections_unchanged(
    scene, label, tmp_path
) -> None:
    rows = stac.create_catalog(
        _stack_items(scene, label, tmp_path), collections=_collections()
    )
    given = _collections()
    before = (
        [item.to_dict() for item in stac_table.to_items(rows)],
        [each.to_dict() for each in given],
    )

    rows.gs.to_geoparquet(tmp_path / "items.parquet", collections=given)

    assert (
        [item.to_dict() for item in stac_table.to_items(rows)],
        [each.to_dict() for each in given],
    ) == before


def test_a_table_writes_to_a_upath(scene, tmp_path) -> None:
    from upath import UPath

    rows = stac_table.from_items(_cog_items(scene, tmp_path))

    path = rows.gs.to_geoparquet(UPath(tmp_path) / "items.parquet")

    assert open_vector(path)["id"].tolist() == rows["id"].tolist()


def test_a_path_without_a_suffix_refuses(scene, tmp_path) -> None:
    rows = stac_table.from_items(_cog_items(scene, tmp_path))

    with pytest.raises(ValueError):
        rows.gs.to_geoparquet(tmp_path / "catalog")


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
        f"{tmp_path}/optical_{{time:%Y%m%d}}_{{band}}.tif"
    )
    (label_scene,) = label.gs.to_cog(f"{tmp_path}/{{band}}.tif")
    rows = create_catalog(
        [
            *create_items(optical_scenes, id_format="optical_{start:%Y%m%d}"),
            *create_items(label_scene, id_format="label", datetime=when),
        ]
    )

    saved = rows.gs.to_geoparquet(tmp_path / "items.parquet")

    restored = stac_table.to_items(open_vector(saved))
    red = stac_io.open_raster([restored[0]]).red
    label = stac_io.open_raster([restored[-1]]).label
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].data_type == "uint16"
    assert red.attrs["units"] == "1"
    assert red.gs.packing.scale_factor == pytest.approx(1e-4)
    assert red.gs.spectral.common_name == "red"
    assert red.gs.spectral.center_wavelength == 0.665
    assert red.gs.nodata == optical.red.gs.nodata
    legend = attrs.Legend.model_validate(label.attrs)
    assert legend.classes == {0: "background", 1: "forest"}


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


def test_missing_collection_is_not_fabricated(tmp_path):
    rows = stac.create_catalog([entry()])
    with pytest.raises(ValueError, match="missing.*optical"):
        rows.gs.to_geoparquet(tmp_path / "missing.parquet")


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

    saved = stac_table.from_items([second, first]).gs.to_geoparquet(
        tmp_path / "items.parquet"
    )

    restored = list(stac_table_to_items(pq.read_table(saved)))
    assert [entry["id"] for entry in restored] == ["second", "first"]
    assert restored[0]["properties"]["new:property"] == 7
    assert restored[1]["assets"]["reflectance"]["bands"] == bands


def test_a_table_keeps_its_bbox_through_read_and_rewrite(tmp_path):
    entry, _ = create_native_item(tmp_path)
    first = stac_table.from_items([entry]).gs.to_geoparquet(
        tmp_path / "catalog.parquet"
    )
    second = open_vector(first).gs.to_geoparquet(tmp_path / "again.parquet")

    for path in (first, second):
        (restored,) = stac_table_to_items(pq.read_table(path))
        assert restored["bbox"] == pytest.approx(entry.bbox)


def test_an_edited_geometry_changes_the_bbox_written(tmp_path):
    entry, _ = create_native_item(tmp_path)
    rows = stac_table.from_items([entry])
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
    saved = stac_table.from_items([entry]).gs.to_geoparquet(home / "catalog.parquet")
    stored = pq.read_table(saved).column("assets")[0].as_py()
    assert stored["reflectance"]["href"] == "./scene.tif"
    assert stored["thumbnail"]["href"] == "https://example.com/thumb.jpg"

    moved = tmp_path / "moved"
    shutil.move(home, moved)

    href = open_vector(moved / "catalog.parquet").iloc[0].assets["reflectance"]["href"]
    assert href == str(moved / "scene.tif")
    assert Path(href).is_file()
