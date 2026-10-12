"""stac-geoparquet: Items to rows, to one file, and back to pixels."""

from __future__ import annotations

import shutil

import dask
import numpy as np
import pyarrow.parquet as pq
import pytest
from pystac.extensions.raster import RasterExtension

from geosave.geodata import GeoVector, cuts, open_vector, stac, stack
from geosave.geodata.core.catalog import GeoCatalog
from geosave.geodata.io.vector import stac_geoparquet as table
from tests.geodata.conftest import cog_template, read_items, stack_templates


def _refuse(*args, **kwargs):
    raise AssertionError("the table computed pixels")


def _per_band(template: str) -> str:
    """Write one file per band, so each asset is one band named by its key."""
    return template.replace(".tif", "_{band}.tif")


def _cog_items(scene, root, name="forest", **options):
    template = _per_band(cog_template(root, name, dated="time" in scene.dims))
    return table.to_items(stac.create_items(scene.gs.to_cog(template), **options))


def _stack_items(scene, label, root, name="s0"):
    sample = stack({"optical": scene, "label": label})
    templates = stack_templates(sample, root / name)
    saved = sample.gs.to_cog({group: _per_band(t) for group, t in templates.items()})
    return table.to_items(stac.create_items(saved, id=name))


def _save(items, path, **options):
    return table.from_items(items).gs.to_geoparquet(path, **options)


def _collections():
    return [
        stac.create_collection(name, description=f"{name} rows", license="CC-BY-4.0")
        for name in ("optical", "label")
    ]


def test_items_become_rows_in_wgs84(scene, tmp_path) -> None:
    rows = table.from_items(_cog_items(scene, tmp_path))

    assert rows["id"].tolist() == ["forest_20250601T000000", "forest_20250611T000000"]
    assert rows.crs.to_epsg() == 4326
    assert "bbox" not in rows


def test_no_items_give_an_empty_catalog() -> None:
    rows = table.from_items([])

    assert rows.empty
    assert rows.crs.to_epsg() == 4326
    assert type(rows.gs) is GeoCatalog


def test_rows_become_the_items_they_came_from(scene, tmp_path) -> None:
    items = _cog_items(scene, tmp_path)

    restored = table.to_items(table.from_items(items))

    assert [item.id for item in restored] == [item.id for item in items]
    assert restored[0].bbox == pytest.approx(items[0].bbox)
    assert restored[0].assets["red"].href == items[0].assets["red"].href
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].nodata == 0


def test_filtered_rows_keep_only_the_assets_they_have(scene, label, tmp_path) -> None:
    rows = table.from_items(_stack_items(scene, label, tmp_path))

    labels = table.to_items(rows[rows["collection"] == "label"])

    assert [list(item.assets) for item in labels] == [["label"]]


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


def test_each_group_of_a_stack_loads_alone(scene, label, tmp_path) -> None:
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

    restored = read_items(table.from_items([item]))

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

    stored = table.get_collections(path)

    assert sorted(stored) == ["label", "optical"]
    assert stored["optical"].license == "CC-BY-4.0"
    assert stored["optical"].extent.temporal.to_dict()["interval"] == [
        ["2025-06-01T00:00:00Z", "2025-06-11T00:00:00Z"]
    ]
    # The Collections handed in keep the open extent they came with.
    assert given[0].extent.spatial.bboxes == [[-180.0, -90.0, 180.0, 90.0]]


def test_items_of_a_stored_collection_link_to_it(scene, label, tmp_path) -> None:
    path = _save(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=_collections(),
    )

    (item, *_) = table.to_items(open_vector(path))

    assert [link.rel for link in item.links] == ["collection"]


def test_a_row_naming_an_absent_collection_is_described(scene, label, tmp_path) -> None:
    path = _save(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=_collections()[:1],
    )

    stored = table.get_collections(path)
    assert sorted(stored) == ["label", "optical"]
    assert stored["label"].description == "label"


def test_a_table_without_collections_reads_none(scene, tmp_path) -> None:
    path = _save(_cog_items(scene, tmp_path), tmp_path / "items.parquet")

    assert table.get_collections(path) == {}


def test_a_batch_concatenated_onto_a_table_reads_back_whole(
    scene, label, tmp_path
) -> None:
    path = _save(
        _stack_items(scene, label, tmp_path, "s0"),
        tmp_path / "items.parquet",
        collections=_collections(),
    )
    more = table.from_items(_stack_items(scene, label, tmp_path, "s1"))

    grown = GeoVector.concat([open_vector(path), more])
    grown.gs.to_geoparquet(
        path,
        collections=table.get_collections(path).values(),
        overwrite=True,
    )

    rows = open_vector(path)
    assert sorted({id.split("/")[0] for id in rows["id"]}) == ["s0", "s1"]
    assert sorted(table.get_collections(path)) == ["label", "optical"]
    restored = read_items(
        rows[rows["id"].str.startswith("s1/") & (rows["collection"] == "optical")]
    )
    np.testing.assert_array_equal(restored.red.values, scene.red.values)
    restored.close()


def test_a_rewrite_without_collections_describes_them_minimally(
    scene, label, tmp_path
) -> None:
    path = _save(
        _stack_items(scene, label, tmp_path),
        tmp_path / "items.parquet",
        collections=_collections(),
    )

    open_vector(path).gs.to_geoparquet(path, overwrite=True)

    stored = table.get_collections(path)
    assert sorted(stored) == ["label", "optical"]
    assert stored["optical"].description == "optical"


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
    rows = table.from_items(_stack_items(scene, label, tmp_path))

    path = rows.gs.to_geoparquet(
        tmp_path / "catalog.parquet", collections=_collections()
    )

    assert sorted(table.get_collections(path)) == ["label", "optical"]
    assert open_vector(path)["id"].tolist() == rows["id"].tolist()
    with pytest.raises(FileExistsError):
        rows.gs.to_geoparquet(path)


def test_a_vector_that_is_not_an_item_table_does_not_write_as_one(tmp_path) -> None:
    from shapely.geometry import Point

    plain = GeoVector.from_geometry(Point(0, 0), properties={"id": "a"})

    with pytest.raises(ValueError, match="item table"):
        table.write(plain, tmp_path / "plain.parquet")
    written = plain.gs.to_geoparquet(tmp_path / "plain.parquet")
    assert table.get_collections(written) == {}


def test_open_vector_reads_an_item_table_with_hrefs_that_open(
    scene, label, tmp_path
) -> None:
    path = _save(_stack_items(scene, label, tmp_path), tmp_path / "catalog.parquet")
    shutil.move(tmp_path, tmp_path.with_name("moved"))
    moved = tmp_path.with_name("moved") / path.name

    rows = open_vector(moved)

    sample = cuts.stacks(rows, on="label").iloc[0].gs.to_stack()
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
    rows = table.from_items(_stack_items(scene, label, tmp_path))

    (item,) = table.to_items(rows[rows["collection"] == "label"])

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


def test_a_given_collection_without_rows_refuses(scene, tmp_path) -> None:
    rows = table.from_items(_cog_items(scene, tmp_path))

    with pytest.raises(ValueError, match=r"\['optical'\]"):
        rows.gs.to_geoparquet(tmp_path / "items.parquet", collections=_collections()[:1])


def test_writing_leaves_the_rows_and_collections_unchanged(
    scene, label, tmp_path
) -> None:
    rows = table.from_items(_stack_items(scene, label, tmp_path))
    given = _collections()
    before = (
        [item.to_dict() for item in table.to_items(rows)],
        [each.to_dict() for each in given],
    )

    rows.gs.to_geoparquet(tmp_path / "items.parquet", collections=given)

    assert (
        [item.to_dict() for item in table.to_items(rows)],
        [each.to_dict() for each in given],
    ) == before


def test_a_table_writes_to_a_upath(scene, tmp_path) -> None:
    from upath import UPath

    rows = table.from_items(_cog_items(scene, tmp_path))

    path = rows.gs.to_geoparquet(UPath(tmp_path) / "items.parquet")

    assert open_vector(path)["id"].tolist() == rows["id"].tolist()


def test_a_path_without_a_suffix_refuses(scene, tmp_path) -> None:
    rows = table.from_items(_cog_items(scene, tmp_path))

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
