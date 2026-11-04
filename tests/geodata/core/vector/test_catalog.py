from copy import deepcopy
from datetime import UTC, datetime

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from pystac.extensions.raster import RasterExtension
from shapely.geometry import Point, box

from geosave.geodata import GeoAnchor, GeoCatalog
from geosave.geodata.stac.catalog import from_items
from tests.conftest import build_dataarray
from tests.geodata.conftest import build_raster, write_collection_cogs
from tests.geodata.core.vector.conftest import (
    create_item,
    create_scene_items,
    read_sample_ids,
)
from tests.geodata.stac.conftest import entry
from tests.geodata.conftest import create_file_items, create_test_collection


def test_catalog_tree_closes_every_scene_once(catalog, monkeypatch):
    from geosave.geodata.io import stac as raster_stac
    from geosave.geodata.stac.item import stamp_item_time

    closed = []

    def read(entry, **options):
        raster = stamp_item_time(build_raster()[["red"]], entry)
        scene = xr.DataTree.from_dict({"r10m": raster})
        scene.set_close(lambda: closed.append(entry.id))
        return scene

    monkeypatch.setattr(raster_stac, "open_scene", read)
    tree = catalog.gs.to_datatree()
    assert closed == []
    tree.close()
    tree.close()
    assert sorted(closed) == sorted(catalog["id"])


def test_column_mutation_keeps_vector_operations_and_checks_items_when_called():
    rows = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)
    accessor = rows.gs
    rows["id"] = ["scene"]
    rows["assets"] = [{}]
    rows["links"] = [[]]
    rows["stac_extensions"] = [[]]
    assert type(rows.gs) is GeoCatalog
    assert rows.gs is accessor
    rows.drop(columns="assets", inplace=True)
    with pytest.raises(ValueError, match="assets"):
        rows.gs.to_items()


@pytest.mark.parametrize(
    ("method", "suffix"),
    [
        ("to_geojson", ".geojson"),
        ("to_geopackage", ".gpkg"),
    ],
)
def test_other_vector_formats_refuse_catalog_metadata_loss(
    catalog, tmp_path, method, suffix
):
    from geosave import open_vector

    with pytest.raises(ValueError, match="to_geoparquet"):
        getattr(catalog.gs, method)(tmp_path / ("wrong" + suffix))
    path = catalog.gs.to_geoparquet(tmp_path / "items.parquet")
    restored = open_vector(path)
    assert set(restored.gs.collections) == {"s2", "s1", "label"}


def _ground(geometry, timespan: str | None = None) -> GeoAnchor:
    """Anchor fitting one longitude/latitude geometry exactly."""
    return GeoAnchor.from_geometry(geometry, crs="EPSG:4326", timespan=timespan)


def test_geographic_operations_require_a_crs() -> None:
    with pytest.raises(AttributeError, match="GeoDataFrame"):
        _ = pd.DataFrame({"a": [1]}).gs
    with pytest.raises(ValueError, match="CRS"):
        _ = gpd.GeoDataFrame(geometry=[Point(0, 0)]).gs.crs


def test_a_renamed_geometry_column_still_answers() -> None:
    frame = gpd.GeoDataFrame(
        {"name": ["a"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:4326"
    ).rename_geometry("footprint")

    assert frame.gs.footprint.geom.equals(box(0, 0, 2, 2))
    assert list(frame.gs.query(_ground(box(1, 1, 3, 3))).name) == ["a"]
    assert GeoCatalog.concat([frame, frame]).geometry.name == "geometry"


def test_concat_unions_columns_resets_index_and_keeps_duplicates() -> None:
    first = gpd.GeoDataFrame(
        [{"name": "first"}], geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    renamed = gpd.GeoDataFrame(
        [{"score": 4}], geometry=[Point(1, 1)], crs="EPSG:4326"
    ).rename_geometry("footprint")

    combined = GeoCatalog.concat([first, renamed, first])

    assert list(combined.index) == [0, 1, 2]
    assert list(combined.geometry) == [Point(0, 0), Point(1, 1), Point(0, 0)]
    assert list(combined.name.iloc[[0, 2]]) == ["first", "first"]
    assert combined.loc[1, "score"] == 4


def test_concat_requires_one_explicit_crs_plane() -> None:
    geographic = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    projected = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3857")

    with pytest.raises(ValueError, match="to_crs"):
        GeoCatalog.concat([geographic, projected])


def _places(names: list[str], geometries: list) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"name": names}, geometry=geometries, crs="EPSG:4326")


def test_query_returns_original_intersecting_rows_in_source_order() -> None:
    collection = _places(
        ["left", "middle", "right"],
        [box(0, 0, 2, 2), box(2, 0, 4, 2), box(8, 0, 10, 2)],
    )

    result = collection.gs.query(_ground(box(1, -1, 3, 3)))

    assert list(result.name) == ["left", "middle"]
    assert result.iloc[0].geometry.equals(collection.iloc[0].geometry)


def test_query_predicate_describes_each_row_relative_to_target() -> None:
    collection = _places(
        ["inside", "outside", "container"],
        [box(1, 1, 2, 2), box(5, 5, 6, 6), box(-1, -1, 4, 4)],
    )
    target = _ground(box(0, 0, 3, 3))

    assert list(collection.gs.query(target, predicate="within").name) == ["inside"]
    assert list(collection.gs.query(target, predicate="contains").name) == ["container"]


def test_query_reprojects_the_anchor_onto_the_table() -> None:
    collection = _places(["place"], [box(112.0, -8.2, 112.2, -8.0)])
    anchor = GeoAnchor.from_geometry(
        GeoAnchor.from_coordinates(-8.1, 112.1).geometry.to_crs("EPSG:32749")
    )

    result = collection.gs.query(anchor)

    assert anchor.crs.epsg != 4326
    assert list(result.name) == ["place"]
    assert result.gs.crs.to_epsg() == 4326


def test_query_takes_a_raster_by_its_anchor(raster) -> None:
    collection = gpd.GeoDataFrame(
        {"label": ["inside", "outside"]},
        geometry=[raster.gs.geobox.extent.geom, box(0, 0, 1, 1)],
        crs=raster.gs.crs,
    )

    assert list(collection.gs.query(raster).label) == ["inside"]


def test_query_empty_collection_preserves_schema() -> None:
    result = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326")).gs.query(
        _ground(box(0, 0, 1, 1))
    )

    assert len(result) == 0
    assert result.gs.crs.to_epsg() == 4326


def test_rasterize_values_back_onto_reference_grid(raster) -> None:
    flags = build_dataarray(
        np.array([[1, 1], [2, 0]], dtype="uint8"),
        raster.gs.geobox,
        dims=("y", "x"),
        nodata=0,
    )
    vector = flags.gs.vectorize(value_name="class_id")

    restored = vector.gs.rasterize(
        like=raster,
        column="class_id",
        fill=0,
        dtype="uint8",
    )

    np.testing.assert_array_equal(restored.values, flags.values)
    assert restored.gs.geobox == raster.gs.geobox
    assert restored.name == "class_id"


def test_the_footprint_bounds_carry_the_vectors_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 1)], crs="EPSG:32633")

    bounds = frame.gs.footprint.boundingbox

    assert tuple(bounds) == (0, 0, 2, 1)
    assert bounds.crs == "EPSG:32633"


@pytest.mark.parametrize(
    "timespan,expected",
    [
        ("2025-01-15", ["instant", "interval", "timeless"]),
        ("2025-01-16", ["timeless"]),
        (None, ["instant", "later", "interval", "timeless"]),
    ],
)
def test_query_combines_space_and_time_without_changing_rows(
    timespan, expected
) -> None:
    dated = _places(
        ["instant", "later", "interval", "timeless", "outside"],
        [box(0, 0, 1, 1)] * 4 + [box(5, 5, 6, 6)],
    )
    dated["datetime"] = ["2025-01-15T01:00:00+01:00", "2025-06-01", None, None, None]
    dated["start_datetime"] = [None, None, "2025-01-01", None, None]
    dated["end_datetime"] = [None, None, "2025-01-15", None, None]
    dated.index = [9, 2, 9, 4, 7]
    dated.attrs["producer"] = "survey"
    original = dated.copy(deep=True)

    result = dated.gs.query(_ground(box(-1, -1, 2, 2), timespan))

    pd.testing.assert_frame_equal(result, dated.loc[dated.name.isin(expected)])
    assert result.attrs == dated.attrs
    pd.testing.assert_frame_equal(dated, original)


def test_query_selects_by_time_alone_when_no_place_is_given() -> None:
    frame = gpd.GeoDataFrame(
        {
            "id": ["march", "april", "undated"],
            "datetime": ["2024-03-08", "2024-04-12", None],
        },
        geometry=[Point(0, 0)] * 3,
        crs="EPSG:4326",
    )

    assert frame.gs.query(when="2024-03")["id"].tolist() == ["march", "undated"]


def test_query_takes_a_plain_geometry_as_wgs84_and_a_separate_span() -> None:
    frame = gpd.GeoDataFrame(
        {"id": ["near", "far"], "datetime": ["2024-03-08", "2024-03-08"]},
        geometry=[Point(0, 0), Point(10, 10)],
        crs="EPSG:4326",
    )

    selected = frame.gs.query(box(-1, -1, 1, 1), when=("2024-03-01", "2024-03-31"))

    assert selected["id"].tolist() == ["near"]
    assert frame.gs.query(box(-1, -1, 1, 1), when="2023").empty


def test_query_reads_time_columns_that_carry_no_timezone() -> None:
    table = _dated()
    for column in ("datetime", "start_datetime", "end_datetime"):
        table[column] = table[column].dt.tz_localize(None)

    assert list(table.gs.query(_ground(box(-1, -1, 2, 2), "2025-01")).id) == ["jan"]


def _dated() -> gpd.GeoDataFrame:
    """Three rows on one ground: two spans and one lone instant."""
    table = gpd.GeoDataFrame(
        {
            "id": ["jan", "jun", "instant"],
            "datetime": [None, None, datetime(2025, 6, 12, tzinfo=UTC)],
            "start_datetime": [
                datetime(2025, 1, 10, tzinfo=UTC),
                datetime(2025, 6, 10, tzinfo=UTC),
                None,
            ],
            "end_datetime": [
                datetime(2025, 1, 15, tzinfo=UTC),
                datetime(2025, 6, 15, tzinfo=UTC),
                None,
            ],
        },
        geometry=[box(0, 0, 1, 1)] * 3,
        crs="EPSG:4326",
    )
    for column in ("datetime", "start_datetime", "end_datetime"):
        table[column] = pd.to_datetime(table[column], utc=True)
    return table


def test_query_filters_time_on_a_table_with_only_datetime() -> None:
    raster = build_raster(times=2)
    footprint = raster.gs.geobox.extent.to_crs("EPSG:4326").geom
    frame = gpd.GeoDataFrame(
        {
            "id": ["first", "second"],
            "datetime": pd.to_datetime(raster.time.values, utc=True),
        },
        geometry=[footprint, footprint],
        crs="EPSG:4326",
    )

    assert frame.gs.query(raster.isel(time=[0]))["id"].tolist() == ["first"]


def test_an_item_rows_anchor_is_its_grid_over_its_span(items, grid):
    anchor = items.iloc[1].gs.anchor

    assert anchor.geometry.geom.equals(items.iloc[1].geometry)
    assert str(anchor.timespan[0]) == "2024-03-08 00:00:00"


def test_a_timeless_item_reads_without_time_coordinates(grid, tmp_path):
    import numpy as np

    from tests.conftest import build_dataset
    from geosave.geodata.warnings import TimelessItemWarning
    from tests.geodata.conftest import item_table

    data = build_dataset(
        {"height": (("y", "x"), np.ones(grid.shape, dtype="float32"))}, grid
    )
    with pytest.warns(TimelessItemWarning):
        items = item_table(
            data.gs.to_cog(tmp_path, path_format="dem.tif"), collection="dem"
        )
    opened = items.iloc[0].gs.to_datatree().gs.to_dataset()
    assert "time" not in opened.coords
    assert "start_datetime" not in opened.coords
    assert "end_datetime" not in opened.coords
    assert isinstance(opened.height.data, da.Array)
    sample = items.gs.sample()[0].gs.to_datatree()["dem"].gs.to_dataset()
    assert sample.identical(opened)


def test_collection_objects_are_independent_of_stored_definitions(items):
    original = deepcopy(items.attrs["stac-geoparquet"]["collections"])
    collection = next(iter(items.gs.collections.values()))
    collection.description = "edited"
    collection.extra_fields["review"] = {"status": "changed"}
    assert items.attrs["stac-geoparquet"]["collections"] == original


def test_concat_keeps_independent_definitions_and_the_first_duplicate(items):
    table = items
    combined = table.gs.concat([table, table])
    assert set(combined.gs.collections) == set(table.gs.collections)
    name = next(iter(combined.attrs["stac-geoparquet"]["collections"]))
    combined.attrs["stac-geoparquet"]["collections"][name]["description"] = "changed"
    assert (
        table.attrs["stac-geoparquet"]["collections"][name]["description"] != "changed"
    )
    assert (
        table.gs.concat([table, combined]).attrs["stac-geoparquet"]["collections"][name]
        == (table.attrs["stac-geoparquet"]["collections"][name])
    )


def test_concat_feeds_multicollection_stacks(tmp_path):
    from tests.conftest import build_tree

    ds = build_raster(times=1)
    layers = build_tree({"optical": ds, "label": ds[["red"]].rename(red="label")})
    scenes = write_collection_cogs(
        layers,
        tmp_path,
        {"optical": "optical_{time:%Y%m%d}.tif", "label": "label_{time:%Y%m%d}.tif"},
    )
    catalogs = []
    for name, paths in scenes.items():
        collection = create_test_collection(name, description=name)
        catalogs.append(
            from_items(
                create_file_items(paths, collection=name), collections=[collection]
            )
        )
    combined = GeoCatalog.concat(catalogs)
    samples = combined.gs.sample()
    assert len(samples) == 1
    loaded = samples[0].gs.to_datatree()
    assert set(loaded.children) == {"optical", "label"}
    assert loaded["optical"].gs.to_dataset().red.chunks is not None


def test_concat_keeps_the_first_definition_even_after_filtering_rows():
    optical = create_test_collection("optical", description="Old optical")
    label = create_test_collection("label", description="Labels")
    rows = from_items([entry(), entry("label")], collections=[optical, label])
    optical.description = "New optical"
    fresh = from_items([entry()], collections=[optical])
    combined = GeoCatalog.concat([rows[rows.collection == "label"], fresh])
    assert combined.gs.collections["optical"].description == "Old optical"


def test_sample_groups_footprints_covering_one_another(catalog):
    samples = catalog.gs.sample()

    assert read_sample_ids(samples) == [
        ["s2_A_jan", "s2_A_feb", "s1_A", "label_A"],
        ["s2_B_jan"],
    ]


def test_sample_shape_is_the_footprint_its_items_share(catalog):
    samples = catalog.gs.sample()

    assert samples.footprints.geometry.iloc[0].equals(box(0, 0, 1, 1))


def test_sample_keeps_footprints_that_only_touch_apart(catalog):
    touching = catalog.copy()
    touching.loc[touching["id"] == "s2_B_jan", "geometry"] = box(1.02, 0, 2, 1)

    assert len(read_sample_ids(touching.gs.sample())) == 2


def test_sample_lists_a_footprint_in_every_sample_it_covers(catalog):
    swath = catalog[catalog["id"] == "s1_A"].copy()
    swath["id"] = "s1_swath"
    swath["geometry"] = box(0, 0, 4, 1)
    wide = GeoCatalog.concat([catalog, swath])

    assert read_sample_ids(wide.gs.sample()) == [
        ["s2_A_jan", "s2_A_feb", "s1_A", "label_A", "s1_swath"],
        ["s2_B_jan", "s1_swath"],
    ]


def test_sample_overlap_sets_how_much_of_a_footprint_an_item_shares(catalog):
    shifted = catalog.copy()
    shifted.loc[shifted["id"] == "s1_A", "geometry"] = box(0.5, 0, 1.5, 1)

    assert len(read_sample_ids(shifted.gs.sample(overlap=0.9))) == 3
    assert len(read_sample_ids(shifted.gs.sample(overlap=0.5))) == 2


def test_an_item_joins_every_footprint_it_shares_enough_of():
    entries = []
    for item_id, left in (("a", 0), ("b", 0.3), ("c", 0.6)):
        entries.append(create_item(item_id, "s2", box(left, 0, left + 1, 1)))
    shifted = from_items(entries)

    assert read_sample_ids(shifted.gs.sample(overlap=0.5)) == [["a", "b"], ["b", "c"]]


def test_a_sample_keeps_the_catalog_collections(catalog):
    rows = catalog.gs.sample()[1]

    assert (
        rows.attrs["stac-geoparquet"]["collections"]
        == catalog.attrs["stac-geoparquet"]["collections"]
    )
    rows.attrs["stac-geoparquet"]["collections"]["s2"]["description"] = "changed"
    assert catalog.attrs["stac-geoparquet"]["collections"]["s2"]["description"] == "s2"


def test_rows_become_the_items_they_came_from(scene, tmp_path) -> None:
    items = create_scene_items(scene, tmp_path)
    restored = from_items(items).gs.to_items()
    assert [item.id for item in restored] == [item.id for item in items]
    assert restored[0].bbox == pytest.approx(items[0].bbox)
    assert restored[0].assets["red"].href == items[0].assets["red"].href
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].nodata == 0


def test_filtered_rows_keep_only_the_assets_they_have(scene, label, tmp_path) -> None:
    rows = from_items(
        create_scene_items(scene, tmp_path)
        + create_scene_items(label, tmp_path, "label")
    )
    labels = rows[rows["collection"] == "label"].gs.to_items()
    assert [list(item.assets) for item in labels] == [["label"]]


def test_concatenated_tables_become_the_items_they_came_from(
    scene, label, tmp_path
) -> None:
    optical_items = create_scene_items(scene, tmp_path / "optical")
    label_items = create_scene_items(label, tmp_path / "labels", "label")
    items = optical_items + label_items
    rows = GeoCatalog.concat([from_items(optical_items), from_items(label_items)])
    restored = rows.gs.to_items()
    assert [item.id for item in restored] == [item.id for item in items]
    assert [sorted(item.assets) for item in restored] == [
        sorted(item.assets) for item in items
    ]


def test_empty_table_converts_to_no_items():
    assert from_items([]).gs.to_items() == []


def test_items_of_a_projected_table_state_their_footprint_in_wgs84(catalog):
    original = catalog.gs.to_items()[0]

    projected = catalog.to_crs("EPSG:3857").gs.to_items()[0]

    assert projected.bbox == pytest.approx(original.bbox)
    assert projected.geometry["coordinates"][0][0] == pytest.approx(
        original.geometry["coordinates"][0][0]
    )


def test_sample_refuses_items_without_area():
    from shapely.geometry import Point

    entries = [
        create_item("pt", "s2", Point(0.5, 0.5)),
        create_item("t1", "s2", box(0, 0, 1, 1)),
        create_item("t2", "s2", box(5, 5, 6, 6)),
    ]

    with pytest.raises(ValueError, match="pt"):
        from_items(entries).gs.sample()


def test_a_sample_shape_is_an_item_when_its_items_only_touch():
    # Each shifted Item covers half the tile; the two meet along x=0.5 only.
    entries = [
        create_item("tile", "s2", box(0, 0, 1, 1)),
        create_item("left", "s2", box(-0.5, 0, 0.5, 1)),
        create_item("right", "s2", box(0.5, 0, 1.5, 1)),
    ]

    samples = from_items(entries).gs.sample(overlap=0.5)

    assert read_sample_ids(samples) == [["tile", "left", "right"]]
    assert samples.footprints.geometry.iloc[0].area > 0


def test_rows_stack_along_time_per_collection_and_grid(tiles):
    rows = tiles[tiles["id"].str.startswith("s2_300000")]

    tree = rows.gs.to_datatree()

    assert list(tree.gs.rasters) == ["s2/r10m"]
    assert tree["s2/r10m"].sizes["time"] == 5
    assert tree["s2"].gs.to_dataset().gs.stac_collection.id == "s2"
