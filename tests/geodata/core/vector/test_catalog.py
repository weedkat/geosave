from __future__ import annotations

from tests.conftest import write_cogs
from copy import deepcopy

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pystac.extensions.raster import RasterExtension
from shapely.geometry import Point, box

from geosave.geodata import GeoCatalog
from geosave.geodata.stac.catalog import create_catalog
from tests.conftest import build_dataarray
from tests.geodata.conftest import build_raster, write_collection_cogs
from tests.geodata.core.vector.conftest import (
    create_item,
    create_scene_items,
    read_sample_ids,
)
from tests.geodata.stac.conftest import entry
from tests.geodata.conftest import create_file_items, create_test_collection

from shapely.geometry import Polygon
from odc.geo.geobox import GeoBox
from geosave.geodata.utils.geometry import to_shapely


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


def test_an_item_rows_region_is_its_grid_over_its_span(items, grid):
    region = items.iloc[1].gs.region

    assert region.extent.geom.equals(items.iloc[1].geometry)
    assert str(region.timespan[0]) == "2024-03-08 00:00:00"


def test_a_timeless_item_reads_without_time_coordinates(grid, tmp_path):
    import numpy as np

    from tests.conftest import build_dataset
    from geosave.geodata.stac.item import TIMELESS
    from tests.geodata.conftest import item_table

    data = build_dataset(
        {"height": (("y", "x"), np.ones(grid.shape, dtype="float32"))}, grid
    )
    items = item_table(
        write_cogs(data, tmp_path, format_paths="dem.tif"),
        collection="dem",
        datetime=TIMELESS,
    )
    opened = items.iloc[0].gs.to_datatree()["dem"].gs.combine_stac()
    assert "time" not in opened.coords
    assert "start_datetime" not in opened.coords
    assert "end_datetime" not in opened.coords
    assert isinstance(opened.height.data, da.Array)
    sample = items.gs.sample()[0].gs.to_datatree()["dem"].gs.combine_stac()
    assert sample.identical(opened)


def test_collection_objects_are_independent_of_stored_definitions(items):
    original = deepcopy(items.attrs["stac-geoparquet"]["collections"])
    collection = next(iter(items.gs.collections.values()))
    collection.description = "edited"
    collection.extra_fields["review"] = {"status": "changed"}
    assert items.attrs["stac-geoparquet"]["collections"] == original


def test_concat_copies_definitions_and_rejects_conflicting_duplicates(items):
    table = items
    combined = table.gs.concat([table, table])
    assert set(combined.gs.collections) == set(table.gs.collections)
    name = next(iter(combined.attrs["stac-geoparquet"]["collections"]))
    combined.attrs["stac-geoparquet"]["collections"][name]["description"] = "changed"
    assert (
        table.attrs["stac-geoparquet"]["collections"][name]["description"] != "changed"
    )
    with pytest.raises(ValueError, match="description"):
        table.gs.concat([table, combined])


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
            create_catalog(
                create_file_items(paths, collection=name), collections=[collection]
            )
        )
    combined = GeoCatalog.concat(catalogs)
    samples = combined.gs.sample()
    assert len(samples) == 1
    loaded = samples[0].gs.to_datatree()
    assert set(loaded.children) == {"optical", "label"}
    assert loaded["optical"].gs.combine_stac().red.chunks is not None


def test_concat_checks_definitions_retained_after_filtering_rows():
    optical = create_test_collection("optical", description="Old optical")
    label = create_test_collection("label", description="Labels")
    rows = create_catalog([entry(), entry("label")], collections=[optical, label])
    optical.description = "New optical"
    fresh = create_catalog([entry()], collections=[optical])
    with pytest.raises(ValueError, match="optical.*description"):
        GeoCatalog.concat([rows[rows.collection == "label"], fresh])


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
    shifted = create_catalog(entries)

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
    restored = create_catalog(items).gs.to_items()
    assert [item.id for item in restored] == [item.id for item in items]
    assert restored[0].bbox == pytest.approx(items[0].bbox)
    assert restored[0].assets["red"].href == items[0].assets["red"].href
    assert RasterExtension.ext(restored[0].assets["red"]).bands[0].nodata == 0


def test_filtered_rows_keep_only_the_assets_they_have(scene, label, tmp_path) -> None:
    rows = create_catalog(
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
    rows = GeoCatalog.concat(
        [create_catalog(optical_items), create_catalog(label_items)]
    )
    restored = rows.gs.to_items()
    assert [item.id for item in restored] == [item.id for item in items]
    assert [sorted(item.assets) for item in restored] == [
        sorted(item.assets) for item in items
    ]


def test_empty_table_converts_to_no_items():
    assert create_catalog([]).gs.to_items() == []


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
        create_catalog(entries).gs.sample()


def test_a_sample_shape_is_an_item_when_its_items_only_touch():
    # Each shifted Item covers half the tile; the two meet along x=0.5 only.
    entries = [
        create_item("tile", "s2", box(0, 0, 1, 1)),
        create_item("left", "s2", box(-0.5, 0, 0.5, 1)),
        create_item("right", "s2", box(0.5, 0, 1.5, 1)),
    ]

    samples = create_catalog(entries).gs.sample(overlap=0.5)

    assert read_sample_ids(samples) == [["tile", "left", "right"]]
    assert samples.footprints.geometry.iloc[0].area > 0


def test_rows_stack_along_time_per_collection_and_grid(tiles):
    rows = tiles[tiles["id"].str.startswith("s2_300000")]

    tree = rows.gs.to_datatree()
    scene = tree["s2"].gs.combine_stac()

    assert len(tree["s2"].children) == 5
    assert scene.sizes["time"] == 5
    assert scene.gs.stac_collection.id == "s2"


def test_validation_can_check_crs_without_requiring_geometry():
    from geosave.geodata.errors import VectorValidationError

    rows = gpd.GeoDataFrame(geometry=[None], crs="EPSG:4326")
    assert rows.gs.validate(geometry=False, crs=True) is None
    with pytest.raises(VectorValidationError, match="position 0.*missing"):
        rows.gs.validate()
    with pytest.raises(VectorValidationError, match="set_crs"):
        gpd.GeoDataFrame(geometry=[Point(0, 0)]).gs.validate(crs=True)


def test_sampling_reports_invalid_geometry_without_an_item_id():
    from geosave.geodata.errors import VectorValidationError

    rows = gpd.GeoDataFrame(geometry=[Point(0, 0)], index=["site"], crs="EPSG:4326")
    with pytest.raises(VectorValidationError, match="position 0.*area"):
        rows.gs.sample()


@pytest.mark.parametrize(
    "start, end, message",
    [
        ("2025-01-01", None, "both start_datetime and end_datetime"),
        ("2025-02-01", "2025-01-01", "end_datetime precedes"),
        ("2025-02-01", "2025-02-28", "datetime lies outside"),
    ],
)
def test_temporal_validation_is_independent_of_geometry(start, end, message):
    from geosave.geodata.errors import VectorValidationError

    rows = gpd.GeoDataFrame(
        {
            "id": ["invalid"],
            "datetime": ["2025-01-15"],
            "start_datetime": [start],
            "end_datetime": [end],
        },
        geometry=[None],
        index=[9],
    )

    with pytest.raises(VectorValidationError, match=f"position 0.*invalid.*{message}"):
        rows.gs.validate(geometry=False, temporal=True)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"collections": []}, "collections"),
        ({"collections": "s2"}, "collections"),
        ({"collections": ["s2", "s2"]}, "repeat"),
        ({"collections": ["s2"], "missing": "invalid"}, "missing"),
        ({"collections": ["s2"], "interval": "invalid"}, "interval"),
        ({"collections": ["s2"], "step": 0}, "step"),
        ({"collections": ["s2"], "window": 1.5}, "window"),
    ],
)
def test_sampling_rejects_invalid_options_before_planning(options, message):
    rows = gpd.GeoDataFrame(
        {"collection": ["s2"], "datetime": [pd.Timestamp("2025-01-01")]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match=message):
        rows.gs.sample(**options)


def test_concat_rejects_mixing_item_tables_and_features(catalog):
    features = gpd.GeoDataFrame(
        {"name": ["place"]}, geometry=[Point(0, 0)], crs=catalog.crs
    )

    with pytest.raises(ValueError, match="STAC Item tables with ordinary features"):
        GeoCatalog.concat([catalog, features])


@pytest.mark.parametrize("record", [False, True])
def test_validation_distinguishes_plain_features_from_item_records(record):
    from geosave.geodata.errors import VectorValidationError

    rows = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)
    value = rows.iloc[0] if record else rows
    value.gs.validate(stac_item=False)
    with pytest.raises(VectorValidationError, match="STAC Item operations need"):
        value.gs.validate(stac_item=True)

    rows["id"] = ["scene"]
    rows["assets"] = [{}]
    rows["links"] = [[]]
    rows["stac_extensions"] = [[]]
    value = rows.iloc[0] if record else rows
    value.gs.validate()
    value.gs.validate(stac_item=True)
    with pytest.raises(VectorValidationError, match="ordinary features required"):
        value.gs.validate(stac_item=False)


@pytest.mark.parametrize("record", [False, True])
def test_plain_feature_validation_rejects_retained_stac_metadata(record):
    from geosave.geodata.errors import VectorValidationError

    rows = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)
    rows.attrs["stac-geoparquet"] = {"collections": {}}
    value = rows.iloc[0] if record else rows

    value.gs.validate()
    with pytest.raises(VectorValidationError, match="to_geoparquet"):
        value.gs.validate(stac_item=False)


def _sliced_geographic_scene():
    """Build a raster whose sliced coordinates drift from its geobox's own."""

    from tests.conftest import build_dataset

    step = 1 / 3600
    grid = GeoBox.from_bbox(
        (13.0, 45.0, 13.0 + 40 * step, 45.0 + 40 * step),
        crs="EPSG:4326",
        resolution=step,
    )
    whole = build_dataset(
        {"red": (("y", "x"), np.ones(grid.shape, "uint16"))}, grid, nodata=0
    )
    return whole.isel(x=slice(5, 25), y=slice(5, 25))


def test_rasterize_lands_on_the_target_rasters_own_coordinates() -> None:
    import shapely
    import xarray as xr

    scene = _sliced_geographic_scene()
    left, bottom, right, top = scene.gs.geobox.boundingbox.bbox
    plot = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, (left + right) / 2, top)],
        crs="EPSG:4326",
    )

    burned = plot.gs.rasterize(scene)

    xr.align(scene, burned, join="exact")
    masked = scene.gs.mask(burned)
    assert (masked.red.values == 0).any() and (masked.red.values == 1).any()


def test_a_presence_mask_refuses_a_truthy_fill() -> None:
    import shapely

    scene = _sliced_geographic_scene()
    left, bottom, right, top = scene.gs.geobox.boundingbox.bbox
    plot = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, right, top)], crs="EPSG:4326"
    )

    with pytest.raises(ValueError, match="presence mask"):
        plot.gs.rasterize(scene, fill=1)


def test_rasterizing_onto_an_explicit_grid_is_undated(grid):
    plot = gpd.GeoDataFrame(geometry=[grid.extent.geom], crs=grid.crs)
    burned = plot.gs.rasterize(grid)
    assert burned.gs.geobox == grid
    assert "time" not in burned.coords


@pytest.mark.parametrize("problem", ["null", "empty"])
def test_rasterize_refuses_a_missing_geometry(raster, problem: str) -> None:
    geometry = None if problem == "null" else Polygon()
    vector = gpd.GeoDataFrame(geometry=[geometry], crs=raster.gs.crs)
    with pytest.raises(ValueError, match="position 0.*(missing|empty)"):
        vector.gs.rasterize(raster)


def test_rasterize_refuses_a_geometry_naming_no_ground(raster) -> None:
    bowtie = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])], crs=raster.gs.crs
    )
    with pytest.raises(ValueError, match="position 0.*invalid"):
        bowtie.gs.rasterize(raster)


def test_rasterize_without_column_returns_boolean_presence(raster) -> None:
    vector = gpd.GeoDataFrame(
        geometry=[to_shapely(raster.gs.geobox.extent)], crs=raster.gs.geobox.extent.crs
    )
    mask = vector.gs.rasterize(like=raster)
    assert mask.dtype == np.dtype("bool")
    assert mask.values.all()


def test_rasterize_later_rows_win_on_overlap(raster) -> None:
    bounds = raster.gs.geobox.extent.geom
    vector = gpd.GeoDataFrame(
        {"class_id": [1, 2]}, geometry=[bounds, bounds], crs=raster.gs.crs
    )
    result = vector.gs.rasterize(like=raster, column="class_id", dtype="uint8")
    assert (result.values == 2).all()


def test_rasterize_rejects_values_outside_dtype(raster) -> None:
    vector = gpd.GeoDataFrame(
        [{"class_id": 300}],
        geometry=[to_shapely(raster.gs.geobox.extent)],
        crs=raster.gs.geobox.extent.crs,
    )
    with pytest.raises(ValueError, match="uint8"):
        vector.gs.rasterize(like=raster, column="class_id", dtype="uint8")


def test_rasterize_rejects_non_numeric_property(raster) -> None:
    vector = gpd.GeoDataFrame(
        [{"label": "forest"}],
        geometry=[to_shapely(raster.gs.geobox.extent)],
        crs=raster.gs.geobox.extent.crs,
    )
    with pytest.raises(ValueError, match="numeric raster dtype"):
        vector.gs.rasterize(like=raster, column="label")


def test_rasterize_empty_vector_returns_only_fill(raster) -> None:
    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=raster.gs.crs))
    result = empty.gs.rasterize(like=raster, fill=0)
    assert not result.values.any()
