from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, Polygon, box

from geosave.geodata import (
    GeoAnchor,
    GeoVector,
)
from geosave.geodata.core.raster.create import array
from geosave.geodata.core.vector.vector import SpatialPredicate
from geosave.geodata.transform.vector import rasterize


def _ground(geometry, timespan: str | None = None) -> GeoAnchor:
    """Anchor fitting one longitude/latitude geometry exactly."""
    return GeoAnchor.from_geometry(
        geometry, shape=4, crs="EPSG:4326", anchor="floating", timespan=timespan
    )


def test_geographic_operations_require_a_crs() -> None:
    with pytest.raises(AttributeError, match="GeoDataFrame"):
        _ = pd.DataFrame({"a": [1]}).gs
    with pytest.raises(ValueError, match="CRS"):
        _ = gpd.GeoDataFrame(geometry=[Point(0, 0)]).gs.crs


def test_the_accessor_survives_native_selection_and_concatenation() -> None:
    frame = gpd.GeoDataFrame(
        {"name": ["a", "b"]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
    )

    assert frame[frame.name == "b"].gs.crs.to_epsg() == 4326
    assert len(pd.concat([frame, frame]).gs.query(_ground(box(-1, -1, 2, 2)))) == 4


def test_a_renamed_geometry_column_still_answers() -> None:
    frame = gpd.GeoDataFrame(
        {"name": ["a"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:4326"
    ).rename_geometry("footprint")

    assert frame.gs.footprint.geom.equals(box(0, 0, 2, 2))
    assert list(frame.gs.query(_ground(box(1, 1, 3, 3))).name) == ["a"]
    assert GeoVector.concat([frame, frame]).geometry.name == "geometry"


def test_geometry_properties_become_columns() -> None:
    vector = GeoVector.from_geometry(
        Point(112.15, -8.05), properties={"place_name": "Malang", "split": "train"}
    )

    assert vector.loc[0, "place_name"] == "Malang"
    assert vector.loc[0, "split"] == "train"


def test_concat_unions_columns_resets_index_and_keeps_duplicates() -> None:
    first = GeoVector.from_geometry(Point(0, 0), properties={"name": "first"})
    renamed = GeoVector.from_geometry(
        Point(1, 1), properties={"score": 4}
    ).rename_geometry("footprint")

    combined = GeoVector.concat([first, renamed, first])

    assert list(combined.index) == [0, 1, 2]
    assert list(combined.geometry) == [Point(0, 0), Point(1, 1), Point(0, 0)]
    assert list(combined.name.iloc[[0, 2]]) == ["first", "first"]
    assert combined.loc[1, "score"] == 4


def test_concat_requires_one_explicit_crs_plane() -> None:
    geographic = GeoVector.from_geometry(Point(0, 0))
    projected = GeoVector.from_geometry(Point(0, 0), crs="EPSG:3857")

    with pytest.raises(ValueError, match="to_crs"):
        GeoVector.concat([geographic, projected])


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
    anchor = GeoAnchor.from_coordinates(-8.1, 112.1, shape=4, resolution=100)

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


def test_query_rejects_unknown_predicate() -> None:
    vector = GeoVector.from_geometry(Point(0, 0))

    with pytest.raises(ValueError, match="predicate"):
        vector.gs.query(
            _ground(box(-1, -1, 1, 1)), predicate=cast("SpatialPredicate", "touch-ish")
        )


@pytest.mark.parametrize("problem", ["null", "empty"])
def test_rasterize_refuses_a_missing_geometry(raster, problem: str) -> None:
    geometry = None if problem == "null" else Polygon()
    vector = gpd.GeoDataFrame(geometry=[geometry], crs=raster.gs.crs)

    with pytest.raises(ValueError, match="null, empty, or invalid"):
        vector.gs.rasterize(raster)


def test_rasterize_refuses_a_geometry_naming_no_ground(raster) -> None:
    bowtie = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])], crs=raster.gs.crs
    )

    with pytest.raises(ValueError, match="invalid geometry"):
        bowtie.gs.rasterize(raster)


def test_rasterize_values_back_onto_reference_grid(raster) -> None:
    flags = array(
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


def test_raster_method_delegates_to_vector_transform(raster) -> None:
    vector = GeoVector.from_geometry(raster.gs.anchor.geobox.extent)

    direct = rasterize(vector, raster)
    through_method = vector.gs.rasterize(raster)

    xr.testing.assert_identical(direct, through_method)


def test_rasterize_without_column_returns_boolean_presence(raster) -> None:
    vector = GeoVector.from_geometry(raster.gs.geobox.extent)

    mask = vector.gs.rasterize(like=raster)

    assert mask.dtype == np.dtype("bool")
    assert mask.values.all()


def test_rasterize_later_rows_win_on_overlap(raster) -> None:
    bounds = raster.gs.geobox.extent.geom
    vector = gpd.GeoDataFrame(
        {"class_id": [1, 2]},
        geometry=[bounds, bounds],
        crs=raster.gs.crs,
    )

    result = vector.gs.rasterize(like=raster, column="class_id", dtype="uint8")

    assert (result.values == 2).all()


def test_rasterize_rejects_values_outside_dtype(raster) -> None:
    vector = GeoVector.from_geometry(
        raster.gs.geobox.extent, properties={"class_id": 300}
    )

    with pytest.raises(ValueError, match="uint8"):
        vector.gs.rasterize(like=raster, column="class_id", dtype="uint8")


def test_rasterize_rejects_non_numeric_property(raster) -> None:
    vector = GeoVector.from_geometry(
        raster.gs.geobox.extent, properties={"label": "forest"}
    )

    with pytest.raises(ValueError, match="numeric raster dtype"):
        vector.gs.rasterize(like=raster, column="label")


def test_rasterize_empty_vector_returns_only_fill(raster) -> None:
    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=raster.gs.crs))

    result = empty.gs.rasterize(like=raster, fill=0)

    assert not result.values.any()


REMOVED = (
    "normalize",
    "timespan",
    "to_anchor",
    "to_items",
    "from_items",
    "upsert",
    "to_raster",
    "raster_ids",
    "stack_ids",
    "empty",
    "vectorize",
)


@pytest.mark.parametrize("name", REMOVED)
def test_geovector_is_spatial_only(name: str) -> None:
    assert not hasattr(GeoVector, name)


def test_a_geometry_factory_adds_no_time_column() -> None:
    frame = GeoVector.from_geometry(Point(0, 0), properties={"name": "a"})

    assert list(frame.columns) == ["name", "geometry"]


def test_the_footprint_bounds_carry_the_vectors_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 1)], crs="EPSG:32633")

    bounds = frame.gs.footprint.boundingbox

    assert tuple(bounds) == (0, 0, 2, 1)
    assert bounds.crs == "EPSG:32633"


def test_odc_geometry_keeps_its_crs() -> None:
    projected = GeoVector.from_geometry(
        "POINT (500000 9000000)",
        crs="EPSG:32749",
    )

    restored = GeoVector.from_geometry(projected.gs.footprint)

    assert restored.crs.to_epsg() == 32749


@pytest.mark.parametrize(
    "timespan,expected",
    [("2025-01", []), ("2025-06", ["place"]), (None, ["place"])],
)
def test_query_filters_vector_dates_when_the_target_is_dated(
    timespan, expected
) -> None:
    dated = _places(["place"], [box(0, 0, 1, 1)])
    dated["datetime"] = pd.to_datetime(["2025-06-01"], utc=True)

    assert list(dated.gs.query(_ground(box(-1, -1, 2, 2), timespan)).name) == expected


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
