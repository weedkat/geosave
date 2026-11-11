"""Where and when data exists: the region a search or a load reads."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest

from geosave.geodata import GeoRegion
from geosave.geodata.utils.geocode import Place
from geosave.geodata.utils.geometry import to_shapely


def _region() -> GeoRegion:
    return GeoRegion.from_coordinates(52, 13, timespan="2025-06-11")


def test_replacing_the_timespan_keeps_the_geometry() -> None:
    region = _region()

    month = region.replace(timespan="2025-06")

    assert month.extent == region.extent
    start, end = month.timespan
    assert pd.Timestamp(start) == pd.Timestamp("2025-06-01")
    assert pd.Timestamp(end) >= pd.Timestamp("2025-06-30")


def test_replacing_the_timespan_with_none_makes_the_region_timeless() -> None:
    assert _region().replace(timespan=None).timespan is None


def test_an_region_covers_every_feature_of_a_vector() -> None:
    import geopandas as gpd
    import shapely

    fields = gpd.GeoDataFrame(
        geometry=[
            shapely.box(500_000, 0, 500_100, 100),
            shapely.box(500_300, 0, 500_400, 50),
        ],
        crs="EPSG:32633",
    )

    from_frame = GeoRegion.from_geometry(fields, timespan="2024-07")
    from_series = GeoRegion.from_geometry(fields.geometry)

    assert from_frame.extent.geom.equals(fields.geometry.union_all())
    assert from_frame.crs == "EPSG:32633"
    assert from_series.extent == from_frame.extent
    assert from_frame.timespan is not None


def test_region_and_vector_construction() -> None:
    vector = gpd.GeoDataFrame(geometry=[to_shapely("POINT (13 52)")], crs="EPSG:4326")

    region = GeoRegion.from_geometry(vector.gs.footprint)

    assert region.crs is not None
    assert region.crs.to_epsg() == 4326


def test_region_formats_place_and_time_through_a_template() -> None:
    region = GeoRegion.from_coordinates(52, 13, timespan="2025-06-01/2025-06-11")

    assert (
        region.format("{lat:.4f}N_{lon:.4f}E_{start:%Y%m%d}-{end:%Y%m%d}")
        == "52.0000N_13.0000E_20250601-20250611"
    )
    assert region.format("{start:%Y}/{start:%m}/{epsg}") == "2025/06/4326"
    assert region.format("{width:.0f}x{height:.0f}") == "0x0"


def test_timeless_region_refuses_a_date_field() -> None:
    region = GeoRegion.from_coordinates(52, 13)

    with pytest.raises(ValueError, match="no 'start'"):
        region.format("{start:%Y}")


def test_region_formats_explicit_grid_without_changing_location() -> None:
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import BoundingBox

    region = _region()
    grid = GeoBox.from_bbox(
        BoundingBox(500000, 5000000, 500040, 5000060, crs="EPSG:32633"),
        resolution=10,
    )

    assert (
        region.format(
            "{lat:.2f}_{lon:.2f}_{epsg}_{res:.0f}_{width:.0f}x{height:.0f}",
            geobox=grid,
        )
        == "52.00_13.00_32633_10_40x60"
    )
    assert region.format("{epsg}_{width:.0f}x{height:.0f}") == "4326_0x0"
    with pytest.raises(ValueError, match="res"):
        region.format("{res}")


def test_region_centroid_reads_longitude_first() -> None:
    longitude, latitude = GeoRegion.from_coordinates(52, 13).lonlat

    assert longitude == pytest.approx(13, abs=1e-3)
    assert latitude == pytest.approx(52, abs=1e-3)


def test_locate_geocodes_the_centroid(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[float, float]] = []

    def resolve(latitude: float, longitude: float) -> Place:
        asked.append((latitude, longitude))
        return Place(city="Berlin")

    monkeypatch.setattr(Place, "from_coordinate", resolve)

    place = GeoRegion.from_coordinates(52, 13).locate()

    assert place == Place(city="Berlin")
    assert asked[0] == pytest.approx((52, 13), abs=1e-3)


def test_frame_region_unions_geometry_and_envelopes_only_dated_rows():
    import shapely

    frame = gpd.GeoDataFrame(
        {"datetime": ["2025-01-01", None, "2025-02-01"]},
        geometry=[
            shapely.box(0, 0, 10, 10),
            shapely.box(20, 0, 30, 10),
            shapely.box(0, 20, 10, 30),
        ],
        crs="EPSG:3857",
    )
    region = frame.gs.region
    assert region.extent.geom.equals(frame.geometry.union_all())
    assert region.crs == "EPSG:3857"
    assert region.timespan == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01"))
    assert frame.iloc[[1]].gs.region.timespan is None


def test_tree_region_covers_different_grids_without_loading_pixels():
    import dask.array as da
    from odc.geo.geobox import GeoBox
    from tests.conftest import build_dataset, build_tree

    fine = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:3857", resolution=10)
    coarse = GeoBox.from_bbox((30, 0, 70, 40), crs="EPSG:3857", resolution=20)
    tree = build_tree(
        {
            "s2/r10m": build_dataset(
                {"red": (("y", "x"), da.zeros((4, 4), chunks=2))}, fine
            ),
            "s2/r20m": build_dataset(
                {"swir": (("y", "x"), da.zeros(tuple(coarse.shape), chunks=1))}, coarse
            ),
        }
    )
    region = tree.gs.region
    assert region.extent == fine.extent.union(coarse.extent)
    assert isinstance(tree["s2/r10m"].red.data, da.Array)


def test_region_rejects_incomplete_time_and_unknown_geometry():
    import shapely

    frame = gpd.GeoDataFrame(
        {"start_datetime": ["2025-01-01"]},
        geometry=[shapely.Point(1, 1)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="both start_datetime and end_datetime"):
        _ = frame.gs.region
    with pytest.raises(ValueError, match="CRS"):
        _ = frame.set_crs(None, allow_override=True).gs.region


def test_catalog_region_ignores_published_timeless_bounds():
    from datetime import datetime, UTC
    import pystac
    from geosave.geodata.stac import create_catalog
    from geosave.geodata.stac.item import TIMELESS

    geometry = GeoRegion.from_bbox((0, 0, 1, 1)).extent.json
    timeless = pystac.Item(
        "dem",
        geometry,
        [0, 0, 1, 1],
        None,
        {},
        start_datetime=TIMELESS[0],
        end_datetime=TIMELESS[1],
    )
    dated = pystac.Item(
        "scene", geometry, [0, 0, 1, 1], datetime(2025, 1, 1, tzinfo=UTC), {}
    )
    assert create_catalog([timeless]).gs.region.timespan is None
    mixed = create_catalog([timeless, dated])
    assert mixed.gs.region.timespan == mixed.iloc[1].gs.region.timespan
