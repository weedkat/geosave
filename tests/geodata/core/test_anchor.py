"""Where and when data exists: the anchor a search or a load reads."""

from __future__ import annotations

import pandas as pd
import pytest

from geosave.geodata import GeoAnchor, GeoVector
from geosave.geodata.utils.geocode import Place


def _anchor() -> GeoAnchor:
    return GeoAnchor.from_coordinates(52, 13, 500, 10, timespan="2025-06-11")


def test_replacing_the_timespan_keeps_the_grid() -> None:
    anchor = _anchor()

    month = anchor.replace(timespan="2025-06")

    assert month.geobox == anchor.geobox
    start, end = month.timespan
    assert pd.Timestamp(start) == pd.Timestamp("2025-06-01")
    assert pd.Timestamp(end) >= pd.Timestamp("2025-06-30")


def test_replacing_the_timespan_with_none_makes_the_anchor_timeless() -> None:
    assert _anchor().replace(timespan=None).timespan is None


def test_replacing_leaves_the_original_anchor_unchanged() -> None:
    anchor = _anchor()

    anchor.replace(timespan="2025-06")

    assert anchor.format("{start:%Y-%m-%d}") == "2025-06-11"


def test_an_anchor_covers_every_feature_of_a_vector() -> None:
    import geopandas as gpd
    import shapely

    fields = gpd.GeoDataFrame(
        geometry=[
            shapely.box(500_000, 0, 500_100, 100),
            shapely.box(500_300, 0, 500_400, 50),
        ],
        crs="EPSG:32633",
    )

    from_frame = GeoAnchor.from_geometry(fields, resolution=10, timespan="2024-07")
    from_series = GeoAnchor.from_geometry(fields.geometry, resolution=10)

    assert from_frame.geobox.shape == (10, 40)
    assert from_frame.geobox.crs == "EPSG:32633"
    assert from_series.geobox == from_frame.geobox
    assert from_frame.timespan is not None


def test_anchor_and_vector_construction() -> None:
    vector = GeoVector.from_geometry("POINT (13 52)")

    anchor = GeoAnchor.from_geometry(vector.gs.footprint, resolution=10)

    assert anchor.geobox.crs is not None
    assert anchor.geobox.crs.to_epsg() == 32633


def test_anchor_formats_place_and_time_through_a_template() -> None:
    anchor = GeoAnchor.from_coordinates(
        52, 13, 500, 10, timespan="2025-06-01/2025-06-11"
    )

    assert (
        anchor.format("{lat:.4f}N_{lon:.4f}E_{start:%Y%m%d}-{end:%Y%m%d}_{res:g}m")
        == "52.0000N_12.9999E_20250601-20250611_10m"
    )
    assert anchor.format("{start:%Y}/{start:%m}/{epsg}") == "2025/06/32633"
    assert anchor.format("{width:.0f}x{height:.0f}") == "5000x5000"


def test_timeless_anchor_refuses_a_date_field() -> None:
    anchor = GeoAnchor.from_coordinates(52, 13, 500, 10)

    with pytest.raises(ValueError, match="no 'start'"):
        anchor.format("{start:%Y}")


def test_anchor_refuses_a_field_it_does_not_offer() -> None:
    anchor = GeoAnchor.from_coordinates(52, 13, 500, 10)

    with pytest.raises(ValueError, match="'place'.*lat.*lon"):
        anchor.format("{place}")


def test_anchor_centroid_reads_longitude_first() -> None:
    longitude, latitude = GeoAnchor.from_coordinates(52, 13, 500, 10).lonlat

    assert longitude == pytest.approx(13, abs=1e-3)
    assert latitude == pytest.approx(52, abs=1e-3)


def test_locate_geocodes_the_centroid(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[float, float]] = []

    def resolve(latitude: float, longitude: float) -> Place:
        asked.append((latitude, longitude))
        return Place(city="Berlin")

    monkeypatch.setattr(Place, "from_coordinate", resolve)

    place = GeoAnchor.from_coordinates(52, 13, 500, 10).locate()

    assert place == Place(city="Berlin")
    assert asked[0] == pytest.approx((52, 13), abs=1e-3)
