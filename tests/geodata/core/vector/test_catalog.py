from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from geosave.geodata.core.vector.catalog import dated, valid_geometry
from geosave.geodata.errors import VectorValidationError


def build_rows(**columns) -> gpd.GeoDataFrame:
    """One Item row per entry of `id`, at the origin, with the Item columns."""
    count = len(columns["id"])
    fields = {"links": [[]] * count, "stac_extensions": [[]] * count}
    fields["assets"] = [{"image": {"href": "/data/image.tif"}}] * count
    fields.update(columns)
    return gpd.GeoDataFrame(fields, geometry=[Point(0, 0)] * count, crs=4326)


def test_a_catalog_keeps_its_collections_through_geoparquet(catalog, tmp_path):
    from geosave import open_vector

    path = catalog.gs.to_geoparquet(tmp_path / "items.parquet")

    assert set(open_vector(path).gs.collections) == {"s2", "s1", "label"}


def test_valid_geometry_names_the_broken_row() -> None:
    rows = build_rows(id=["fine", "broken"])
    rows.loc[1, "geometry"] = None

    with pytest.raises(VectorValidationError, match="position 1.*broken"):
        rows.gs.validate(valid_geometry)


def test_dated_refuses_a_period_that_ends_before_it_starts():
    rows = build_rows(
        id=["invalid"],
        datetime=["2025-01-15"],
        start_datetime=["2025-02-01"],
        end_datetime=["2025-01-01"],
    )

    with pytest.raises(VectorValidationError, match="end_datetime precedes"):
        rows.gs.validate(dated)
