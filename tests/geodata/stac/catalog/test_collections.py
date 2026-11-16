"""An item table carries its Collections in its footer, reconciled when tables join."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from geosave.geodata.stac.catalog import (
    concat_catalogs,
    create_catalog,
    merge_collections,
)
from tests.geodata.conftest import create_test_collection
from tests.geodata.stac.conftest import entry


def test_catalog_rejects_conflicting_collection_definitions():
    collection = create_test_collection("optical", description="Optical imagery")
    conflicting = collection.clone()
    conflicting.license = "CC-BY-4.0"

    with pytest.raises(ValueError, match="optical.*license"):
        create_catalog([], collections=[collection, conflicting])
    with pytest.raises(ValueError, match="optical.*license"):
        merge_collections([collection, conflicting])


def test_concatenated_catalogs_keep_every_collection():
    optical = create_test_collection("optical", description="Optical imagery")
    radar = create_test_collection("radar", description="Radar imagery")
    optical_rows = create_catalog([entry("optical")], collections=[optical])
    radar_rows = create_catalog([entry("radar")], collections=[radar])

    rows = concat_catalogs([optical_rows, radar_rows])

    assert rows["collection"].tolist() == ["optical", "radar"]
    assert list(rows.gs.collections) == ["optical", "radar"]
    with pytest.raises(ValueError, match="at least one"):
        concat_catalogs([])


def test_a_concatenated_footer_is_a_copy_and_conflicts_are_refused():
    optical = create_test_collection("optical", description="Optical imagery")
    catalog = create_catalog([entry("optical")], collections=[optical])

    combined = concat_catalogs([catalog, catalog])
    combined.attrs["stac-geoparquet"]["collections"]["optical"]["description"] = "changed"

    assert len(combined) == 2
    assert catalog.gs.collections["optical"].description == "Optical imagery"
    with pytest.raises(ValueError, match="description"):
        concat_catalogs([catalog, combined])


def test_concat_refuses_tables_on_different_crs() -> None:
    geographic = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    projected = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3857")

    with pytest.raises(ValueError, match="CRS"):
        concat_catalogs([geographic, projected])
