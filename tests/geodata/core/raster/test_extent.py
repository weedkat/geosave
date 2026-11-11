"""Declared spatial extent survives native raster selection and persistence."""

import numpy as np
import pytest
from shapely import from_wkt
import xarray as xr
from odc.geo.geom import Geometry

from geosave.geodata import io
from tests.geodata.conftest import build_raster


@pytest.fixture
def city():
    return Geometry(from_wkt("POLYGON ((13 52, 14 52, 13 53, 13 52))"), "EPSG:4326")


def test_extent_survives_band_selection_without_changing_pixels(city):
    source = build_raster().chunk()
    written = source.gs.write_extent(city)

    assert written.red.data is source.red.data
    assert "geospatial_bounds" not in source.attrs
    assert "geospatial_bounds" not in source.red.attrs
    assert written.gs.geobox == source.gs.geobox
    assert written.gs.region.extent == city
    assert written.red.gs.region.extent == city
    assert written.red.isel(x=slice(0, 2)).gs.region.extent == city

    cleared = written.gs.write_extent(None)
    assert cleared.gs.region.extent == source.gs.geobox.extent
    assert cleared.red.gs.region.extent == source.gs.geobox.extent
    assert written.gs.region.extent == city


def test_extent_needs_no_grid_and_tree_nodes_stand_alone(city):
    raster = xr.Dataset({"red": (("y", "x"), np.zeros((2, 2)))})
    tree = xr.DataTree.from_dict({"a": raster, "b/deep": raster})
    written = tree.gs.write_extent(city)

    assert written.gs.region.extent == city
    assert written["b/deep"].to_dataset().red.gs.region.extent == city
    assert raster.attrs == {}
    assert written["a"].gs.geobox is None


def test_dataset_and_tree_union_variable_extents(city):
    left = build_raster()[["red"]].gs.write_extent(city)
    other = Geometry(from_wkt("POLYGON ((15 52, 16 52, 15 53, 15 52))"), "EPSG:4326")
    right = build_raster()[["nir"]].gs.write_extent(other.to_crs("EPSG:3857"))
    dataset = xr.Dataset({"red": left.red, "nir": right.nir})
    tree = xr.DataTree.from_dict({"left": left, "right": right})
    expected = city.union(other)
    assert dataset.gs.region.extent.geom.equals_exact(expected.geom, 1e-8)
    assert tree.gs.region.extent.geom.equals_exact(expected.geom, 1e-8)


@pytest.mark.parametrize("suffix", ["nc", "zarr", "tif"])
def test_extent_round_trip(tmp_path, suffix):
    geometry = Geometry(
        from_wkt(
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0), "
            "(1 1, 1 2, 2 2, 2 1, 1 1)), ((5 0, 6 0, 5 1, 5 0)))"
        ),
        "EPSG:3857",
    )
    raster = build_raster().gs.write_extent(geometry)
    path = tmp_path / f"raster.{suffix}"
    if suffix == "nc":
        io.formats.netcdf.write_netcdf(raster, path)
    elif suffix == "zarr":
        io.formats.zarr.write_zarr(raster, path)
    else:
        io.formats.geotiff.write_cog(raster, path)
    with io.open_raster(path) as restored:
        assert restored.gs.region.extent == geometry
        assert restored.red.gs.region.extent == geometry
        np.testing.assert_array_equal(restored.red.values, raster.red.values)
