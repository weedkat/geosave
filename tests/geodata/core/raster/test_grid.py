"""Placing rasters on grids, and making a placed raster one collection's grid node."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.gcp import GCPGeoBox, GCPMapping
from odc.geo.geobox import GeoBox
from odc.geo.geom import multipoint

from geosave.geodata.core.raster.grid import name_grid
from tests.conftest import build_dataset


def grid_of(crs: str = "EPSG:32749") -> GeoBox:
    return GeoBox((2, 3), Affine(10, 0, 500000, 0, -10, 9000020), crs)


def bare(pixels=None) -> xr.Dataset:
    pixels = np.ones((2, 3), "uint16") if pixels is None else pixels
    return xr.Dataset({"red": (("y", "x"), pixels)})


def test_name_grid_follows_eopf():
    metres = GeoBox.from_bbox((500000, 9000000, 500640, 9000640), crs="EPSG:32749", resolution=10)
    degrees = GeoBox.from_bbox((100, 0, 100.01, 0.01), crs="EPSG:4326", resolution=0.0001)

    assert name_grid(metres) == "r10m"
    assert name_grid(degrees) == "r0.0001deg"


def test_a_bare_raster_has_no_grid_until_one_is_written():
    assert bare().gs.geobox is None
    with pytest.raises(ValueError, match="write_grid"):
        bare().gs.write_crs("EPSG:32749")


@pytest.mark.parametrize("as_band", [False, True])
def test_write_grid_places_pixels_on_the_grid(as_band):
    raster = bare().red if as_band else bare()

    placed = raster.gs.write_grid(grid_of())

    assert placed.gs.geobox == grid_of()
    assert placed.y.attrs["axis"] == "Y"
    assert placed.x.attrs["axis"] == "X"


def test_write_grid_links_every_variable_to_the_crs():
    placed = bare().assign(nir=bare().red).gs.write_grid(grid_of())

    for name in ("red", "nir"):
        assert placed[name].encoding.get("grid_mapping") == "spatial_ref" or (
            placed[name].attrs.get("grid_mapping") == "spatial_ref"
        )


def test_write_grid_keeps_pixels_lazy_and_other_coordinates():
    pixels = da.ones((2, 2, 3), chunks=(1, 2, 3))
    dates = np.array(["2025-06-01", "2025-06-02"], "datetime64[ns]")
    raster = xr.Dataset({"red": (("time", "y", "x"), pixels)}, coords={"time": dates})

    placed = raster.gs.write_grid(grid_of())

    assert placed.red.data is pixels
    np.testing.assert_array_equal(placed.time.values, dates)


@pytest.mark.parametrize(
    ("crs", "name", "units"),
    [
        ("EPSG:32749", "projection_y_coordinate", None),
        ("EPSG:4326", "latitude", "degrees_north"),
    ],
)
def test_write_grid_names_axes_by_the_crs(crs, name, units):
    placed = bare().gs.write_grid(grid_of(crs))

    assert placed.y.attrs["standard_name"] == name
    if units:
        assert placed.y.attrs["units"] == units


def test_write_grid_replaces_the_grid_a_raster_carried():
    first = bare().gs.write_grid(grid_of("EPSG:32749"))

    moved = first.gs.write_grid(grid_of("EPSG:32633"))

    assert moved.gs.geobox == grid_of("EPSG:32633")
    assert "spatial_ref" in moved.coords


@pytest.mark.parametrize(
    ("raster", "match"),
    [
        (xr.Dataset({"red": (("x", "y"), np.ones((3, 2)))}), "must end in"),
        (xr.Dataset({"red": (("y", "x"), np.ones((4, 4)))}), "pixels"),
    ],
)
def test_write_grid_refuses_pixels_that_do_not_fit(raster, match):
    with pytest.raises(ValueError, match=match):
        raster.gs.write_grid(grid_of())


def test_write_grid_refuses_a_grid_without_a_crs():
    with pytest.raises(ValueError, match="CRS"):
        bare().gs.write_grid(GeoBox((2, 3), Affine(10, 0, 0, 0, -10, 20), None))


def test_write_grid_places_pixels_on_ground_control_points():
    pixels = multipoint([(0, 0), (4, 0), (0, 4), (4, 4)], None)
    world = multipoint([(10, 50), (10.1, 50), (10, 49.9), (10.1, 49.9)], "EPSG:4326")
    grid = GCPGeoBox((4, 4), GCPMapping(pixels, world))

    placed = xr.Dataset({"red": (("y", "x"), np.zeros((4, 4), "uint8"))}).gs.write_grid(grid)

    assert isinstance(placed.gs.geobox, GCPGeoBox)
    # Its axes count pixels, so no CF standard name may claim they are ground.
    assert "standard_name" not in placed.y.attrs


def test_write_crs_names_what_existing_coordinates_measure():
    placed = bare().gs.write_grid(grid_of())
    stripped = placed.assign_coords(y=placed.y.copy(data=placed.y.values))
    stripped.y.attrs.clear()

    rewritten = stripped.gs.write_crs()

    assert rewritten.y.attrs["standard_name"] == "projection_y_coordinate"


@pytest.mark.parametrize("crs", ["EPSG:32749", "EPSG:4326"])
@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_a_placed_raster_keeps_its_grid_through_a_store(tmp_path: Path, crs, suffix):
    placed = bare().gs.write_grid(grid_of(crs))

    store = getattr(placed.gs, "to_netcdf" if suffix == ".nc" else "to_zarr")(tmp_path / f"grid{suffix}")

    with xr.open_dataset(store, decode_coords="all", **({"engine": "zarr"} if suffix == ".zarr" else {})) as restored:
        assert restored.odc.geobox == grid_of(crs)
        assert restored.rio.crs.to_epsg() == int(crs.split(":")[1])


def test_to_datatree_names_the_grid_node_after_its_grid():
    tree = bare().gs.write_grid(grid_of()).gs.to_datatree("s2")

    assert list(tree.gs.rasters) == ["s2/r10m"]
    assert tree.gs.groups == ("s2",)
    assert tree.gs.geobox == grid_of()


def test_to_datatree_moves_dataset_attrs_to_the_collection_and_back():
    scene = bare().gs.write_grid(grid_of()).assign_attrs(title="Forest")

    tree = scene.gs.to_datatree("s2")

    assert tree["s2"].attrs["title"] == "Forest"
    assert "title" not in tree["s2/r10m"].attrs
    assert tree["s2"].gs.to_dataset().attrs["title"] == "Forest"
    np.testing.assert_array_equal(tree["s2"].gs.to_dataset().red, scene.red)


def test_to_datatree_refuses_a_raster_without_a_grid():
    with pytest.raises(ValueError, match="write_grid"):
        bare().gs.to_datatree("s2")


def test_to_datatree_takes_a_band():
    band = bare().red.gs.write_grid(grid_of())

    assert list(band.gs.to_datatree("red").gs.rasters) == ["red/r10m"]


def test_merge_combines_collections_and_one_collections_grids():
    fine = grid_of()
    coarse = GeoBox((1, 1), Affine(30, 0, 500000, 0, -30, 9000020), fine.crs)
    s2 = bare().gs.write_grid(fine).gs.to_datatree("s2")
    swir = xr.Dataset({"swir": (("y", "x"), np.ones((1, 1), "uint16"))}).gs.write_grid(coarse)
    dem = bare().rename(red="height").gs.write_grid(fine).gs.to_datatree("dem")

    tree = xr.merge([s2, swir.gs.to_datatree("s2"), dem])

    assert list(tree.gs.rasters) == ["s2/r10m", "s2/r30m", "dem/r10m"]
    assert tree.gs.geobox is None
    assert tree.gs.groups == ("s2", "dem")


def test_a_tree_on_one_grid_reads_it_from_its_grid_nodes():
    tree = xr.merge([
        bare().gs.write_grid(grid_of()).gs.to_datatree("s2"),
        bare().rename(red="height").gs.write_grid(grid_of()).gs.to_datatree("dem"),
    ])

    assert tree.gs.geobox == grid_of()
    assert build_dataset({"red": (("y", "x"), np.ones((2, 3)))}, grid_of()).gs.geobox == grid_of()
