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

from geosave.geodata.core.raster.utils import grid
from geosave.geodata.errors import RasterValidationError
from tests.conftest import build_dataset


def grid_of(crs: str = "EPSG:32749") -> GeoBox:
    return GeoBox((2, 3), Affine(10, 0, 500000, 0, -10, 9000020), crs)


def bare(pixels=None) -> xr.Dataset:
    pixels = np.ones((2, 3), "uint16") if pixels is None else pixels
    return xr.Dataset({"red": (("y", "x"), pixels)})


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
        (xr.Dataset({"red": (("row", "col"), np.ones((3, 2)))}), "spatial dimensions"),
        (xr.Dataset({"red": (("y", "x"), np.ones((4, 4)))}), "pixels"),
    ],
)
def test_write_grid_refuses_pixels_that_do_not_fit(raster, match):
    with pytest.raises(RasterValidationError, match=match):
        raster.gs.write_grid(grid_of())


def test_write_grid_refuses_a_grid_without_a_crs():
    with pytest.raises(ValueError, match="CRS"):
        bare().gs.write_grid(GeoBox((2, 3), Affine(10, 0, 0, 0, -10, 20), None))


def test_write_grid_places_pixels_on_ground_control_points():
    pixels = multipoint([(0, 0), (4, 0), (0, 4), (4, 4)], None)
    world = multipoint([(10, 50), (10.1, 50), (10, 49.9), (10.1, 49.9)], "EPSG:4326")
    grid = GCPGeoBox((4, 4), GCPMapping(pixels, world))

    placed = xr.Dataset({"red": (("y", "x"), np.zeros((4, 4), "uint8"))}).gs.write_grid(
        grid
    )

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

    store = getattr(placed.gs, "to_netcdf" if suffix == ".nc" else "to_zarr")(
        tmp_path / f"grid{suffix}"
    )

    with xr.open_dataset(
        store, decode_coords="all", **({"engine": "zarr"} if suffix == ".zarr" else {})
    ) as restored:
        assert restored.odc.geobox == grid_of(crs)
        assert restored.rio.crs.to_epsg() == int(crs.split(":")[1])


def test_collection_metadata_reaches_the_merged_dataset():
    scene = bare().gs.write_grid(grid_of())
    tree = xr.DataTree.from_dict(
        {
            "s2": xr.Dataset(attrs={"title": "Forest"}),
            "s2/r10m": scene,
        }
    )

    assert tree["s2"].gs.to_dataset().attrs["title"] == "Forest"
    np.testing.assert_array_equal(tree["s2"].gs.to_dataset().red, scene.red)


def test_merge_combines_collections_and_one_collections_grids():
    fine = grid_of()
    coarse = GeoBox((1, 1), Affine(30, 0, 500000, 0, -30, 9000020), fine.crs)
    s2 = xr.DataTree.from_dict({"s2/r10m": bare().gs.write_grid(fine)})
    swir = xr.Dataset({"swir": (("y", "x"), np.ones((1, 1), "uint16"))}).gs.write_grid(
        coarse
    )
    dem = xr.DataTree.from_dict(
        {"dem/r10m": bare().rename(red="height").gs.write_grid(fine)}
    )

    tree = xr.merge([s2, xr.DataTree.from_dict({"s2/r30m": swir}), dem])

    assert list(tree.gs.datasets) == ["s2/r10m", "s2/r30m", "dem/r10m"]
    assert tree.gs.geobox is None
    assert tree.gs.groups == ("s2", "dem")


def test_a_tree_on_one_grid_reads_it_from_its_grid_nodes():
    tree = xr.merge(
        [
            xr.DataTree.from_dict({"s2/r10m": bare().gs.write_grid(grid_of())}),
            xr.DataTree.from_dict(
                {"dem/r10m": bare().rename(red="height").gs.write_grid(grid_of())}
            ),
        ]
    )

    assert tree.gs.geobox == grid_of()
    assert (
        build_dataset({"red": (("y", "x"), np.ones((2, 3)))}, grid_of()).gs.geobox
        == grid_of()
    )


@pytest.mark.parametrize("as_band", [False, True])
def test_write_grid_preserves_spatial_names_order_and_lazy_pixels(as_band):
    pixels = da.arange(6, chunks=3).reshape((3, 2))
    raster = xr.DataArray(pixels, dims=("longitude", "latitude"), name="red")
    if not as_band:
        raster = raster.to_dataset()

    placed = raster.gs.write_grid(grid_of("EPSG:4326"))

    band = placed if as_band else placed.red
    assert band.dims == ("longitude", "latitude")
    assert band.data is pixels
    assert placed.gs.geobox == grid_of("EPSG:4326")
    assert placed.latitude.attrs["axis"] == "Y"
    assert placed.longitude.attrs["axis"] == "X"


def test_write_grid_identifies_a_nonspatial_dataset_variable():
    raster = bare().assign(quality=xr.DataArray([1, 2], dims="time"))
    with pytest.raises(RasterValidationError, match="quality.*spatial dimensions"):
        raster.gs.write_grid(grid_of())


UTM = GeoBox((3, 4), Affine(10, 0, 500, 0, -10, 900), "EPSG:32633")


def test_write_grid_replaces_stale_crs_carriers_and_keeps_axis_names():
    raster = xr.Dataset({"red": (("latitude", "longitude"), np.ones((3, 4)))})
    raster = raster.assign_coords(
        old=xr.DataArray(0, attrs={"crs_wkt": "stale"}),
        spatial_ref=xr.DataArray(0, attrs={"spatial_ref": "stale"}),
    )
    assert raster.gs.grid_mapping_coords == ("old", "spatial_ref")

    placed = grid.write_grid(raster, UTM)

    # One carrier remains, under the name odc already read the CRS from.
    assert placed.gs.grid_mapping_coords == (str(raster.odc.crs_coord.name),)
    assert placed.red.dims == ("latitude", "longitude")
    assert placed.gs.geobox == UTM
    assert placed.latitude.attrs["axis"] == "Y"


def test_write_grid_rejects_a_shape_the_grid_does_not_have():
    raster = xr.Dataset({"red": (("y", "x"), np.ones((2, 2)))})
    with pytest.raises(RasterValidationError, match="shape"):
        grid.write_grid(raster, UTM)


def test_write_crs_keeps_coordinates_and_writes_cf_axes():
    placed = grid.write_grid(xr.Dataset({"red": (("y", "x"), np.ones((3, 4)))}), UTM)
    located = grid.write_crs(placed.drop_vars("spatial_ref"), "EPSG:32633")
    np.testing.assert_array_equal(located.x, placed.x)
    assert located.gs.geobox == UTM
    assert located.y.attrs["standard_name"] == "projection_y_coordinate"


@pytest.mark.parametrize("name", [None, "red"])
def test_dataarray_writes_keep_its_name_attrs_and_laziness(name):
    band = xr.DataArray(
        np.ones((3, 4)), dims=("y", "x"), name=name, attrs={"units": "1"}
    ).chunk()
    placed = band.gs.write_grid(UTM)
    assert isinstance(placed, xr.DataArray)
    assert placed.name == name
    assert placed.attrs["units"] == "1"
    assert placed.chunks is not None
    assert placed.gs.geobox == UTM
    assert band.gs.write_grid(UTM).gs.write_crs().gs.geobox == UTM
