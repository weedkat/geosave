"""Tests for the raster structure convention: ODC's names, CF in files."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from tests.conftest import place

GRID = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")


def loaded(dims: tuple[str, str], name: str = "red") -> xr.Dataset:
    """Build a lazy raster whose grid a loader named `dims`."""
    built = xr.Dataset(
        {name: (dims, da.ones(tuple(GRID.shape), chunks=(1, 3)))},
        coords=xr_coords(GRID, dims=dims),
    )
    built[name].encoding["grid_mapping"] = "spatial_ref"
    return built


@pytest.mark.parametrize(
    "dims", [("y", "x"), ("latitude", "longitude"), ("lat", "lon")]
)
def test_spatial_dims_follow_odc_names_without_computing(dims) -> None:
    from dask.callbacks import Callback

    raster = loaded(dims)
    with Callback(pretask=lambda *_: pytest.fail("discovery computed pixels")):
        assert raster.gs.spatial_dims == dims
        assert raster.red.gs.spatial_dims == dims
        assert raster.gs.geobox == GRID


def test_cf_attrs_alone_do_not_name_spatial_axes() -> None:
    raster = xr.DataArray(
        np.zeros((2, 3)),
        dims=("north", "east"),
        coords={
            "north": ("north", [0, 1], {"axis": "Y"}),
            "east": ("east", [0, 1, 2], {"axis": "X"}),
        },
    )
    assert raster.gs.spatial_dims is None


def test_reprojecting_onto_a_geographic_grid_names_latitude_longitude() -> None:
    mercator = GeoBox((2, 3), Affine(1e5, 0, 1.1e6, 0, -1e5, 2.2e6), "EPSG:3857")
    projected = loaded(("y", "x")).pipe(place, mercator)

    result = projected.gs.reproject(GRID)

    assert result.red.dims == ("latitude", "longitude")
    assert result.gs.geobox == GRID


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
def test_files_carry_cf_axes_that_memory_does_not(tmp_path, suffix) -> None:
    from geosave.geodata import open_raster

    raster = loaded(("y", "x"))
    path = tmp_path / f"grid.{suffix}"
    if suffix == "nc":
        raster.gs.to_netcdf(path)
    else:
        raster.gs.to_zarr(path)

    assert "axis" not in raster.y.attrs
    opened = open_raster(path)
    assert opened.y.attrs["axis"] == "Y"
    assert opened.x.attrs["standard_name"] == "longitude"
    assert opened.gs.geobox == GRID


def _monthly() -> xr.Dataset:
    from tests.geodata.conftest import assign_cells, build_raster

    raster = build_raster(times=2)
    return assign_cells(
        raster,
        [
            (np.datetime64("2025-06-01"), np.datetime64("2025-07-01")),
            (np.datetime64("2025-07-01"), np.datetime64("2025-08-01")),
        ],
    )


@pytest.mark.parametrize("suffix", ["nc"])
def test_a_raster_stating_its_cells_twice_is_refused_on_write(tmp_path, suffix):
    monthly = _monthly()
    monthly["cells"] = (
        ("time", "vertices"),
        np.stack([monthly.start_datetime.values, monthly.end_datetime.values], axis=-1),
    )
    monthly.time.attrs["bounds"] = "cells"

    with pytest.raises(ValueError, match="cells.*start_datetime"):
        if suffix == "nc":
            monthly.gs.to_netcdf(tmp_path / "monthly.nc")
        else:
            monthly.gs.to_zarr(tmp_path / "monthly.zarr")


def test_curvilinear_coordinates_do_not_imply_affine_spatial_axes():
    raster = xr.DataArray(
        np.zeros((2, 3)),
        dims=("row", "col"),
        coords={
            "latitude": (
                ("row", "col"),
                np.ones((2, 3)),
                {"standard_name": "latitude"},
            ),
            "longitude": (
                ("row", "col"),
                np.ones((2, 3)),
                {"standard_name": "longitude"},
            ),
        },
    )
    assert raster.gs.spatial_dims is None
