"""Tests for the raster structure convention: ODC's names, CF in files."""

from __future__ import annotations

import dask.array as da
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
