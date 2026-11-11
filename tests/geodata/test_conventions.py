"""Tests for the spatial dimension convention."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave.geodata.conventions import to_yx

GRID = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")


def loaded(dims: tuple[str, str], name: str = "red") -> xr.Dataset:
    """Build a lazy raster the way a foreign loader names its grid."""
    built = xr.Dataset(
        {name: (dims, da.ones(tuple(GRID.shape), chunks=(1, 3)))},
        coords=xr_coords(GRID, dims=dims),
    )
    built[name].encoding["grid_mapping"] = "spatial_ref"
    return built


@pytest.mark.parametrize("dims", [("latitude", "longitude"), ("lat", "lon")])
def test_to_yx_renames_a_foreign_grid_without_computing(dims) -> None:
    result = to_yx(loaded(dims))

    assert result.red.dims == ("y", "x")
    assert result.odc.geobox == GRID
    assert isinstance(result.red.data, da.Array)
    assert result.red.encoding["grid_mapping"] == "spatial_ref"


def test_to_yx_finds_a_pair_only_its_cf_attrs_name() -> None:
    labelled = loaded(("north", "east")).expand_dims(depth=[0.5, 1.5], axis=-1)
    labelled.north.attrs["axis"] = "Y"
    labelled.east.attrs["standard_name"] = "longitude"

    result = to_yx(labelled)

    assert result.red.dims == ("y", "x", "depth")
    assert result.odc.geobox == GRID


def test_to_yx_leaves_an_unlabelled_pair_alone() -> None:
    unlabelled = loaded(("north", "east"))
    unlabelled.north.attrs = {}
    unlabelled.east.attrs = {}

    assert to_yx(unlabelled) is unlabelled


def test_to_yx_renames_one_band() -> None:
    result = to_yx(loaded(("latitude", "longitude")).red)

    assert result.dims == ("y", "x")
    assert result.odc.geobox == GRID


def test_a_tree_renames_node_by_node_with_map_over_datasets() -> None:
    foreign = loaded(("latitude", "longitude"))
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(coords=xr_coords(GRID)),
            "/optical": foreign,
            "/dem": loaded(("y", "x"), name="elevation"),
        }
    )

    result = tree.map_over_datasets(to_yx)

    assert result["optical"].dataset.red.dims == ("y", "x")
    assert result["dem"].dataset.elevation.dims == ("y", "x")
    assert result.dataset.odc.geobox == GRID
    assert isinstance(result["optical"].dataset.red.data, da.Array)


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


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
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


def test_cf_units_identify_spatial_axes_without_name_guesses():
    from dask.callbacks import Callback

    raster = loaded(("north", "east"))
    with Callback(pretask=lambda *_: pytest.fail("discovery computed pixels")):
        normalized = to_yx(raster)
    assert normalized.red.dims == ("y", "x")
    assert normalized.gs.geobox == GRID


def test_ambiguous_cf_axes_do_not_choose_a_grid():
    raster = xr.DataArray(
        np.zeros((2, 2, 3)),
        dims=("north", "other", "east"),
        coords={
            "north": ("north", [0, 1], {"axis": "Y"}),
            "other": ("other", [0, 1], {"axis": "Y"}),
            "east": ("east", [0, 1, 2], {"axis": "X"}),
        },
    )
    assert raster.gs.spatial_dims is None


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
