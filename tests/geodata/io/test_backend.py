"""xarray opens GeoSave rasters and stacks by engine name."""

from __future__ import annotations

from pathlib import Path

import dask
import dask.array as da
import pytest
import xarray as xr

from geosave.geodata import open_raster
from tests.conftest import build_tree
from tests.geodata.conftest import build_raster


def _refuse(*args, **kwargs):
    raise AssertionError("pixels were computed")


def test_the_engine_stays_lazy_and_forwards_reader_options(tmp_path: Path) -> None:
    path = build_raster(times=2, packed=True).gs.to_zarr(tmp_path / "scene.zarr")

    with dask.config.set(scheduler=_refuse):
        lazy = xr.open_dataset(path, engine="geosave", chunks={})
        decoded = xr.open_dataset(path, engine="geosave", mask_and_scale=True)

    assert isinstance(lazy.red.data, da.Array)
    assert lazy.red.dtype == "uint16"
    assert decoded.red.dtype.kind == "f"
    lazy.close()
    decoded.close()


def test_the_engine_drops_named_variables(tmp_path: Path) -> None:
    path = build_raster().gs.to_zarr(tmp_path / "scene.zarr")

    with xr.open_dataset(path, engine="geosave", drop_variables="nir") as opened:
        assert list(opened.data_vars) == ["red"]


def test_the_engine_opens_a_tree_as_a_tree(tmp_path: Path) -> None:
    sample = build_tree({"optical": build_raster(times=2), "dem": build_raster()})
    store = sample.gs.to_zarr(tmp_path / "s0.zarr")

    with xr.open_datatree(store, engine="geosave") as opened:
        assert opened.identical(open_raster(store))
        assert sorted(opened.children) == ["dem", "optical"]


def test_the_engine_claims_no_file_unasked(tmp_path: Path) -> None:
    path = build_raster().gs.to_netcdf(tmp_path / "scene.nc")

    with xr.open_dataset(path) as opened:
        assert "gs" not in opened.encoding.get("engine", "")
    assert "geosave" in xr.backends.list_engines()
    assert not xr.backends.list_engines()["geosave"].guess_can_open(path)


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_xarray_controls_chunking_when_using_the_geosave_engine(tmp_path, suffix):
    source = build_raster()
    path = tmp_path / f"scene{suffix}"
    if suffix == ".nc":
        source.gs.to_netcdf(path)
    else:
        source.gs.to_zarr(path)
    with xr.open_dataset(path, engine="geosave", chunks=None) as opened:
        assert not isinstance(opened.red.data, da.Array)
    with xr.open_dataset(path, engine="geosave", chunks={}) as opened:
        assert isinstance(opened.red.data, da.Array)
        xr.testing.assert_equal(opened, source)
