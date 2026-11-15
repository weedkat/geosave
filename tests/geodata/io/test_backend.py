"""xarray opens GeoSave rasters and stacks by engine name."""

from __future__ import annotations

from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from geosave.geodata import open_raster_tree
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


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_the_engine_opens_a_tree_as_a_tree(tmp_path: Path, suffix: str) -> None:
    sample = build_tree(
        {"optical": build_raster(times=2), "dem": build_raster(times=1)}
    )
    writer = sample.gs.to_zarr if suffix == ".zarr" else sample.gs.to_netcdf
    store = writer(tmp_path / f"s0{suffix}")

    with xr.open_datatree(store, engine="geosave") as opened:
        with open_raster_tree(store) as expected:
            assert opened.identical(expected)
        assert sorted(opened.children) == ["dem", "optical"]


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
def test_the_engine_opens_one_group_as_a_dataset(tmp_path, suffix):
    source = build_raster(times=2)
    sample = build_tree({"scene": source})
    writer = sample.gs.to_zarr if suffix == ".zarr" else sample.gs.to_netcdf
    store = writer(tmp_path / f"archive{suffix}")

    with xr.open_dataset(store, engine="geosave", group="scene") as opened:
        xr.testing.assert_equal(opened, source)
    with xr.open_dataset(store, engine="geosave") as root:
        assert not root.data_vars


def test_the_engine_opens_a_banded_file_as_a_root_only_tree(tmp_path):
    from geosave.geodata.io import write_cog

    source = build_raster()
    path = write_cog(source, tmp_path / "scene.tif")

    with xr.open_datatree(path, engine="geosave") as opened:
        assert opened.groups == ("/",)
        assert opened.gs.geobox == source.gs.geobox
        np.testing.assert_array_equal(opened.to_dataset().red, source.red)


def test_the_engine_claims_no_file_unasked(tmp_path: Path) -> None:
    path = build_raster().gs.to_netcdf(tmp_path / "scene.nc")

    with xr.open_dataset(path) as opened:
        assert "gs" not in opened.encoding.get("engine", "")
    assert "geosave" in xr.backends.list_engines()
    assert not xr.backends.list_engines()["geosave"].guess_can_open(path)
