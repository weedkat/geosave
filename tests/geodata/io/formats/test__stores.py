"""Zarr and netCDF stores write a Dataset or a tree whole, now or deferred."""

import dask
import numpy as np
import pytest
import xarray as xr

from geosave.geodata.io.formats import netcdf, zarr
from tests.conftest import build_tree
from tests.geodata.conftest import build_raster


@pytest.mark.parametrize(
    ("method", "suffix"), [("to_zarr", ".zarr"), ("to_netcdf", ".nc")]
)
def test_band_encoding_reaches_the_store(tmp_path, method, suffix):
    from geosave import open_raster

    source = build_raster(times=2).chunk()
    encoding = {"red": {"dtype": "int32"}, "nir": {"dtype": "int16"}}

    saved = getattr(source.gs, method)(tmp_path / f"scene{suffix}", encoding=encoding)

    with open_raster(saved) as restored:
        assert restored.red.dtype == "int32"
        assert restored.nir.dtype == "int16"
        np.testing.assert_array_equal(restored.red, source.red)
    assert encoding == {"red": {"dtype": "int32"}, "nir": {"dtype": "int16"}}


@pytest.mark.parametrize("compute", [True, False])
def test_a_bad_store_suffix_is_rejected_before_writing(tmp_path, compute):
    tree = build_tree({"a": build_raster(), "b": build_raster()})

    with pytest.raises(ValueError, match="end in"):
        tree.gs.to_netcdf(tmp_path / "scene.bad", compute=compute)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("method", "suffix"), [("to_zarr", ".zarr"), ("to_netcdf", ".nc")]
)
def test_an_existing_store_is_kept(tmp_path, method, suffix):
    existing = tmp_path / f"scene{suffix}"
    existing.write_bytes(b"keep me")
    tree = build_tree({"a": build_raster(), "b": build_raster()})

    with pytest.raises(FileExistsError):
        getattr(tree.gs, method)(existing)
    assert existing.read_bytes() == b"keep me"


@pytest.mark.parametrize(
    ("method", "suffix"), [("to_zarr", ".zarr"), ("to_netcdf", ".nc")]
)
def test_an_unknown_encoding_name_is_rejected_before_creating_outputs(
    tmp_path, method, suffix
):
    with pytest.raises(KeyError, match="typo"):
        getattr(build_raster().gs, method)(
            tmp_path / f"scene{suffix}", encoding={"typo": {"dtype": "int32"}}
        )
    assert not list(tmp_path.iterdir())


def test_tree_encoding_reaches_each_node_holding_the_variable(tmp_path):
    from geosave import open_raster

    source = build_raster()
    tree = build_tree({"a": source[["red"]], "b": source[["nir"]]})

    saved = tree.gs.to_netcdf(
        tmp_path / "scene.nc",
        encoding={"red": {"dtype": "int32"}, "nir": {"dtype": "int16"}},
    )

    with open_raster(saved) as restored:
        assert restored["a"].gs.to_dataset().red.dtype == "int32"
        assert restored["b"].gs.to_dataset().nir.dtype == "int16"


@pytest.mark.parametrize("module", [zarr, netcdf])
def test_a_deferred_store_computes_to_its_path(tmp_path, module):
    suffix = ".zarr" if module is zarr else ".nc"
    tree = build_tree({"s2": build_raster(), "dem": build_raster()})

    pending = getattr(module, f"write_{module.__name__.split('.')[-1]}")(
        tree, tmp_path / f"scene{suffix}", compute=False
    )

    assert not list(tmp_path.iterdir())
    (written,) = dask.compute(pending)
    assert written == tmp_path / f"scene{suffix}"


def test_a_deferred_store_rechecks_its_destination_when_computed(tmp_path):
    tree = build_tree({"a": build_raster()})
    pending = tree.gs.to_netcdf(tmp_path / "scene.nc", compute=False)
    (tmp_path / "scene.nc").write_bytes(b"created after planning")

    with pytest.raises(FileExistsError):
        pending.compute()
    assert (tmp_path / "scene.nc").read_bytes() == b"created after planning"


def test_a_store_refuses_anything_but_a_raster(tmp_path):
    with pytest.raises(TypeError, match="Dataset or DataTree"):
        zarr.write_zarr(build_raster().red, tmp_path / "band.zarr")
    assert isinstance(build_raster(), xr.Dataset)
