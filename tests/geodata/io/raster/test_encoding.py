"""Zarr and netCDF stores write a Dataset or a tree whole, now or deferred."""

import numpy as np
import pytest

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


def test_a_bad_store_suffix_is_rejected_before_writing(tmp_path):
    tree = build_tree({"a": build_raster(times=1), "b": build_raster(times=1)})

    with pytest.raises(ValueError, match="end in"):
        tree.gs.to_netcdf(tmp_path / "scene.bad")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("method", "suffix"), [("to_zarr", ".zarr"), ("to_netcdf", ".nc")]
)
def test_an_existing_store_is_kept(tmp_path, method, suffix):
    existing = tmp_path / f"scene{suffix}"
    existing.write_bytes(b"keep me")
    tree = build_tree({"a": build_raster(times=1), "b": build_raster(times=1)})

    with pytest.raises(FileExistsError):
        getattr(tree.gs, method)(existing)
    assert existing.read_bytes() == b"keep me"
