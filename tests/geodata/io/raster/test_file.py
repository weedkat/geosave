"""Saved files and the scenes one write saves them as."""

from __future__ import annotations

import os

import dask
import numpy as np
import pytest
import xarray as xr
from upath import UPath

from geosave.geodata import open_raster
from geosave.geodata.io.raster.file import RasterFile, Scene
from tests.geodata.conftest import flat


def test_the_header_a_write_returns_is_the_one_the_file_reads_back(scene, tmp_path):
    (file, *_) = flat(scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif"))

    read = RasterFile.open(file.path)

    assert file.header.gs.variables == read.header.gs.variables
    assert file.header.gs.geobox == read.header.gs.geobox
    assert file.header.attrs == read.header.attrs
    assert {k: v.attrs for k, v in file.header.items()} == {
        k: v.attrs for k, v in read.header.items()
    }
    assert file.header.red.chunks is not None
    assert read.header.red.shape == file.header.red.shape


def test_a_remote_file_has_no_local_path() -> None:
    remote = RasterFile(UPath("memory:///bucket/r.tif"), xr.Dataset())

    with pytest.raises(TypeError, match="remote"):
        os.fspath(remote)


def test_files_saved_elsewhere_open_as_one_scene(scene, tmp_path) -> None:
    (red, nir) = flat(scene.isel(time=[0]).gs.to_cog(f"{tmp_path}/{{band}}.tif"))

    opened = Scene.open({"red": red.path, "nir": nir.path}, collection="s2")

    assert opened.collection == "s2"
    assert {k: f.header.gs.variables for k, f in opened.written.items()} == {
        "red": ("red",),
        "nir": ("nir",),
    }


def test_a_pending_scene_writes_nothing_until_computed(scene, tmp_path) -> None:
    scenes = scene.chunk().gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif", compute=False)

    assert list(tmp_path.iterdir()) == []
    assert repr(scenes[0]) == "Scene(None, pending)"
    (written,) = dask.compute(scenes)
    assert [file.name for file in flat(written)] == ["s_20250601.tif", "s_20250611.tif"]
    assert all(each.pending is None for each in written)
    assert flat(written)[0].header.gs.variables == ("red", "nir")


def test_a_pending_scene_writes_the_pixels_it_was_given(scene, tmp_path) -> None:
    pending = scene.chunk().gs.to_zarr(tmp_path / "s.zarr", compute=False)

    with open_raster(pending.compute()) as restored:
        np.testing.assert_array_equal(restored.red, scene.red)


def test_a_pending_scene_has_no_files_to_read(scene, tmp_path) -> None:
    pending = scene.gs.to_zarr(tmp_path / "s.zarr", compute=False)

    with pytest.raises(RuntimeError, match=r"\.compute\(\)"):
        pending.written
    with pytest.raises(RuntimeError, match=r"\.compute\(\)"):
        open_raster(pending)


def test_a_written_scene_computes_to_itself(scene, tmp_path) -> None:
    written = scene.gs.to_zarr(tmp_path / "s.zarr")

    assert written.compute() is written
    assert dask.compute(written) == (written,)
