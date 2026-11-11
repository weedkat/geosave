"""Several GeoTIFFs are checked together before the first is written."""

import pytest
from upath import UPath

from geosave.geodata.io.formats import geotiff
from tests.conftest import build_tree
from tests.geodata.conftest import build_raster


def test_each_raster_is_written_to_its_destination(tmp_path):
    files = {
        UPath(tmp_path / "red.tif"): build_raster()[["red"]],
        UPath(tmp_path / "nir.tif"): build_raster()[["nir"]],
    }

    written = geotiff.write_files(files)

    assert [path.name for path in written] == ["red.tif", "nir.tif"]
    assert all(path.is_file() for path in written)


def test_a_deferred_batch_waits_for_compute(tmp_path):
    files = {UPath(tmp_path / "red.tif"): build_raster()[["red"]].chunk()}

    pending = geotiff.write_files(files, compute=False)

    assert not list(tmp_path.iterdir())
    assert [path.name for path in pending.compute()] == ["red.tif"]


@pytest.mark.parametrize("compute", [True, False])
def test_a_later_invalid_raster_prevents_every_write(tmp_path, compute):
    files = {
        UPath(tmp_path / "a.tif"): build_raster(times=1),
        UPath(tmp_path / "b.tif"): build_raster(times=2),
    }

    with pytest.raises(ValueError, match="one instant"):
        geotiff.write_files(files, compute=compute)

    assert list(tmp_path.iterdir()) == []


def test_an_existing_later_destination_prevents_earlier_writes(tmp_path):
    (tmp_path / "b.tif").write_bytes(b"keep me")
    files = {
        UPath(tmp_path / "a.tif"): build_raster(),
        UPath(tmp_path / "b.tif"): build_raster(),
    }

    with pytest.raises(FileExistsError):
        geotiff.write_files(files)

    assert not (tmp_path / "a.tif").exists()
    assert (tmp_path / "b.tif").read_bytes() == b"keep me"


def test_a_deferred_batch_rechecks_its_destinations_when_computed(tmp_path):
    files = {
        UPath(tmp_path / "a.tif"): build_raster(),
        UPath(tmp_path / "b.tif"): build_raster(),
    }
    pending = geotiff.write_files(files, compute=False)
    (tmp_path / "b.tif").write_bytes(b"created after planning")

    with pytest.raises(FileExistsError):
        pending.compute()

    assert not (tmp_path / "a.tif").exists()


def test_two_destinations_resolving_to_one_file_collide(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = {
        UPath("same.tif"): build_raster(),
        UPath(tmp_path / "same.tif"): build_raster(),
    }

    with pytest.raises(ValueError, match="collide"):
        geotiff.write_files(files)
    assert not list(tmp_path.iterdir())


def test_collections_sharing_a_path_format_without_collection_collide(tmp_path):
    tree = build_tree({"s2": build_raster(), "dem": build_raster()})

    with pytest.raises(ValueError, match="collide"):
        tree.gs.to_cog(tmp_path, format_paths="scene.tif")
    assert not list(tmp_path.iterdir())


def test_a_path_naming_a_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="format_paths"):
        build_raster().gs.to_cog(tmp_path / "scene.tif")
    assert not list(tmp_path.iterdir())
