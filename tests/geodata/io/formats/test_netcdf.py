from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from upath import UPath

from geosave.geodata import (
    open_raster,
)
from geosave.geodata.io.formats import netcdf
from tests.geodata.conftest import _size, build_raster


@pytest.mark.parametrize("crs", ["EPSG:32749", "EPSG:4326"])
@pytest.mark.parametrize("times", [0, 2])
def test_netcdf_round_trip_preserves_the_profile(
    tmp_path: Path, crs: str, times: int
) -> None:
    written = build_raster(crs=crs, times=times, packed=True)

    destination = netcdf.write_netcdf(written, tmp_path / "scene.nc")
    restored = netcdf.open_netcdf(destination)

    assert restored.gs.geobox == written.gs.geobox
    assert restored.red.dtype == np.dtype("uint16")
    assert int(restored.red.values.ravel()[0]) == 1000
    assert restored["y"].attrs["standard_name"] == written["y"].attrs["standard_name"]


def test_netcdf_reads_a_file_url(tmp_path: Path) -> None:
    written = build_raster()
    netcdf.write_netcdf(written, tmp_path / "scene.nc")

    with netcdf.open_netcdf((tmp_path / "scene.nc").as_uri()) as restored:
        np.testing.assert_array_equal(restored["red"], written["red"])


def test_a_deferred_file_is_written_only_once_computed(tmp_path: Path) -> None:
    destination = tmp_path / "deferred.nc"

    pending = build_raster().gs.to_netcdf(destination, compute=False)

    assert not destination.exists()
    saved = pending.compute()
    assert isinstance(saved, UPath)
    assert saved == destination
    with open_raster(saved) as restored:
        assert restored.gs.geobox == build_raster().gs.geobox


def test_netcdf_remote_round_trip(bucket: str, tmp_path) -> None:

    written = build_raster(times=2)

    saved = written.gs.to_netcdf(f"{bucket}/scene.nc", storage_options={})

    assert saved == UPath(f"{bucket}/scene.nc")
    saved.fs.get(saved.path, str(tmp_path / "copy.nc"))
    with open_raster(tmp_path / "copy.nc") as restored:
        np.testing.assert_array_equal(restored.red, written.red)
    with open_raster(saved, chunks={}) as restored:
        assert restored.red.chunks is not None
        np.testing.assert_array_equal(restored.red, written.red)
    # A deferred upload stages the whole file when computed.
    pending = written.gs.to_netcdf(f"{bucket}/later.nc", compute=False)
    assert pending.compute() == UPath(f"{bucket}/later.nc")


def _foreign() -> xr.Dataset:
    """Build a geographic raster named the way odc and most tools write it."""
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    built = xr.Dataset(
        {"red": (("latitude", "longitude"), np.ones(grid.shape, "uint16"))},
        coords=xr_coords(grid),
    )
    built.red.encoding["grid_mapping"] = "spatial_ref"
    return built


def test_netcdf_reads_a_foreign_grid_onto_y_x(tmp_path: Path) -> None:
    foreign = _foreign()
    foreign.to_netcdf(tmp_path / "foreign.nc")

    restored = netcdf.open_netcdf(tmp_path / "foreign.nc", chunks={})

    assert restored.red.dims == ("y", "x")
    assert restored.gs.geobox == foreign.odc.geobox
    assert restored.red.chunks is not None
    restored.close()


@pytest.mark.integration
def test_a_netcdf_file_uploads(remote) -> None:
    prefix, options = remote

    written = build_raster(times=2).gs.to_netcdf(
        f"{prefix}/forest.nc", storage_options=options
    )

    assert written == f"{prefix}/forest.nc"
    assert _size(written, options) > 0


def test_remote_netcdf_staging_lives_until_close(bucket, monkeypatch):
    from tempfile import TemporaryDirectory

    folders = []

    def temporary():
        folder = TemporaryDirectory()
        folders.append(Path(folder.name))
        return folder

    source = build_raster(times=1)
    saved = source.gs.to_netcdf(f"{bucket}/lifetime.nc")
    monkeypatch.setattr(netcdf, "TemporaryDirectory", temporary)
    restored = open_raster(saved, chunks={})
    assert folders[0].exists()
    np.testing.assert_array_equal(restored.red, source.red)
    restored.close()
    restored.close()
    assert not folders[0].exists()

    with pytest.raises(FileNotFoundError):
        open_raster(f"{bucket}/missing.nc")
    assert not folders[1].exists()


def test_a_tree_round_trips_through_one_netcdf_file(tmp_path):
    from geosave.geodata import open_raster
    from tests.geodata.core.raster.test_datatree import build_mixed_collection

    tree = build_mixed_collection()

    path = tree.gs.to_netcdf(tmp_path / "scene.nc")

    assert path == tmp_path / "scene.nc"
    with open_raster(path) as restored:
        assert list(restored.gs.rasters) == ["s2/r10m", "s2/r20m"]
        for name, raster in tree.gs.rasters.items():
            loaded = restored.gs.rasters[name]
            assert loaded.gs.geobox == raster.gs.geobox
            np.testing.assert_array_equal(loaded.time.values, raster.time.values)
