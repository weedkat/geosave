from pathlib import Path

import numpy as np
import pytest
from upath import UPath
import xarray as xr

from geosave.geodata.io import netcdf
from tests.geodata.conftest import build_raster


@pytest.mark.parametrize("crs", ["EPSG:32749", "EPSG:4326"])
@pytest.mark.parametrize("times", [0, 2])
def test_netcdf_round_trip_preserves_the_profile(
    tmp_path: Path, crs: str, times: int
) -> None:
    written = build_raster(crs=crs, times=times, packed=True)

    destination = netcdf.write(written, tmp_path / "scene.nc").path
    restored = netcdf.open_raster(destination)

    assert restored.gs.geobox == written.gs.geobox
    assert restored.red.dtype == np.dtype("uint16")
    assert int(restored.red.values.ravel()[0]) == 1000
    assert restored["y"].attrs["standard_name"] == written["y"].attrs["standard_name"]


def test_netcdf_reads_a_file_url(tmp_path: Path) -> None:
    written = build_raster()
    netcdf.write(written, tmp_path / "scene.nc")

    with netcdf.open_raster((tmp_path / "scene.nc").as_uri()) as restored:
        np.testing.assert_array_equal(restored["red"], written["red"])


def test_netcdf_refuses_an_existing_destination(tmp_path: Path) -> None:
    destination = netcdf.write(build_raster(), tmp_path / "scene.nc").path

    with pytest.raises(FileExistsError):
        netcdf.write(build_raster(), destination)

    assert netcdf.write(build_raster(), destination, overwrite=True).path == destination


def test_netcdf_refuses_a_foreign_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in"):
        netcdf.write(build_raster(), tmp_path / "scene.tif")


def test_a_deferred_file_is_written_only_once_computed(tmp_path: Path) -> None:
    destination = tmp_path / "deferred.nc"

    pending = build_raster().gs.to_netcdf(destination, compute=False)

    assert not destination.exists()
    (file,) = pending.compute().written.values()
    assert file.path == destination
    assert file.header.gs.geobox == build_raster().gs.geobox


def test_netcdf_uploads_but_does_not_read_remotely(bucket: str, tmp_path) -> None:

    from geosave.geodata import open_raster

    written = build_raster(times=2)

    (file,) = written.gs.to_netcdf(
        f"{bucket}/scene.nc", storage_options={}
    ).written.values()
    saved = file.path

    assert saved == UPath(f"{bucket}/scene.nc")
    saved.fs.get(saved.path, str(tmp_path / "copy.nc"))
    with open_raster(tmp_path / "copy.nc") as restored:
        np.testing.assert_array_equal(restored.red, written.red)
    with pytest.raises(ValueError, match="local"):
        open_raster(saved)
    # A deferred upload stages the whole file when computed.
    pending = written.gs.to_netcdf(f"{bucket}/later.nc", compute=False)
    (file,) = pending.compute().written.values()
    assert file.path == UPath(f"{bucket}/later.nc")


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

    restored = netcdf.open_raster(tmp_path / "foreign.nc", chunks={})

    assert restored.red.dims == ("y", "x")
    assert restored.gs.geobox == foreign.odc.geobox
    assert restored.red.chunks is not None
    restored.close()
