from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from upath import UPath

from geosave.geodata import (
    open_raster,
)
from geosave.geodata.io.raster import netcdf
from tests.geodata.conftest import build_raster


@pytest.mark.parametrize("crs", ["EPSG:32749", "EPSG:4326"])
def test_netcdf_round_trip_preserves_the_profile(tmp_path: Path, crs: str) -> None:
    times = 2
    written = build_raster(crs=crs, times=times, packed=True)

    destination = netcdf.write_netcdf(written, tmp_path / "scene.nc")
    restored = netcdf.open_netcdf(destination)

    assert restored.gs.geobox == written.gs.geobox
    assert restored.red.dtype == np.dtype("uint16")
    assert int(restored.red.values.ravel()[0]) == 1000
    # The file states the CF axes; reading hands back the attrs as written.
    expected = "latitude" if crs == "EPSG:4326" else "projection_y_coordinate"
    with xr.open_dataset(destination, engine="h5netcdf") as stored:
        assert stored["y"].attrs["standard_name"] == expected
    assert restored["y"].attrs == written["y"].attrs


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


def test_netcdf_preserves_foreign_cf_coordinate_names(tmp_path: Path) -> None:
    foreign = _foreign()
    foreign.to_netcdf(tmp_path / "foreign.nc")

    restored = netcdf.open_netcdf(tmp_path / "foreign.nc", chunks={})

    assert restored.red.dims == ("latitude", "longitude")
    assert restored.gs.geobox == foreign.odc.geobox
    assert restored.red.chunks is not None
    restored.close()


def test_a_remote_netcdf_reads_lazily_through_its_filesystem(bucket):
    source = build_raster(times=1)
    saved = source.gs.to_netcdf(f"{bucket}/lifetime.nc")

    restored = open_raster(saved, chunks={})

    assert restored.red.chunks is not None
    np.testing.assert_array_equal(restored.red, source.red)
    restored.close()
    with pytest.raises(FileNotFoundError):
        open_raster(f"{bucket}/missing.nc")


def test_a_tree_round_trips_through_one_netcdf_file(tmp_path):
    from geosave.geodata import open_raster_tree
    from tests.geodata.core.raster.test_datatree import build_stac_catalog

    tree = build_stac_catalog()["s2"].copy()

    path = tree.gs.to_netcdf(tmp_path / "scene.nc")

    assert path == tmp_path / "scene.nc"
    with open_raster_tree(path) as restored:
        assert restored.isomorphic(tree)
        for leaf in tree.leaves:
            raster = leaf.to_dataset(inherit="all_coords")
            loaded = restored[leaf.relative_to(tree)].to_dataset(inherit="all_coords")
            assert loaded.gs.geobox == raster.gs.geobox
            np.testing.assert_array_equal(loaded.time.values, raster.time.values)


def test_netcdf_stores_cells_as_cf_bounds_and_reads_them_back(tmp_path):
    import netCDF4

    from tests.geodata.conftest import build_dated_raster

    raster = build_dated_raster()
    path = netcdf.write_netcdf(raster, tmp_path / "a.nc")

    with netCDF4.Dataset(path) as raw:
        assert raw["time"].bounds == "time_bounds"
        assert "start_datetime" not in raw.variables
        bounds_units = getattr(raw["time_bounds"], "units", raw["time"].units)
        assert bounds_units == raw["time"].units
        assert "_FillValue" not in raw["x"].ncattrs()
        assert "_FillValue" not in raw["y"].ncattrs()
    back = netcdf.open_netcdf(path)
    for name in ("time", "start_datetime", "end_datetime"):
        np.testing.assert_array_equal(back[name].values, raster[name].values)
    assert "time_bounds" not in back.variables
    assert "bounds" not in back.dims


def test_a_foreign_cf_file_decodes_to_start_and_end(tmp_path):
    from tests.geodata.conftest import build_raster

    foreign = build_raster(times=2).cf.add_bounds("time")
    foreign = foreign.rename_vars(time_bounds="time_bnds").rename_dims(bounds="nv")
    foreign.time.attrs["bounds"] = "time_bnds"
    foreign.to_netcdf(tmp_path / "foreign.nc")

    back = netcdf.open_netcdf(tmp_path / "foreign.nc")

    assert {"start_datetime", "end_datetime"} <= set(back.coords)
    assert "nv" not in back.dims


def test_closing_an_empty_group_releases_its_backend_and_file(tmp_path, monkeypatch):
    path = tmp_path / "empty.nc"
    path.touch()
    closed = []
    handles = []
    stored = xr.Dataset()
    stored.set_close(lambda: closed.append("backend"))

    def open_empty(handle, **options):
        handles.append(handle)
        return stored

    monkeypatch.setattr(netcdf.xr, "open_dataset", open_empty)

    with netcdf.open_netcdf(path) as restored:
        assert not restored.data_vars
        assert closed == []
        assert not handles[0].closed

    assert closed == ["backend"]
    assert handles[0].closed
    restored.close()
    assert closed == ["backend"]
