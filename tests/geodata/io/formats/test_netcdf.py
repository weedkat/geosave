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


from geosave.geodata import io


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
    expected = "latitude" if crs == "EPSG:4326" else "projection_y_coordinate"
    assert restored["y"].attrs["standard_name"] == expected


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


def test_netcdf_preserves_foreign_cf_coordinate_names(tmp_path: Path) -> None:
    foreign = _foreign()
    foreign.to_netcdf(tmp_path / "foreign.nc")

    restored = netcdf.open_netcdf(tmp_path / "foreign.nc", chunks={})

    assert restored.red.dims == ("latitude", "longitude")
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
        assert list(restored.gs.datasets) == ["s2/r10m", "s2/r20m"]
        for name, raster in tree.gs.datasets.items():
            loaded = restored.gs.datasets[name]
            assert loaded.gs.geobox == raster.gs.geobox
            np.testing.assert_array_equal(loaded.time.values, raster.time.values)


@pytest.mark.parametrize("scalar", [False, True])
def test_netcdf_stores_cells_as_cf_bounds_and_reads_them_back(tmp_path, scalar):
    import netCDF4

    from tests.geodata.conftest import build_dated_raster

    raster = build_dated_raster()
    raster = raster.isel(time=0) if scalar else raster
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


@pytest.mark.parametrize("suffix", ["nc"])
def test_tree_bounds_roundtrip_as_start_and_end(tmp_path, suffix):
    from geosave.geodata.io import open_raster

    time = np.array(["2025-01-15"], dtype="datetime64[ns]")
    root = xr.Dataset(
        {
            "cells": (
                ("time", "vertices"),
                np.array([["2025-01-01", "2025-02-01"]], dtype="datetime64[ns]"),
            )
        },
        coords={"time": ("time", time, {"standard_name": "time", "bounds": "cells"})},
    )
    tree = xr.DataTree.from_dict(
        {".": root, "image": build_raster().expand_dims(time=time)}
    )
    path = tmp_path / f"tree.{suffix}"
    if suffix == "nc":
        tree.gs.to_netcdf(path)
    else:
        tree.gs.to_zarr(path)
    with open_raster(path) as restored:
        assert list(restored.gs.datasets) == ["image"]
        assert restored.gs.timespan[0].day == 1
        image = restored.gs.datasets["image"]
        # Stored CF bounds come back as the cell coordinates every reader uses.
        assert "cells" not in image.variables
        np.testing.assert_array_equal(
            image.start_datetime.values, root.cells.values[:, 0]
        )
        np.testing.assert_array_equal(
            image.end_datetime.values, root.cells.values[:, 1]
        )


def test_explicit_output_encoding_uses_current_physical_values(tmp_path):
    source = build_raster(packed=True)
    result = source.gs.mask_and_scale() * 2
    path = tmp_path / "new.nc"
    settings = {}
    for name in result.data_vars:
        settings[name] = {
            "dtype": "uint16",
            "scale_factor": 0.0002,
            "_FillValue": 65535,
        }

    netcdf.write_netcdf(result, path, encoding=settings)
    with io.open_raster(path, mask_and_scale=True) as restored:
        xr.testing.assert_allclose(restored, result)
