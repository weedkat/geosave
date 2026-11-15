from __future__ import annotations

from pathlib import Path

import dask
import numpy as np
import orjson
import pytest
import xarray as xr
from upath import UPath

from geosave.geodata import open_raster, open_raster_tree
from geosave.geodata.io.raster import zarr
from tests.geodata.conftest import _refuse, build_raster


@pytest.mark.parametrize("crs", ["EPSG:32749", "EPSG:4326"])
def test_zarr_round_trip_preserves_the_profile(tmp_path: Path, crs: str) -> None:
    times = 2
    written = build_raster(crs=crs, times=times, packed=True)

    destination = zarr.write_zarr(written, tmp_path / "scene.zarr")
    restored = zarr.open_zarr(destination)

    assert restored.gs.geobox == written.gs.geobox
    assert restored.red.dtype == np.dtype("uint16")
    assert int(restored.red.values.ravel()[0]) == 1000
    assert restored.red.dims[-2:] == ("y", "x")
    # The store states the CF axes; reading hands back the attrs as written.
    expected = "latitude" if crs == "EPSG:4326" else "projection_y_coordinate"
    stored = xr.open_zarr(destination, consolidated=False)
    assert stored["y"].attrs["standard_name"] == expected
    assert stored["x"].attrs["axis"] == "X"
    assert restored["y"].attrs == written["y"].attrs
    assert restored["x"].attrs == written["x"].attrs
    if times:
        assert restored.sizes["time"] == times


def build_shuffled(names: list[str]) -> xr.Dataset:
    """Build a raster whose variable order is neither sorted nor reversed.

    Args:
        names: Variable names, in the order the raster should carry them.

    Returns:
        Raster Dataset holding one `uint16` variable per name.
    """
    base = build_raster()
    return xr.Dataset({name: base.red for name in names}, coords=base.coords)


def test_zarr_reads_its_variables_in_the_order_they_were_written(
    tmp_path: Path,
) -> None:
    # Six names scramble on every read; two happen to come back written order.
    names = ["zulu", "alpha", "mike", "bravo", "yankee", "charlie"]

    destination = zarr.write_zarr(build_shuffled(names), tmp_path / "ordered.zarr")

    # A Zarr group lists its members in no order, so every read must agree.
    for _ in range(3):
        assert list(zarr.open_zarr(destination).data_vars) == names


def test_digital_numbers_stay_stored_and_decode_on_request(tmp_path: Path) -> None:
    destination = zarr.write_zarr(build_raster(packed=True), tmp_path / "packed.zarr")

    stored = zarr.open_zarr(destination)
    decoded = zarr.open_zarr(destination, mask_and_scale=True)

    assert stored.red.dtype == np.dtype("uint16")
    assert stored.red.attrs["scale_factor"] == pytest.approx(1e-4)
    assert stored.red.attrs["_FillValue"] == 0
    assert "scale_factor" not in decoded.red.attrs
    assert "_FillValue" not in decoded.red.attrs
    assert decoded.red.encoding["scale_factor"] == pytest.approx(1e-4)
    assert float(decoded.red.values.ravel()[0]) == pytest.approx(0.1)
    assert bool(np.isnan(decoded.red.values.ravel()[-1]))


def test_overwrite_guards_an_existing_store(tmp_path: Path) -> None:
    raster = build_raster()
    destination = tmp_path / "scene.zarr"
    zarr.write_zarr(raster, destination)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        zarr.write_zarr(raster, destination)

    assert zarr.write_zarr(raster, destination, overwrite=True) == destination


def test_a_zarr_array_holds_the_fill_value_a_gdal_reader_masks_on(
    tmp_path: Path,
) -> None:
    written = build_raster().gs.write_nodata(255)

    destination = zarr.write_zarr(written, tmp_path / "scene.zarr")

    # A CF attr is not the array's own fill value, and GDAL masks on the latter.
    stored = orjson.loads((destination / "red" / "zarr.json").read_bytes())
    assert stored["fill_value"] == 255
    assert zarr.open_zarr(destination).red.attrs["_FillValue"] == 255


def test_a_raster_round_trips_through_a_remote_store(
    bucket: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    written = build_raster(times=2)
    destination = f"{bucket}/scene.zarr"

    # zarr maps `memory://` to its own store and refuses unused storage_options.
    saved = written.gs.to_zarr(destination)

    assert saved == UPath(destination)
    assert list(tmp_path.iterdir()) == []
    local = written.gs.to_zarr(tmp_path / "local.zarr")
    with open_raster(saved) as restored, open_raster(local) as expected:
        xr.testing.assert_identical(restored, expected)


@pytest.mark.integration
def test_a_zarr_store_round_trips_lazily(remote) -> None:
    prefix, options = remote
    source = build_raster(times=2)

    store = source.gs.to_zarr(f"{prefix}/forest.zarr", storage_options=options)

    with dask.config.set(scheduler=_refuse):
        restored = open_raster(store, chunks={}, storage_options=options)
    np.testing.assert_array_equal(restored.red.values, source.red.values)
    restored.close()


def test_a_tree_round_trips_through_one_zarr_store(tmp_path):
    from tests.geodata.core.raster.test_datatree import build_stac_catalog

    tree = build_stac_catalog()["s2"].copy()
    tree.attrs["title"] = "Sentinel-2"

    store = tree.gs.to_zarr(tmp_path / "scene.zarr")

    assert store == tmp_path / "scene.zarr"
    with open_raster_tree(store) as restored:
        assert restored.isomorphic(tree)
        assert restored.attrs["title"] == "Sentinel-2"
        for leaf in tree.leaves:
            raster = leaf.to_dataset(inherit="all_coords")
            loaded = restored[leaf.relative_to(tree)].to_dataset(inherit="all_coords")
            assert loaded.gs.geobox == raster.gs.geobox
            np.testing.assert_array_equal(loaded.time.values, raster.time.values)
            xr.testing.assert_equal(
                loaded[list(raster.data_vars)], raster[list(raster.data_vars)]
            )


def test_zarr_stores_cells_as_start_and_end_coordinates(tmp_path):
    from tests.geodata.conftest import build_dated_raster

    raster = build_dated_raster()
    path = zarr.write_zarr(raster, tmp_path / "a.zarr")

    assert (path / "start_datetime").exists()
    assert not (path / "time_bounds").exists()
    back = zarr.open_zarr(path)
    for name in ("time", "start_datetime", "end_datetime"):
        np.testing.assert_array_equal(back[name].values, raster[name].values)
    assert "time_bounds" not in back.variables
