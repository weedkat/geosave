"""Every raster format reads back the Dataset that was written to it."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata import attrs, io
from tests.geodata.conftest import build_raster


def build_described_raster() -> xr.Dataset:
    """Build one observation carrying every kind of metadata GeoSave states.

    Returns:
        Raster holding `red`, `nir` and `blue` over one instant, with packing,
        an RGB display, a title, a time cell, units and per-band attrs.
    """
    raster = build_raster(times=1).gs.write_packing(
        attrs.CFPacking(scale_factor=1e-4, fill_value=0)
    )
    raster = raster.assign(blue=raster.red)
    raster = attrs.stamp_attrs(raster, attrs.Display(rgb=["red", "nir", "blue"]))
    raster = raster.assign_attrs(title="Forest")
    raster = raster.assign_coords(
        start_datetime=("time", raster.time.values),
        end_datetime=("time", raster.time.values + np.timedelta64(1, "D")),
    )
    for name in raster.data_vars:
        raster[name].attrs["units"] = "1"
    raster.red.attrs.update(long_name="Red reflectance", wavelength="0.665")
    return raster


def assert_same_raster(restored: xr.Dataset, written: xr.Dataset) -> None:
    """Assert two rasters state the same variables, grid, coords, attrs and pixels."""
    assert list(restored.data_vars) == list(written.data_vars)
    assert set(restored.coords) == set(written.coords)
    assert restored.gs.geobox == written.gs.geobox
    assert restored.attrs == written.attrs
    for name in written.coords:
        assert restored[name].attrs == written[name].attrs, name
        np.testing.assert_array_equal(restored[name], written[name])
    for name in written.data_vars:
        assert restored[name].attrs == written[name].attrs, name
        assert restored[name].dtype == written[name].dtype
        np.testing.assert_array_equal(restored[name], written[name])


@pytest.mark.parametrize(
    ("write", "suffix"), [(io.write_zarr, ".zarr"), (io.write_netcdf, ".nc")]
)
def test_a_store_reads_back_the_raster_written(tmp_path, write, suffix) -> None:
    written = build_described_raster()

    with io.open_raster(write(written, tmp_path / f"scene{suffix}")) as restored:
        assert_same_raster(restored, written)


@pytest.mark.parametrize("write", [io.write_geotiff, io.write_cog])
def test_a_geotiff_reads_back_the_observation_written(tmp_path, write) -> None:
    written = build_described_raster()

    restored = io.open_raster(write(written, tmp_path / "scene.tif"))

    # GDAL states the pixel-area convention in the file's own tags.
    assert restored.attrs.pop("AREA_OR_POINT") == "Area"
    # A GeoTIFF holds one instant, so its time dim comes back as scalar coords.
    assert_same_raster(restored, written.squeeze("time"))


@pytest.mark.parametrize(
    ("write", "suffix"), [(io.write_zarr_tree, ".zarr"), (io.write_netcdf_tree, ".nc")]
)
def test_a_store_reads_back_the_tree_written(tmp_path, write, suffix) -> None:
    written = xr.DataTree.from_dict(
        {"/optical": build_described_raster(), "/radar": build_raster(times=1)}
    )

    with io.open_raster_tree(write(written, tmp_path / f"scene{suffix}")) as restored:
        assert list(restored.children) == ["optical", "radar"]
        for name in written.children:
            assert_same_raster(restored[name].to_dataset(), written[name].to_dataset())
