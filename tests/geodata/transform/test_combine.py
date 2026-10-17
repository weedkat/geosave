"""Combining several rasters into one."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave.geodata.core.raster.create import raster as build_raster
from geosave.geodata.transform.combine import merge
from geosave.geodata.warnings import DroppedAttrsWarning

from .conftest import build, geobox


def test_merge_preserves_band_order_values_and_metadata():
    optical = build(geobox(), labelled=True)
    radar = build(geobox()).rename_vars({"red": "VV"})
    radar["VV"].values[:] = 300
    result = merge([radar, optical])
    assert tuple(result.data_vars) == ("VV", "red", "cls")
    np.testing.assert_array_equal(result.red, optical.red)
    np.testing.assert_array_equal(result.VV, radar.VV)
    assert result.cls.attrs == optical.cls.attrs
    assert result.red.attrs == optical.red.attrs
    assert result.gs.geobox == optical.gs.geobox


@pytest.mark.parametrize(
    "kind", ["grid", "crs", "time", "scalar_time", "timeless", "collision", "dims"]
)
def test_merge_rejects_implicit_alignment_or_broadcasting(kind):
    first = build(geobox())
    second = first.rename_vars({"red": "nir"})
    if kind == "grid":
        second = second.assign_coords(x=second.x + 5)
    elif kind == "crs":
        second = build(geobox(crs="EPSG:32632")).rename_vars({"red": "nir"})
    elif kind == "time":
        second = build(geobox(), start="2025-01-01").rename_vars({"red": "nir"})
    elif kind == "scalar_time":
        first, second = first.isel(time=0), second.isel(time=1)
    elif kind == "timeless":
        second = second.isel(time=0, drop=True)
    elif kind == "collision":
        second = first
    else:
        second = second.transpose("y", "x", "time")
    with pytest.raises(ValueError):
        merge([first, second])


@pytest.mark.parametrize("rasters", [[], [xr.Dataset()]])
def test_merge_requires_nonempty_rasters(rasters):
    with pytest.raises(ValueError, match="non-empty"):
        merge(rasters)


def test_merge_ignores_the_grid_mappings_placeholder_value() -> None:
    # Readers disagree on the CF placeholder: rioxarray stores 0, odc-geo the EPSG code.
    box = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32749", resolution=10)
    red = build_raster({"red": (("y", "x"), np.ones((2, 2), "uint16"))}, box)
    nir = build_raster({"nir": (("y", "x"), np.ones((2, 2), "uint16"))}, box)
    red = red.assign_coords(spatial_ref=red.spatial_ref.copy(data=0))
    nir = nir.assign_coords(spatial_ref=nir.spatial_ref.copy(data=32749))

    merged = merge([red, nir])

    assert list(merged.data_vars) == ["red", "nir"]
    assert merged.gs.geobox == box


def test_merge_keeps_one_spelling_of_the_grid_coordinates() -> None:
    # A band sliced from a file has bare x/y; one odc-geo warped has CF attrs.
    box = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32749", resolution=10)
    red = build_raster({"red": (("y", "x"), np.ones((2, 2), "uint16"))}, box)
    nir = build_raster({"nir": (("y", "x"), np.ones((2, 2), "uint16"))}, box)
    red = red.copy()
    red.x.attrs, red.y.attrs = {}, {}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DroppedAttrsWarning)
        merged = merge([red, nir])

    assert merged.x.attrs["axis"] == "X"
    assert merged.y.attrs["standard_name"] == "projection_y_coordinate"
