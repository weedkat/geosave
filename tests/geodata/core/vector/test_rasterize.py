"""Converting between geolocated rasters and vector features."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from odc.geo.geobox import GeoBox
from geosave.geodata.utils.geometry import to_shapely


def _sliced_geographic_scene():
    """Build a raster whose sliced coordinates drift from its geobox's own."""

    from tests.conftest import build_dataset

    step = 1 / 3600
    grid = GeoBox.from_bbox(
        (13.0, 45.0, 13.0 + 40 * step, 45.0 + 40 * step),
        crs="EPSG:4326",
        resolution=step,
    )
    whole = build_dataset(
        {"red": (("y", "x"), np.ones(grid.shape, "uint16"))}, grid, nodata=0
    )
    return whole.isel(x=slice(5, 25), y=slice(5, 25))


def test_rasterize_lands_on_the_target_rasters_own_coordinates() -> None:
    import shapely
    import xarray as xr

    scene = _sliced_geographic_scene()
    left, bottom, right, top = scene.gs.geobox.boundingbox.bbox
    plot = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, (left + right) / 2, top)],
        crs="EPSG:4326",
    )

    burned = plot.gs.rasterize(scene)

    xr.align(scene, burned, join="exact")
    masked = scene.gs.mask(burned)
    assert (masked.red.values == 0).any() and (masked.red.values == 1).any()


def test_a_presence_mask_refuses_a_truthy_fill() -> None:
    import shapely

    scene = _sliced_geographic_scene()
    left, bottom, right, top = scene.gs.geobox.boundingbox.bbox
    plot = gpd.GeoDataFrame(
        geometry=[shapely.box(left, bottom, right, top)], crs="EPSG:4326"
    )

    with pytest.raises(ValueError, match="presence mask"):
        plot.gs.rasterize(scene, fill=1)


def test_rasterizing_onto_an_explicit_grid_is_undated(grid):
    plot = gpd.GeoDataFrame(geometry=[grid.extent.geom], crs=grid.crs)
    burned = plot.gs.rasterize(grid)
    assert burned.gs.geobox == grid
    assert "time" not in burned.coords


@pytest.mark.parametrize("problem", ["null", "empty"])
def test_rasterize_refuses_a_missing_geometry(raster, problem: str) -> None:
    geometry = None if problem == "null" else Polygon()
    vector = gpd.GeoDataFrame(geometry=[geometry], crs=raster.gs.crs)
    with pytest.raises(ValueError, match="position 0.*(missing|empty)"):
        vector.gs.rasterize(raster)


def test_rasterize_refuses_a_geometry_naming_no_ground(raster) -> None:
    bowtie = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])], crs=raster.gs.crs
    )
    with pytest.raises(ValueError, match="position 0.*invalid"):
        bowtie.gs.rasterize(raster)


def test_rasterize_without_column_returns_boolean_presence(raster) -> None:
    vector = gpd.GeoDataFrame(
        geometry=[to_shapely(raster.gs.geobox.extent)], crs=raster.gs.geobox.extent.crs
    )
    mask = vector.gs.rasterize(like=raster)
    assert mask.dtype == np.dtype("bool")
    assert mask.values.all()


def test_rasterize_later_rows_win_on_overlap(raster) -> None:
    bounds = raster.gs.geobox.extent.geom
    vector = gpd.GeoDataFrame(
        {"class_id": [1, 2]}, geometry=[bounds, bounds], crs=raster.gs.crs
    )
    result = vector.gs.rasterize(like=raster, column="class_id", dtype="uint8")
    assert (result.values == 2).all()


def test_rasterize_rejects_values_outside_dtype(raster) -> None:
    vector = gpd.GeoDataFrame(
        [{"class_id": 300}],
        geometry=[to_shapely(raster.gs.geobox.extent)],
        crs=raster.gs.geobox.extent.crs,
    )
    with pytest.raises(ValueError, match="uint8"):
        vector.gs.rasterize(like=raster, column="class_id", dtype="uint8")


def test_rasterize_rejects_non_numeric_property(raster) -> None:
    vector = gpd.GeoDataFrame(
        [{"label": "forest"}],
        geometry=[to_shapely(raster.gs.geobox.extent)],
        crs=raster.gs.geobox.extent.crs,
    )
    with pytest.raises(ValueError, match="numeric raster dtype"):
        vector.gs.rasterize(like=raster, column="label")


def test_rasterize_empty_vector_returns_only_fill(raster) -> None:
    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=raster.gs.crs))
    result = empty.gs.rasterize(like=raster, fill=0)
    assert not result.values.any()
