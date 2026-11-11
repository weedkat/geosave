"""Bulk model acquisition using real local STAC assets."""

import dask.array as da
from dask.callbacks import Callback
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point
from pathlib import Path

from geosave.geodata import GeoRegion
from geosave.model.ingestion import ingest_stac, regions_from_features
from tests.geodata.stac.conftest import local_catalog as local_catalog


def test_saved_bulk_recipe_ingests_encodes_and_normalizes_once(local_catalog, tmp_path):
    import torch
    from geosave.model.spec import ModelSpec

    catalog, region, grid, options = local_catalog
    path = Path(__file__).parents[2] / "docs/guides/model_spec_bulk.yaml"
    spec = ModelSpec.load(path)
    saved = tmp_path / "bulk.yaml"
    spec.save(saved)
    result = ModelSpec.load(saved).run(
        regions=[region, region],
        catalog=catalog,
        collection="local",
        crs=grid.crs,
        resolution=10,
        bands=["red", "nir"],
        mean=[0.1, 0.5],
        std=[0.1, 0.3],
        stac_cfg=options["stac_cfg"],
    )
    assert result.shape == (2, 2, 1, 2, 2)
    assert result[:, :, :, 0, 0].isnan().all()
    torch.testing.assert_close(result[:, 0, :, 1, :], torch.ones((2, 1, 2)))
    torch.testing.assert_close(result[:, 1, :, 1, :], torch.full((2, 1, 2), 2.0))


def test_regions_preserve_features_crs_and_order():
    features = gpd.GeoDataFrame(
        geometry=[Point(500010, 5000010), Point(500020, 5000020)],
        crs="EPSG:32633",
        index=[7, 7],
    )
    regions = regions_from_features(features, timespan="2025-01")
    assert len(regions) == 2
    assert regions[0].extent.coords[0] == (500010, 5000010)
    assert regions[1].extent.coords[0] == (500020, 5000020)
    assert regions[0].crs.epsg == 32633
    assert regions[0].timespan == regions[1].timespan

    features.iloc[1, 0] = None
    with pytest.raises(ValueError, match="position 1"):
        regions_from_features(features)


def test_ingestion_keeps_order_grids_and_lazy_pixels(local_catalog):
    catalog, region, grid, options = local_catalog
    left = GeoRegion(grid[:, :1].extent, region.timespan)
    right = GeoRegion(grid[:, 1:].extent, region.timespan)
    reads = []
    with Callback(posttask=lambda *args: reads.append(1)):
        rasters = ingest_stac(
            [right, left],
            catalog=catalog,
            collection="local",
            crs=grid.crs,
            resolution=10,
            **options,
        )

    assert reads == []
    assert len(rasters) == 2
    assert rasters[0].gs.geobox == grid[:, 1:]
    assert rasters[1].gs.geobox == grid[:, :1]
    assert isinstance(rasters[0].red.data, da.Array)
    np.testing.assert_array_equal(rasters[0].red.values[0], [[2000], [2000]])
    np.testing.assert_array_equal(rasters[1].red.values[0], [[0], [2000]])
    assert rasters[0].red.attrs["scale_factor"] == 0.0001


def test_point_ingestion_uses_exact_window_and_missing_dates_fail(local_catalog):
    catalog, region, grid, options = local_catalog
    point = GeoRegion.from_geometry(grid.extent.centroid, timespan="2025-01")
    (raster,) = ingest_stac(
        [point],
        catalog=catalog,
        collection="local",
        crs=grid.crs,
        resolution=10,
        shape=(2, 2),
        **options,
    )
    assert raster.gs.geobox == grid
    np.testing.assert_array_equal(raster.red.values[0], [[0, 2000], [2000, 2000]])

    missing = region.replace(timespan="2026-01")
    with pytest.raises(ValueError, match="position 1.*local.*2026"):
        ingest_stac(
            [region, missing],
            catalog=catalog,
            collection="local",
            crs=grid.crs,
            resolution=10,
            **options,
        )

    with pytest.raises(ValueError, match="position 0.*Point regions need shape"):
        ingest_stac(
            [point],
            catalog=catalog,
            collection="local",
            crs=grid.crs,
            resolution=10,
        )
