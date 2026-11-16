"""Every ingestor writes its acquisitions to stores the same way."""

import pytest
import xarray as xr

import geosave  # noqa: F401  registers the .gs accessors
from geosave.geodata.io import open_raster_tree
from geosave.ml.processing.raster.ingestion.base import RasterIngestor
from tests.geodata.sampling.conftest import build_scene


class TwoDays(RasterIngestor):
    def __call__(self, region):
        scene = build_scene()
        return [
            xr.DataTree.from_dict({"image": scene.isel(time=0)}),
            xr.DataTree.from_dict({"image": scene.isel(time=1)}),
        ]


def test_a_region_is_ingested_to_one_lazy_store_per_acquisition(tmp_path):
    scene = build_scene()

    stores = TwoDays().ingest_to_zarr(region=None, destination=tmp_path)

    assert [store.name for store in stores] == [
        "20250603T000000.zarr",
        "20250613T000000.zarr",
    ]
    sources = open_raster_tree(stores[1])
    assert sources["image"]["red"].chunks is not None
    xr.testing.assert_equal(sources["image"].to_dataset(), scene.isel(time=1))
    assert sources["image"].to_dataset().gs.geobox == scene.gs.geobox


def test_an_ingestor_must_state_where_its_rasters_come_from():
    class Nowhere(RasterIngestor):
        pass

    with pytest.raises(TypeError, match="__call__"):
        Nowhere()
