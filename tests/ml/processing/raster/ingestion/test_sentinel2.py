"""Sentinel-2 ingestion keeps one processing per acquisition and decodes by baseline."""

from datetime import datetime

import pystac
import pytest

from geosave.ml.processing.raster.ingestion.sentinel2 import (
    drop_superseded,
    read_esa_packing,
)


def _item(
    item_id: str,
    baseline: str = "05.00",
    tile: str = "48MXU",
    generated: str = "2020-01-01T12:00:00Z",
) -> pystac.Item:
    return pystac.Item(
        item_id,
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 7, 1),
        properties={
            "s2:processing_baseline": baseline,
            "platform": "sentinel-2a",
            "s2:mgrs_tile": tile,
            "s2:generation_time": generated,
        },
    )


def test_only_the_newest_processing_of_an_acquisition_is_kept():
    original = _item("original")
    reprocessed = _item("reprocessed", generated="2023-01-01T12:00:00Z")
    other = _item("other", tile="48MXV")

    kept = drop_superseded([reprocessed, original, other], "s2:mgrs_tile")

    assert kept == [reprocessed, other]


@pytest.mark.parametrize(("baseline", "offset"), [("05.00", -0.1), ("02.12", 0.0)])
def test_the_processing_baseline_sets_the_reflectance_offset(baseline, offset):
    packing = read_esa_packing([_item("a", baseline)])

    assert (packing.scale_factor, packing.add_offset) == (1e-4, offset)


def test_tiles_of_mixed_baselines_are_refused():
    with pytest.raises(ValueError, match="mix processing baselines"):
        read_esa_packing([_item("a", "05.00"), _item("b", "02.12")])


@pytest.mark.parametrize(
    ("baselines", "offset"),
    [
        (("02.12", "03.00"), 0.0),
        (("04.00", "05.00"), -0.1),
    ],
)
def test_tiles_with_the_same_decoding_can_mosaic(baselines, offset):
    packing = read_esa_packing(
        [
            _item("first", baselines[0]),
            _item("second", baselines[1]),
        ]
    )

    assert packing.scale_factor == 1e-4
    assert packing.add_offset == offset
    assert packing.fill_value == 0


def test_scl_keeps_classes_when_resampled_and_saved(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import xarray as xr
    import pandas as pd
    import rasterio
    from odc.geo.geobox import GeoBox
    from rasterio.transform import from_origin

    import geosave as gs
    from geosave.geodata import GeoRegion
    from geosave.geodata.features import scl_valid_mask
    from geosave.ml.processing.raster.ingestion.sentinel2 import Sentinel2Ingestor

    transform = from_origin(500000, 10000, 10, 10)
    grid = GeoBox((4, 4), transform, "EPSG:32649")
    footprint = grid.extent.to_crs("EPSG:4326").geom
    item = pystac.Item(
        "local",
        footprint.__geo_interface__,
        list(footprint.bounds),
        datetime(2024, 12, 31),
        {"s2:processing_baseline": "05.00"},
    )
    classes = np.array([[4, 9], [3, 11]], dtype="uint16")
    for name, pixels, resolution in [
        ("B04", np.full((4, 4), 2000, dtype="uint16"), 10),
        ("SCL", classes, 20),
    ]:
        path = tmp_path / f"{name}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            count=1,
            dtype="uint16",
            height=pixels.shape[0],
            width=pixels.shape[1],
            crs="EPSG:32649",
            transform=from_origin(500000, 10000, resolution, resolution),
        ) as dst:
            dst.write(pixels, 1)
        item.add_asset(
            name, pystac.Asset(str(path), media_type=pystac.MediaType.GEOTIFF)
        )

    ingestor = Sentinel2Ingestor(["B04"], scl="scl")
    ingestor.__dict__["client"] = SimpleNamespace(patch_url=None)
    ingestor.search_items = lambda region: [item]
    rasters = ingestor.ingest_grid(grid, pd.Timestamp("2024-12-31"), items=[item])
    assert list(rasters.children) == ["image", "scl"]
    image = rasters["image"].to_dataset()
    scl = rasters["scl"].to_dataset()
    assert list(image.data_vars) == ["B04"]
    expected = classes.repeat(2, axis=0).repeat(2, axis=1)
    np.testing.assert_array_equal(scl.SCL.values, expected)
    assert image.B04.attrs["scale_factor"] == 1e-4
    assert image.B04.attrs["add_offset"] == -0.1
    assert scl.SCL.attrs.get("scale_factor", 1) == 1
    assert scl.SCL.attrs.get("add_offset", 0) == 0
    image = xr.merge([image, scl])

    # Reflectance and categorical codes are independent file representations.
    from geosave.geodata import stac

    tree = image.gs.to_tree("s2", groups={"image": ["B04"], "scl": ["SCL"]})
    template = stac.StacItem.from_region(image.gs.region, id="observation")
    published = stac.publish_item(
        template,
        assets={key: node.gs.to_dataset() for key, node in tree["s2"].children.items()},
        root=tmp_path / "published",
    )
    with gs.open_raster(published.assets["scl"].href) as restored:
        np.testing.assert_array_equal(restored.SCL.values, expected)
        np.testing.assert_array_equal(restored.gs.mask_and_scale().SCL.values, expected)
        np.testing.assert_array_equal(
            scl_valid_mask(restored, scl="SCL"), np.isin(expected, [4, 11])
        )
    with gs.open_raster(published.assets["image"].href) as restored:
        pixels, context = restored.gs.to_tensor(
            context=["centroid", "year", "day_of_year"]
        )
        assert context["year"].tolist() == [2024]
        assert context["day_of_year"].tolist() == [366]
        assert context["centroid"].shape == (2,)

    # Region ingestion uses the same categorical resampling and packing rules.
    region = GeoRegion.from_geometry(grid.extent, timespan="2024-12-31")
    trees = ingestor(region)
    loaded = trees[0]["scl"].to_dataset()
    np.testing.assert_array_equal(loaded.SCL.values, expected)
    assert loaded.SCL.attrs.get("scale_factor", 1) == 1


def test_reflectance_bands_exclude_scl():
    from geosave.ml.processing.raster.ingestion.sentinel2 import Sentinel2Ingestor

    with pytest.raises(ValueError, match="not a Sentinel-2 reflectance band"):
        Sentinel2Ingestor(["B04", "SCL"])


def test_unknown_catalog_is_refused():
    from geosave.ml.processing.raster.ingestion.sentinel2 import Sentinel2Ingestor

    with pytest.raises(ValueError, match="unknown catalog"):
        Sentinel2Ingestor(["B04"], catalog="cdse")
