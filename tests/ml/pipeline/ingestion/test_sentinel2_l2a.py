"""Sentinel-2 L2A ingestion states ESA's meaning under every provider's naming."""

from datetime import datetime

import dask
import numpy as np
import pystac
import pytest
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_origin

from geosave.geodata import GeoRegion
from geosave.ml.pipeline.ingestion.sentinel2_l2a import (
    SCL_CLASSES,
    Provider,
    Sentinel2L2AIngestor,
)

GRID = GeoBox((4, 4), from_origin(500000, 10000, 10, 10), "EPSG:32649")
CLASSES = np.array([[4, 9], [3, 11]], dtype="uint16")


def write_item(
    tmp_path,
    provider: Provider,
    *,
    baseline: str = "05.10",
    day: datetime = datetime(2024, 12, 31),
) -> pystac.Item:
    """Write 10 m reflectance and 20 m SCL under a provider's asset keys."""
    footprint = GRID.extent.to_crs("EPSG:4326").geom
    source = provider.value
    item = pystac.Item(
        f"{provider.name}-{day:%Y%m%d}",
        footprint.__geo_interface__,
        list(footprint.bounds),
        day,
        {source.baseline: baseline},
        collection=source.collection,
    )
    if provider is Provider.cdse:
        # CDSE states each acquisition's interval, not only its instant.
        item.properties.update(
            start_datetime=f"{day:%Y-%m-%d}T10:30:00Z",
            end_datetime=f"{day:%Y-%m-%d}T10:30:05Z",
        )
    for band, common_name in [
        ("B04", "red"),
        ("B03", "green"),
        ("B02", "blue"),
        ("SCL", None),
    ]:
        if common_name is None:
            pixels = CLASSES
            resolution = 20
            fields = {}
        else:
            pixels = np.full((4, 4), 2000, dtype="uint16")
            resolution = 10
            fields = {"eo:bands": [{"name": band, "common_name": common_name}]}
        key = source.assets[band]
        path = tmp_path / f"{item.id}-{key}.tif"
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
            key,
            pystac.Asset(
                str(path), media_type=pystac.MediaType.GEOTIFF, extra_fields=fields
            ),
        )
    return item


def create_ingestor(provider: Provider, **options) -> Sentinel2L2AIngestor:
    """Build an ingestor; loading local files never connects to its catalog."""
    return Sentinel2L2AIngestor(
        "s2",
        ["B04", "B03", "B02", "SCL"],
        groups={"image": ["B04", "B03", "B02"], "scl": ["SCL"]},
        provider=provider,
        chunks=2,
        **options,
    )


def refuse(*args, **kwargs):
    raise AssertionError("loading must stay lazy")


@pytest.mark.parametrize("provider", list(Provider))
def test_every_provider_loads_into_the_configured_tree(tmp_path, provider):
    ingestor = create_ingestor(provider)
    item = write_item(tmp_path, provider)

    with dask.config.set(scheduler=refuse):
        tree = ingestor.load([item], GRID)

    assert tree.groups == ("/", "/s2", "/s2/image", "/s2/scl")
    image = tree["s2/image"].to_dataset()
    scl = tree["s2/scl"].to_dataset()
    assert list(image.data_vars) == ["B04", "B03", "B02"]
    assert image.odc.geobox == GRID
    assert image.B04.chunks == ((1,), (2, 2), (2, 2))
    assert image.B04.attrs["common_name"] == "red"
    packing = image.B04.gs.packing
    assert (packing.scale_factor, packing.add_offset, packing.fill_value) == (
        1e-4,
        -0.1,
        0,
    )
    np.testing.assert_allclose(image.gs.mask_and_scale().B04.values, 0.1)

    # SCL keeps its class IDs when 20 m pixels land on the 10 m grid.
    assert scl.SCL.gs.flags.classes == SCL_CLASSES
    assert scl.SCL.gs.flags.colors[6] == "#0000FF"  # Water in the Sentinel-2 palette.
    expected = CLASSES.repeat(2, axis=0).repeat(2, axis=1)
    np.testing.assert_array_equal(scl.SCL.isel(time=0).values, expected)
    assert scl.SCL.attrs.get("scale_factor", 1) == 1


@pytest.mark.parametrize(
    ("baseline", "offset"), [("03.00", 0.0), ("04.00", -0.1), ("05.10", -0.1)]
)
def test_packing_follows_the_processing_baseline(tmp_path, baseline, offset):
    for provider in Provider:
        ingestor = create_ingestor(provider)
        item = write_item(tmp_path, provider, baseline=baseline)

        assert ingestor.read_packing([item]).add_offset == offset


def test_baselines_on_both_sides_of_04_00_are_refused(tmp_path):
    provider = Provider.planetary_computer
    ingestor = create_ingestor(provider)
    items = [
        write_item(tmp_path, provider, baseline="03.00", day=datetime(2022, 1, 20)),
        write_item(tmp_path, provider, baseline="04.00", day=datetime(2022, 1, 30)),
    ]

    with pytest.raises(ValueError, match="mix reflectance offsets"):
        ingestor.load(items, GRID)


def test_a_region_loads_one_step_per_day_on_the_configured_grid(tmp_path):
    provider = Provider.planetary_computer
    ingestor = create_ingestor(provider, resampling="nearest")
    items = [
        write_item(tmp_path, provider, day=datetime(2024, 12, 31)),
        write_item(tmp_path, provider, day=datetime(2025, 1, 1)),
    ]
    ingestor.search = lambda region: items
    region = GeoRegion.from_geometry(GRID.extent, timespan=("2024-12-31", "2025-01-01"))

    image = ingestor(region)["s2/image"].to_dataset()

    assert image.time.dt.strftime("%Y-%m-%d").values.tolist() == [
        "2024-12-31",
        "2025-01-01",
    ]
    assert image.odc.crs == "EPSG:32649"
    assert image.odc.geobox.resolution.x == 10


def test_bands_outside_the_product_are_refused():
    with pytest.raises(ValueError, match="not a Sentinel-2 L2A band"):
        Sentinel2L2AIngestor("s2", ["B04", "B10"])


def test_no_items_are_refused():
    ingestor = create_ingestor(Provider.planetary_computer)

    with pytest.raises(ValueError, match="no Sentinel-2 L2A acquisition"):
        ingestor.load([], GRID)


@pytest.mark.integration
@pytest.mark.parametrize("provider", [Provider.planetary_computer, Provider.element84])
def test_a_provider_ingests_a_region(provider):
    region = GeoRegion.from_bbox(
        (5.10, 52.05, 5.13, 52.08), timespan=("2023-07-31", "2023-07-31")
    )
    ingestor = Sentinel2L2AIngestor(
        "s2",
        ["B04", "B08", "SCL"],
        groups={"image": ["B04", "B08"], "scl": ["SCL"]},
        provider=provider,
    )

    tree = ingestor(region)

    image = tree["s2/image"].to_dataset()
    assert image.sizes["time"] == 1
    assert image.B04.gs.packing.add_offset == -0.1
    assert tree["s2/scl"].to_dataset().SCL.gs.flags.classes == SCL_CLASSES


def create_region_ingestor(tmp_path, **options) -> Sentinel2L2AIngestor:
    """An ingestor whose search returns two local acquisitions."""
    provider = Provider.planetary_computer
    ingestor = create_ingestor(provider, **options)
    items = [
        write_item(tmp_path, provider, day=datetime(2024, 12, 31)),
        write_item(tmp_path, provider, day=datetime(2025, 1, 1)),
    ]
    ingestor.search = lambda region: items
    return ingestor


REGION = GeoRegion.from_geometry(GRID.extent, timespan=("2024-12-31", "2025-01-01"))


def test_without_groups_the_collection_is_one_node(tmp_path):
    provider = Provider.planetary_computer
    ingestor = Sentinel2L2AIngestor(
        "image", ["B04", "SCL"], provider=provider, chunks=2
    )
    item = write_item(tmp_path, provider, day=datetime(2025, 1, 1))

    tree = ingestor.load([item], GRID)

    assert list(tree.children) == ["image"]
    assert list(tree["image"].to_dataset().data_vars) == ["B04", "SCL"]
    pixels = tree.gs.to_tensor()
    assert list(pixels) == ["image"]
