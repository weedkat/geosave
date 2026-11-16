"""Load Sentinel-2 L2A acquisitions from Planetary Computer."""

from functools import cached_property

import pandas as pd
import pystac
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from geosave.geodata import GeoRegion, stac
from geosave.geodata.stac import StacClient, StacQuery
from geosave.ml.processing.raster.ingestion.base import INGESTORS, RasterIngestor

# Planetary Computer's L2A; its STAC states no scale or offset (see read_esa_packing).
COLLECTION = "sentinel-2-l2a"


@INGESTORS.register("sentinel2_l2a")
class Sentinel2Ingestor(RasterIngestor):
    """Load the Sentinel-2 L2A acquisitions over a region from Planetary Computer.

    Args:
        bands: Sentinel-2 bands to load, in model order.
        resolution: Pixel size in metres of an ingested raster.
        name: Node an acquisition's raster is returned under; the name the
            preprocessor reads it by.
    """

    def __init__(self, bands: list[str], resolution: float = 10.0, name: str = "image"):
        self.bands = bands
        self.resolution = resolution
        self.name = name

    @cached_property
    def client(self) -> StacClient:
        """Open Planetary Computer on first search, so predicting a file needs no network."""
        return StacClient.planetary_computer()

    def search_items(self, region: GeoRegion) -> list[pystac.Item]:
        """Find the acquisitions covering a region in its timespan, newest processing only."""
        query = StacQuery(collections=[COLLECTION]).set_region(region)

        return drop_superseded(self.client.search_items(query))

    def __call__(self, region: GeoRegion) -> list[xr.DataTree]:
        """Load each acquisition over a region as one lazy raster.

        Args:
            region: Place and timespan to predict.

        Returns:
            One tree per acquisition time, whose one node holds digital
            numbers in the region's UTM zone and states its packing and `time`.

        Raises:
            ValueError: No acquisition covers the region in its timespan.
        """
        items = self.search_items(region)
        if not items:
            raise ValueError(
                "no Sentinel-2 acquisition covers the region in its timespan"
            )

        acquisitions = stac.load_dataset(
            items,
            geopolygon=region.extent,
            crs="utm",
            resolution=self.resolution,
            bands=self.bands,
            groupby="solar_day",
            resampling="bilinear",
            dtype="uint16",
            nodata=0,
            chunks={},
            patch_url=self.client.patch_url,
        )
        packing = read_esa_packing(items)

        acquired = []
        for step in range(acquisitions.sizes["time"]):
            image = acquisitions.isel(time=step).gs.write_packing(packing)
            acquired.append(xr.DataTree.from_dict({self.name: image}))
        return acquired

    def ingest_grid(
        self,
        geobox: GeoBox,
        day: pd.Timestamp,
        *,
        items: list[pystac.Item] | None = None,
    ) -> xr.Dataset:
        """Load one day's acquisition onto an exact grid, such as a label's.

        Neighbouring tiles of one pass are mosaicked into one image. Raster loading
        and ESA reflectance packing are owned here, including for saved Items.

        Args:
            geobox: Exact output grid.
            day: Acquisition day, used when searching.
            items: Previously selected Items for this acquisition. If supplied,
                load these exact Items without another catalogue search.

        Raises:
            ValueError: No acquisition, or more than one, covers the grid that day.
        """
        if items is None:
            region = GeoRegion.from_geometry(geobox.extent, timespan=f"{day:%Y-%m-%d}")
            items = self.search_items(region)
        if not items:
            raise ValueError(
                f"no Sentinel-2 acquisition covers the grid on {day:%Y-%m-%d}"
            )

        packing = read_esa_packing(items)
        with stac.load_dataset(
            items,
            geobox=geobox,
            bands=self.bands,
            resampling="bilinear",
            dtype="uint16",
            nodata=0,
            chunks=None,
            patch_url=self.client.patch_url,
        ) as image:
            if image.sizes["time"] != 1:
                raise ValueError(
                    f"expected one acquisition on {day:%Y-%m-%d}, found {image.sizes['time']}"
                )
            return image.isel(time=0).gs.write_packing(packing)


def drop_superseded(items: list[pystac.Item]) -> list[pystac.Item]:
    """Keep only the newest processing of each tile acquisition.

    Planetary Computer lists an acquisition once per ESA processing, e.g. the
    2020 original (baseline 02.12) and the 2023 reprocessing (05.00). Their
    ids differ only in the last field, the processing time. ODC would fuse
    both as one observation, so the first listed would silently win.
    """
    newest = {}
    for item in items:
        acquisition, processed = item.id.rsplit("_", 1)
        if acquisition not in newest or processed > newest[acquisition][0]:
            newest[acquisition] = (processed, item)

    kept = []
    for _, item in newest.values():
        kept.append(item)
    return kept


def read_esa_packing(items: list[pystac.Item]) -> attrs.CFPacking:
    """Derive reflectance decoding from ESA's product format and processing baseline.

    Planetary Computer's STAC states no scale or offset, only the ESA
    processing baseline (`s2:processing_baseline`, the `N0500` of a product
    name). ESA: reflectance = (DN + BOA_ADD_OFFSET) / 10000, where
    BOA_ADD_OFFSET is -1000 from baseline 04.00 and 0 before
    (https://sentiwiki.copernicus.eu/web/s2-processing). Different baselines
    may share the same decoding; reject only combinations with different offsets.
    """
    baselines = set()
    for item in items:
        baselines.add(item.properties["s2:processing_baseline"])
    offsets = {-0.1 if float(baseline) >= 4 else 0.0 for baseline in baselines}
    if len(offsets) != 1:
        raise ValueError(
            f"tiles mix processing baselines with different reflectance offsets: {sorted(baselines)}"
        )

    offset = offsets.pop()
    return attrs.CFPacking(scale_factor=1e-4, add_offset=offset, fill_value=0)
