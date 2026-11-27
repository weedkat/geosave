"""Load Sentinel-2 L2A from the provider a config names.

A collection module has three parts, which the next collection copies:

1. Product: what the producer (ESA) defines, the same on every provider.
2. Providers: how each catalog publishes the product.
3. Ingestor: search one provider, load onto a grid, state the product's meaning.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import pystac
import xarray as xr
from odc.geo import CRS
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from geosave.geodata import GeoRegion, stac
from geosave.geodata.stac import StacClient, StacQuery

from .base import Ingestor

# 1. Product ------------------------------------------------------------------

REFLECTANCE = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
)  # fmt: skip
QUALITY = ("SCL",)

# Scene classification class IDs, https://sentiwiki.copernicus.eu/web/s2-processing
SCL_CLASSES = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area_pixels",
    3: "cloud_shadows",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow",
}

# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
_SCL_COLORS = {
    0: "#000000", 1: "#FF0000", 2: "#2F2F2F", 3: "#643200",
    4: "#00A000", 5: "#FFE65A", 6: "#0000FF", 7: "#808080",
    8: "#C0C0C0", 9: "#FFFFFF", 10: "#64C8FF", 11: "#FF96FF",
}  # fmt: skip


# 2. Providers ----------------------------------------------------------------


class Assets(TypedDict):
    """A provider's asset key for each Sentinel-2 L2A band."""

    B01: str
    B02: str
    B03: str
    B04: str
    B05: str
    B06: str
    B07: str
    B08: str
    B8A: str
    B09: str
    B11: str
    B12: str
    SCL: str


@dataclass(frozen=True)
class Source:
    """How one catalog publishes Sentinel-2 L2A.

    Args:
        open_client: Builds the catalog's client, with its access and signing.
        collection: The catalog's collection ID.
        assets: The catalog's asset key for each band.
        baseline: Item property holding ESA's processing baseline.
    """

    open_client: Callable[[], StacClient]
    collection: str
    assets: Assets
    baseline: str


class Provider(Enum):
    """Catalogs serving Sentinel-2 L2A, selected by name in a config."""

    planetary_computer = Source(
        StacClient.planetary_computer,
        "sentinel-2-l2a",
        Assets(
            B01="B01", B02="B02", B03="B03", B04="B04", B05="B05", B06="B06",
            B07="B07", B08="B08", B8A="B8A", B09="B09", B11="B11", B12="B12",
            SCL="SCL",
        ),
        "s2:processing_baseline",
    )  # fmt: skip
    element84 = Source(
        StacClient.element84,
        "sentinel-2-c1-l2a",
        Assets(
            B01="coastal", B02="blue", B03="green", B04="red", B05="rededge1",
            B06="rededge2", B07="rededge3", B08="nir", B8A="nir08", B09="nir09",
            B11="swir16", B12="swir22", SCL="scl",
        ),
        "s2:processing_baseline",
    )  # fmt: skip
    cdse = Source(
        StacClient.cdse,
        "sentinel-2-l2a",
        Assets(
            B01="B01_60m", B02="B02_10m", B03="B03_10m", B04="B04_10m",
            B05="B05_20m", B06="B06_20m", B07="B07_20m", B08="B08_10m",
            B8A="B8A_20m", B09="B09_60m", B11="B11_20m", B12="B12_20m",
            SCL="SCL_20m",
        ),
        "processing:version",
    )  # fmt: skip


# 3. Ingestor -----------------------------------------------------------------


class Sentinel2L2AIngestor(Ingestor):
    """Load the Sentinel-2 L2A acquisitions over a region.

    Reflectance bands carry ESA's scale and offset. SCL carries its class names
    and colors. Every band is uint16 with fill 0, one time step per solar day.

    Args:
        name: Name of the collection node.
        bands: Bands to load, such as `["B04", "B08", "SCL"]`.
        groups: Child node names mapping a partition of `bands`, such as
            `{"image": ["B04", "B08"], "scl": ["SCL"]}`. None keeps every band
            in the collection node itself.
        provider: Catalog to read.
        resolution: Pixel size of the region grid, in CRS units.
        crs: CRS of the region grid; None uses the region's UTM zone.
        resampling: Resampling of reflectance bands; SCL is always nearest.
        filter: CQL2 text narrowing the search, such as `eo:cloud_cover <= 20`.
        chunks: Spatial chunk side in pixels.
        time_chunks: Dates per chunk; -1 puts every date in one chunk, which
            suits composites, and 1 suits reading a few dates at a time. A
            staged tree keeps this layout.

    Raises:
        ValueError: A band is not a Sentinel-2 L2A band.
    """

    def __init__(
        self,
        name: str,
        bands: list[str],
        *,
        groups: dict[str, list[str]] | None = None,
        provider: Provider = Provider.planetary_computer,
        resolution: float = 10.0,
        crs: str | None = None,
        resampling: str = "bilinear",
        filter: str | None = None,
        chunks: int = 1024,
        time_chunks: int = 1,
    ) -> None:
        for band in bands:
            if band not in REFLECTANCE and band not in QUALITY:
                raise ValueError(f"{band!r} is not a Sentinel-2 L2A band")
        self.name = name
        self.groups = groups
        self.bands = bands
        self.provider = provider
        self.source = provider.value
        self.client = self.source.open_client()
        self.resolution = resolution
        self.crs = crs
        self.resampling = resampling
        self.filter = filter
        self.chunks = chunks
        self.time_chunks = time_chunks

    def __call__(self, region: GeoRegion) -> xr.DataTree:
        """Load every acquisition over a region onto the configured grid, lazily.

        Args:
            region: Place and time span to read.

        Returns:
            The collection's subtree; pixels are read when computed.
        """
        crs = self.crs if self.crs is not None else CRS.utm(region.extent.centroid)
        geobox = GeoBox.from_geopolygon(
            region.extent, resolution=self.resolution, crs=crs
        )
        return self.load(self.search(region), geobox)

    def search(self, region: GeoRegion) -> list[pystac.Item]:
        """Find the provider's Items covering a region in its timespan."""
        query = StacQuery(collections=[self.source.collection]).set_region(region)
        if self.filter is not None:
            query = query.set_filter(self.filter)
        return self.client.search_items(query)

    def load(self, items: list[pystac.Item], geobox: GeoBox) -> xr.DataTree:
        """Load the provider's Items onto a grid, one step per solar day.

        Raises:
            ValueError: No Items, Items of different packing, or groups that
                are empty or do not partition the bands.
        """
        if not items:
            raise ValueError("no Sentinel-2 L2A acquisition to load")

        resampling = {}
        reflectance = []
        quality = []
        for band in self.bands:
            if band in QUALITY:
                resampling[band] = "nearest"
                quality.append(band)
            else:
                resampling[band] = self.resampling
                reflectance.append(band)

        # ESA's encoding, which some catalogs leave out of their Items.
        encoding = {"*": {"data_type": "uint16", "nodata": 0}}
        aliases = dict(self.source.assets)
        dataset = stac.load_dataset(
            items,
            stamps=("eo",),
            bands=self.bands,
            stac_cfg={self.source.collection: {"assets": encoding, "aliases": aliases}},
            patch_url=self.client.patch_url,
            geobox=geobox,
            chunks={"time": self.time_chunks, "y": self.chunks, "x": self.chunks},
            groupby="solar_day",
            nodata=0,
            resampling=resampling,
        )
        dataset = dataset.gs.write_packing(self.read_packing(items), target=reflectance)
        dataset = dataset.gs.write_flags(SCL_CLASSES, _SCL_COLORS, target=quality)
        return dataset.gs.to_tree(self.name, groups=self.groups)

    def read_packing(self, items: list[pystac.Item]) -> attrs.CFPacking:
        """Return ESA's reflectance packing, which depends on the processing baseline.

        Raises:
            ValueError: Items differ in packing, such as baselines on both
                sides of 04.00.
        """
        offsets = set()
        for item in items:
            baseline = float(item.properties[self.source.baseline])
            # https://sentiwiki.copernicus.eu/web/s2-processing
            offsets.add(-0.1 if baseline >= 4 else 0.0)
        if len(offsets) != 1:
            raise ValueError(
                f"Sentinel-2 L2A Items mix reflectance offsets {sorted(offsets)}"
            )
        return attrs.CFPacking(
            scale_factor=1e-4, add_offset=offsets.pop(), fill_value=0
        )
