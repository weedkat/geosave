"""Load Sentinel-1 terrain-corrected backscatter from the provider a config names.

A collection module has three parts, which the next collection copies:

1. Product: what the producer defines, the same on every provider.
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

from geosave.geodata import GeoRegion, stac
from geosave.geodata.stac import StacClient, StacQuery

from .base import Ingestor

# 1. Product ------------------------------------------------------------------

POLARIZATIONS = ("VV", "VH", "HH", "HV")

# Fill of the float32 backscatter rasters.
NODATA = -32768.0


# 2. Providers ----------------------------------------------------------------


class Assets(TypedDict):
    """A provider's asset key for each polarization."""

    VV: str
    VH: str
    HH: str
    HV: str


@dataclass(frozen=True)
class Source:
    """How one catalog publishes Sentinel-1 RTC.

    Args:
        open_client: Builds the catalog's client, with its access and signing.
        collection: The catalog's collection ID.
        assets: The catalog's asset key for each polarization.
    """

    open_client: Callable[[], StacClient]
    collection: str
    assets: Assets


class Provider(Enum):
    """Catalogs serving Sentinel-1 RTC, selected by name in a config."""

    planetary_computer = Source(
        StacClient.planetary_computer,
        "sentinel-1-rtc",
        Assets(VV="vv", VH="vh", HH="hh", HV="hv"),
    )


# 3. Ingestor -----------------------------------------------------------------


class Sentinel1RTCIngestor(Ingestor):
    """Load the Sentinel-1 RTC acquisitions over a region.

    Every band is float32 terrain-flattened gamma0 backscatter in linear
    power, with fill -32768, one time step per solar day labelled by its
    acquisition time. A region's acquisitions offer VV and VH, or HH and HV
    in polar regions.

    Args:
        name: Name of the collection node.
        bands: Polarizations to load, such as `["VV", "VH"]`.
        groups: Child node names mapping a partition of `bands`, such as
            `{"sar": ["VV", "VH"]}`. None keeps every band in the collection
            node itself.
        provider: Catalog to read.
        resolution: Pixel size of the region grid, in CRS units.
        crs: CRS of the region grid; None uses the region's UTM zone.
        resampling: Resampling of the backscatter.
        filter: CQL2 text narrowing the search, such as
            `sat:orbit_state = 'ascending'`.
        chunks: Spatial chunk side in pixels.
        time_chunks: Dates per chunk; -1 puts every date in one chunk, which
            suits composites, and 1 suits reading a few dates at a time. A
            staged tree keeps this layout.

    Raises:
        ValueError: A band is not a Sentinel-1 polarization.

    Examples:
        >>> ingestor = Sentinel1RTCIngestor(
        ...     "sar", ["VV", "VH"], filter="sat:orbit_state = 'ascending'"
        ... )
        >>> ingestor(region)["sar"]
        <xarray.DataTree 'sar'> ...
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
            if band not in POLARIZATIONS:
                raise ValueError(f"{band!r} is not a Sentinel-1 polarization")
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
            ValueError: No Items, or groups that are empty or do not partition
                the bands.
        """
        if not items:
            raise ValueError("no Sentinel-1 RTC acquisition to load")

        encoding = {"*": {"data_type": "float32", "nodata": NODATA}}
        aliases = dict(self.source.assets)
        dataset = stac.load_dataset(
            items,
            bands=self.bands,
            stac_cfg={self.source.collection: {"assets": encoding, "aliases": aliases}},
            patch_url=self.client.patch_url,
            geobox=geobox,
            chunks={"time": self.time_chunks, "y": self.chunks, "x": self.chunks},
            groupby="solar_day",
            nodata=NODATA,
            resampling=self.resampling,
        )
        return dataset.gs.to_tree(self.name, groups=self.groups)
