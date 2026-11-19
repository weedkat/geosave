"""Load Sentinel-2 L2A acquisitions from a public STAC catalog."""

from functools import cached_property

import pandas as pd
import pystac
import xarray as xr
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from geosave.geodata import GeoRegion, stac
from geosave.geodata.stac import StacClient, StacQuery
from geosave.ml.processing.raster.ingestion.base import INGESTORS, RasterIngestor

COLLECTION = "sentinel-2-l2a"

# Asset key of each Sentinel-2 band in each catalog's L2A collection.
ASSET_KEYS = {
    "planetary_computer": {
        "B01": "B01",
        "B02": "B02",
        "B03": "B03",
        "B04": "B04",
        "B05": "B05",
        "B06": "B06",
        "B07": "B07",
        "B08": "B08",
        "B8A": "B8A",
        "B09": "B09",
        "B11": "B11",
        "B12": "B12",
        "SCL": "SCL",
    },
    "element84": {
        "B01": "coastal",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "rededge1",
        "B06": "rededge2",
        "B07": "rededge3",
        "B08": "nir",
        "B8A": "nir08",
        "B09": "nir09",
        "B11": "swir16",
        "B12": "swir22",
        "SCL": "scl",
    },
}

# Property naming each Item's MGRS tile in each catalog.
TILE_PROPERTIES = {"planetary_computer": "s2:mgrs_tile", "element84": "grid:code"}


@INGESTORS.register("sentinel2_l2a")
class Sentinel2Ingestor(RasterIngestor):
    """Load the Sentinel-2 L2A acquisitions over a region from a STAC catalog.

    Reflectance and the scene classification (SCL) are separate nodes: a
    raster node holds one pixel meaning, and SCL is categorical, resampled
    by nearest neighbour without reflectance packing.

    Args:
        bands: Reflectance bands to load, in model order, such as `B04`.
        resolution: Pixel size in metres of an ingested raster.
        name: Node the reflectance is returned under; the name the
            preprocessor reads it by.
        scl: Node the SCL is returned under; None loads no SCL.
        catalog: Catalog to search and load from: `planetary_computer` or
            `element84`.

    Raises:
        ValueError: The catalog is unknown, a band is not a reflectance
            band, or `scl` names the reflectance node.
    """

    def __init__(
        self,
        bands: list[str],
        resolution: float = 10.0,
        name: str = "image",
        scl: str | None = None,
        catalog: str = "planetary_computer",
    ):
        if catalog not in ASSET_KEYS:
            raise ValueError(
                f"unknown catalog {catalog!r}; use one of {list(ASSET_KEYS)}"
            )
        for band in bands:
            if band == "SCL" or band not in ASSET_KEYS[catalog]:
                raise ValueError(f"{band!r} is not a Sentinel-2 reflectance band")
        if scl == name:
            raise ValueError(f"SCL and reflectance cannot share the node {name!r}")

        self.bands = bands
        self.resolution = resolution
        self.name = name
        self.scl = scl
        self.catalog = catalog

    @cached_property
    def client(self) -> StacClient:
        """Open the catalog on first search, so predicting a file needs no network."""
        return getattr(StacClient, self.catalog)()

    def search_items(self, region: GeoRegion) -> list[pystac.Item]:
        """Find the acquisitions covering a region in its timespan, newest processing only."""
        query = StacQuery(collections=[COLLECTION]).set_region(region)

        items = self.client.search_items(query)
        return drop_superseded(items, TILE_PROPERTIES[self.catalog])

    def __call__(self, region: GeoRegion) -> list[xr.DataTree]:
        """Load each acquisition over a region as lazy rasters.

        Args:
            region: Place and timespan to predict.

        Returns:
            One tree per acquisition time, holding digital numbers in the
            region's UTM zone: reflectance with its packing under `name`, and
            SCL under `scl` when requested. Each node states its `time`.

        Raises:
            ValueError: No acquisition covers the region in its timespan.
        """
        items = self.search_items(region)
        if not items:
            raise ValueError(
                "no Sentinel-2 acquisition covers the region in its timespan"
            )

        acquisitions = self.load(
            items,
            geopolygon=region.extent,
            crs="utm",
            resolution=self.resolution,
            groupby="solar_day",
            chunks={},
        )

        trees = []
        for step in range(acquisitions.sizes["time"]):
            trees.append(self.split_nodes(acquisitions.isel(time=step)))
        return trees

    def ingest_grid(
        self,
        geobox: GeoBox,
        day: pd.Timestamp,
        *,
        items: list[pystac.Item] | None = None,
    ) -> xr.DataTree:
        """Load one day's acquisition onto an exact grid, such as a label's.

        Neighbouring tiles of one pass are mosaicked into one image.

        Args:
            geobox: Exact output grid.
            day: Acquisition day, used when searching.
            items: Previously selected Items for this acquisition. If supplied,
                load these exact Items without another catalogue search.

        Returns:
            Eager tree with the nodes `__call__` returns.

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

        image = self.load(items, geobox=geobox, groupby="solar_day", chunks=None)
        if image.sizes["time"] != 1:
            raise ValueError(
                f"expected one acquisition on {day:%Y-%m-%d}, found {image.sizes['time']}"
            )
        return self.split_nodes(image.isel(time=0))

    def load(self, items: list[pystac.Item], **options) -> xr.Dataset:
        """Load Items by Sentinel-2 band name, with reflectance packing stated.

        Args:
            items: Items of this ingestor's catalog.
            **options: Grid and chunk options of `stac.load_dataset`.

        Returns:
            Digital numbers of the bands, and SCL when requested, over `time`.
        """
        names = list(self.bands)
        if self.scl is not None:
            names.append("SCL")

        assets = {}
        resampling = {}
        for band in names:
            key = ASSET_KEYS[self.catalog][band]
            assets[key] = band
            resampling[key] = "nearest" if band == "SCL" else "bilinear"

        dataset = stac.load_dataset(
            items,
            bands=list(assets),
            resampling=resampling,
            dtype="uint16",
            nodata=0,
            patch_url=self.client.patch_url,
            **options,
        )
        dataset = dataset.rename_vars(assets)
        return dataset.gs.write_packing(read_esa_packing(items), target=self.bands)

    def split_nodes(self, acquisition: xr.Dataset) -> xr.DataTree:
        """Place reflectance and SCL of one acquisition in their own nodes."""
        nodes = {self.name: acquisition[self.bands]}
        if self.scl is not None:
            nodes[self.scl] = acquisition[["SCL"]]
        return xr.DataTree.from_dict(nodes)


def drop_superseded(items: list[pystac.Item], tile_property: str) -> list[pystac.Item]:
    """Keep only the newest processing of each tile acquisition.

    A catalog can list an acquisition once per ESA processing, e.g. the
    2020 original (baseline 02.12) and the 2023 reprocessing (05.00). A
    satellite passes a tile at most once a day, but a reprocessing may state
    a different sensing time, so the tile and day identify an acquisition.
    ODC would fuse both as one observation, so the first listed would
    silently win.

    Args:
        items: Searched Items.
        tile_property: Property naming an Item's MGRS tile.
    """
    newest = {}
    for item in items:
        acquisition = (
            item.properties["platform"],
            item.properties[tile_property],
            item.datetime.date(),
        )
        generated = item.properties["s2:generation_time"]
        if acquisition not in newest or generated > newest[acquisition][0]:
            newest[acquisition] = (generated, item)

    kept = []
    for _, item in newest.values():
        kept.append(item)
    return kept


def read_esa_packing(items: list[pystac.Item]) -> attrs.CFPacking:
    """Derive reflectance decoding from ESA's product format and processing baseline.

    Catalogs disagree on whether they state scale and offset, but each states
    the ESA processing baseline (`s2:processing_baseline`, the `N0500` of a
    product name). ESA: reflectance = (DN + BOA_ADD_OFFSET) / 10000, where
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
