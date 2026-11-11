"""Bulk region construction and STAC ingestion for model recipes.

A region describes where and when to look. Ingestion owns searching that area
and loading its pixels onto the requested grid. Results retain input order;
missing imagery raises instead of silently removing a sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import TYPE_CHECKING, Unpack

from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox

from geosave.geodata.core.region import GeoRegion
from geosave.geodata.stac.load import StacLoadOptions, load_dataset
from geosave.geodata.stac.search.query import StacQuery

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr
    from odc.geo import SomeCRS

    from geosave.geodata.stac.search.api import StacClient
    from geosave.geodata.stac.search.table import StacTable
    from geosave.geodata.utils.datetime import RegionDatetime


def regions_from_features(
    features: gpd.GeoDataFrame | gpd.GeoSeries,
    *,
    timespan: RegionDatetime | None = None,
) -> tuple[GeoRegion, ...]:
    """Create one region per feature, preserving row order and CRS.

    Args:
        features: Points or areas with a CRS. Features are never unioned.
        timespan: Common observation interval; None imposes no date selection.

    Returns:
        Ordered regions. For per-feature dates, construct GeoRegions explicitly
        and pass that sequence directly to ingest_stac.

    Raises:
        ValueError: CRS is missing or a feature is invalid.
    """
    if features.crs is None:
        raise ValueError(
            "Features need a CRS before they can define ingestion regions."
        )

    regions = []
    for position in range(len(features)):
        try:
            region = GeoRegion.from_geometry(
                features.iloc[position : position + 1], timespan=timespan
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot create region for feature at position {position}: {error}"
            ) from error
        regions.append(region)
    return tuple(regions)


def ingest_stac(
    regions: Sequence[GeoRegion],
    *,
    catalog: StacClient | StacTable,
    collection: str,
    crs: SomeCRS,
    resolution: float,
    shape: tuple[int, int] | None = None,
    groupby: str = "solar_day",
    **options: Unpack[StacLoadOptions],
) -> tuple[xr.Dataset, ...]:
    """Search and load one collection for an ordered sequence of regions.

    Args:
        regions: Areas, or points when shape is supplied. Each owns its dates.
        catalog: Open STAC API client or STAC GeoParquet table.
        collection: Collection ID to search; metadata is fetched once per call.
        crs: Output CRS, explicitly shared by this batch.
        resolution: Positive square pixel size in output CRS units.
        shape: Height and width in pixels for point-centered windows. Omit for
            polygon coverage; no polygon masking is performed.
        groupby: ODC observation grouping, defaulting to solar_day mosaics.
        **options: Native load options, including bands, chunks, resampling,
            and stac_cfg. Catalog URL signing is used unless overridden.

    Returns:
        One Dataset per region, in order. Pixels remain lazy with the default
        chunk settings. All matching observations are retained before grouping;
        no best-observation selection or temporal compositing is implicit.

    Raises:
        ValueError: Grid settings are invalid or imagery cannot be acquired for
            a region. The error identifies its input position.
    """
    if not isfinite(resolution) or resolution <= 0:
        raise ValueError(
            "resolution must be a positive finite pixel size in CRS units."
        )
    if shape is not None and (
        len(shape) != 2
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in shape
        )
    ):
        raise ValueError("shape must contain two positive pixel counts: height, width.")

    grids = []
    for position, region in enumerate(regions):
        try:
            grids.append(_build_grid(region, crs, resolution, shape))
        except ValueError as error:
            raise ValueError(
                f"Cannot grid region at position {position}: {error}"
            ) from error

    if not grids:
        return ()

    metadata = catalog.collection(collection)
    settings = options.copy()
    settings.setdefault("patch_url", catalog.patch_url)
    query = StacQuery(collections=[collection])
    rasters = []
    for position, (region, grid) in enumerate(zip(regions, grids, strict=True)):
        # Search the full pixel window so neighboring tiles can fill its edges.
        coverage = GeoRegion(grid.extent, region.timespan)
        try:
            items = catalog.search_items(query.set_region(coverage))
            raster = load_dataset(
                items, geobox=grid, collection=metadata, groupby=groupby, **settings
            )
        except Exception as error:
            raise ValueError(
                f"Cannot ingest region at position {position} from collection "
                f"{collection!r} for dates {region.timespan}: {error}"
            ) from error
        rasters.append(raster)
    return tuple(rasters)


def _build_grid(
    region: GeoRegion,
    crs: SomeCRS,
    resolution: float,
    shape: tuple[int, int] | None,
) -> GeoBox:
    """Cover an area or center a fixed pixel window on a point."""
    if shape is None:
        if region.extent.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                "Point regions need shape=(height, width) to define a window."
            )
        return GeoBox.from_geopolygon(region.extent, crs=crs, resolution=resolution)

    if region.extent.geom_type != "Point":
        raise ValueError(
            "shape defines point windows; omit it when loading polygon regions."
        )

    x, y = region.extent.to_crs(crs).coords[0]
    height, width = shape
    half_width = width * resolution / 2
    half_height = height * resolution / 2
    bounds = BoundingBox(
        x - half_width, y - half_height, x + half_width, y + half_height, crs=crs
    )
    return GeoBox.from_bbox(bounds, shape=shape, tight=True)
