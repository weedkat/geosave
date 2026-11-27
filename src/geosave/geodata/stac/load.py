"""Load pixels from selected STAC Items and apply explicit metadata stamps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict, Unpack, cast

import odc.stac
import pystac
import xarray as xr
from odc.geo import CRS, MaybeCRS, Resolution, SomeResolution
from odc.geo.geobox import GeoBox, GeoboxAnchor
from odc.loader.types import Band_DType, RasterSource, ReaderDriverSpec
from odc.stac import ConversionConfig
from odc.stac.model import ParsedItem

from geosave.geodata.stac.stamps import STAMPS, Stamp, stamp_nodata
from geosave.geodata.utils.functions import resolve_fn


class StacLoadOptions(TypedDict, total=False):
    """Options of `odc.stac.load`, as it names and defaults them.

    Choose the grid with one of them: `geobox`, `like=raster` to copy a
    raster's grid, or `crs` and `resolution` with a region (`bbox`, `lon`/`lat`,
    `x`/`y`, `geopolygon`, `intersects`).

    `with_properties` loads one data variable per requested Item property,
    with one value per time step. Pass `stamps=[stamp_properties]`
    to promote these variables to coordinates marked with their STAC keys. ODC's
    request defaults apply:

        "eo:cloud_cover"                          eo_cloud_cover, float32
        {"key": "eo:cloud_cover", "dtype": "float64"}   keeps 24.001125 exact
        {"key": "platform", "dtype": "str"}       text; float32 fails on it
        {"key": "eo:cloud_cover", "name": "cloud"}      named cloud

    A step fusing several Items averages numbers and joins text with ",";
    pass `fuser` to combine them otherwise.

    ODC resamples into the requested pixel `dtype` and maps missing pixels to
    the output `nodata`. A scalar `dtype` also applies to property variables.
    Omit it to keep the dtypes in the property requests. In odc-loader 0.6.4,
    per-band dtype mappings are supported for eager loads and fail during
    Dask chunk planning.

    The Items must intersect the grid, as a search over the same region
    returns them: ODC 0.5.2 fails on properties when the last step's Items
    all miss it.
    """

    bands: str | Sequence[str] | None
    groupby: str | Callable[[pystac.Item, ParsedItem, int], Any] | None
    resampling: str | dict[str, str] | None
    dtype: Band_DType
    nodata: float | None
    chunks: dict[str, int | Literal["auto"]] | None
    pool: ThreadPoolExecutor | int | None
    crs: MaybeCRS
    resolution: SomeResolution | None
    anchor: GeoboxAnchor | None
    geobox: GeoBox | None
    bbox: tuple[float, float, float, float] | None
    lon: tuple[float, float] | None
    lat: tuple[float, float] | None
    x: tuple[float, float] | None
    y: tuple[float, float] | None
    like: Any
    geopolygon: Any
    intersects: Any
    progress: Any
    fail_on_error: bool
    stac_cfg: ConversionConfig | None
    with_properties: Sequence[str | Mapping[str, Any]] | None
    patch_url: Callable[[str], str] | None
    preserve_original_order: bool
    driver: ReaderDriverSpec | None
    fuse_func: str | Mapping[str, str | None] | None


def load_dataset(
    items: Iterable[pystac.Item],
    *,
    stamps: Sequence[Stamp | str] = (),
    **options: Unpack[StacLoadOptions],
) -> xr.Dataset:
    """Load the selected Items' pixels onto one ODC grid.

    Choose the grid with `geobox`, `like`, or `crs` and `resolution` with
    a region. ODC derives a grid from the Items when these are omitted.
    Pass `chunks={}` or sizes for a lazy load. ODC groups Items into time
    steps and fuses their pixels. Every load applies `stamp_nodata` to write
    each band's effective fill as CF `_FillValue` before any optional stamps.

    Args:
        items: Selected Items; generators are consumed once.
        stamps: Stamps in execution order, each a callable or its name in
            `stac.stamps.STAMPS`. Each receives the Dataset, the Items and the
            `stac_cfg` option.
        **options: Options of `odc.stac.load`.

    Returns:
        The ODC Dataset with the metadata written by the requested stamps.

    Raises:
        KeyError: A stamp name is absent from `STAMPS`.
        ValueError: ODC refuses the Items or options, such as an empty
            selection or a grid it cannot infer.

    Examples:
        >>> scene = stac.load_dataset(
        ...     items, geobox=grid, chunks={}, with_properties=["eo:cloud_cover"],
        ...     stamps=["properties"],
        ... )
        >>> scene["eo_cloud_cover"].attrs
        {'units': '1', 'stac_property': 'eo:cloud_cover'}
        >>> labels = stac.load_dataset(label_items, like=scene, chunks={})
    """
    selected = [resolve_fn(stamp, STAMPS) for stamp in stamps]
    items = list(items)

    # ODC inserts its property requests into the provided config.
    stac_cfg = options.get("stac_cfg")
    if stac_cfg is not None and options.get("with_properties") is not None:
        options["stac_cfg"] = dict(stac_cfg)

    dataset = odc.stac.load(items, **options)
    dataset = stamp_nodata(dataset, items, stac_cfg)

    for stamp in selected:
        dataset = stamp(dataset, items, stac_cfg)

    return dataset


def group_bands_by_grid(
    items: Iterable[pystac.Item],
    bands: str | Sequence[str] | None = None,
    *,
    stac_cfg: ConversionConfig | None = None,
) -> tuple[tuple[CRS, Resolution, list[str]], ...]:
    """Group the selected bands by the CRS and resolution their `proj:*` states.

    Bands of one group load together onto their native grid: ODC picks the
    finest resolution among a load's bands, which is every band's own. Bands
    an Item lacks, or whose grid `proj:*` does not state, are left out.

    Args:
        items: Items whose bands to group, as ODC parses them.
        bands: Bands to group, as `odc.stac.load` selects them; None groups all.
        stac_cfg: ODC configuration used to identify assets and aliases.

    Returns:
        One `(crs, resolution, bands)` per grid, finest first.

    Examples:
        >>> for crs, resolution, bands in stac.group_bands_by_grid(items):
        ...     print(crs, resolution, bands)
        EPSG:32633 Resolution(x=10, y=-10) ['B02', 'B03', 'B04', 'B08']
        EPSG:32633 Resolution(x=20, y=-20) ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    """
    bands_by_grid: dict[tuple[CRS, Resolution], list[str]] = {}
    for parsed_item in odc.stac.parse_items(items, cfg=stac_cfg):
        for band, band_source in parsed_item.resolve_bands(bands).items():
            # ODC names every band of the collection; None is one this Item lacks.
            if not isinstance(band_source, RasterSource):
                continue
            grid = band_source.geobox
            if grid is None:
                continue
            grid_key = (cast("CRS", grid.crs), grid.resolution)
            grid_bands = bands_by_grid.setdefault(grid_key, [])
            if band not in grid_bands:
                grid_bands.append(band)

    groups = []
    for (crs, resolution), grid_bands in bands_by_grid.items():
        groups.append((crs, resolution, grid_bands))
    # Finest first; one resolution in several CRSs keeps a stable order.
    groups.sort(
        key=lambda group: (min(abs(group[1].x), abs(group[1].y)), str(group[0]))
    )
    return tuple(groups)


def load_native_datasets(
    items: Iterable[pystac.Item],
    *,
    stamps: Sequence[Stamp | str] = (),
    **options: Unpack[StacLoadOptions],
) -> tuple[tuple[CRS, Resolution, xr.Dataset], ...]:
    """Load the selected Items once per native grid, so no band is resampled.

    Each group of `group_bands_by_grid` loads with `load_dataset` and no grid
    option, so ODC picks the group's own grid over the Items' footprints. A
    region option (`bbox`, `geopolygon`, ...) crops each group at its own
    resolution; a grid option (`geobox`, `like`, `resolution`, ...) would put
    every group on one grid.

    Args:
        items: Selected Items of one collection.
        stamps: Names or callables applied to each Dataset, as for `load_dataset`.
        **options: Options of `odc.stac.load`, as for `load_dataset`; `bands`
            selects which bands to group.

    Returns:
        One `(crs, resolution, dataset)` per grid, finest first.

    Examples:
        >>> for crs, resolution, native in stac.load_native_datasets(items, chunks={}):
        ...     native.gs.to_zarr(f"s2_{abs(resolution.x):g}m.zarr")
    """
    items = list(items)

    loaded = []
    groups = group_bands_by_grid(
        items, options.get("bands"), stac_cfg=options.get("stac_cfg")
    )
    for crs, resolution, bands in groups:
        options["bands"] = bands
        native = load_dataset(items, stamps=stamps, **options)
        loaded.append((crs, resolution, native))
    return tuple(loaded)
