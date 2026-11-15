"""Load selected STAC Items with ODC and stamp what their documents state."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict, Unpack, cast

import numpy as np
import odc.stac
import pystac
import xarray as xr
from odc.geo import CRS, MaybeCRS, Resolution, SomeResolution
from odc.geo.geobox import GeoBox, GeoboxAnchor
from odc.loader.types import Band_DType, RasterSource, ReaderDriverSpec
from odc.stac import ConversionConfig
from odc.stac.model import ParsedItem, PropertyLoadRequest

from geosave.geodata.core.conventions import END_DATETIME, START_DATETIME
from geosave.geodata.stac.stamps import (
    DEFAULT_STAMPS,
    LoadSource,
    check_stamps,
    run_stamps,
)
from geosave.geodata.stac.stamps.properties import (
    PERIOD_REQUESTS,
    stamp_period,
    stamp_properties,
)


class StacLoadOptions(TypedDict, total=False):
    """Options of `odc.stac.load`, as it names and defaults them.

    Choose the grid with one of them: `geobox`, `like=raster` to copy a
    raster's grid, or `crs` and `resolution` with a region (`bbox`, `lon`/`lat`,
    `x`/`y`, `geopolygon`, `intersects`).

    `with_properties` loads one value per time step, which `load_dataset`
    makes a coordinate marked with its STAC key. Every load also requests
    `start_datetime` and `end_datetime`, so those two keys are taken. ODC's request
    defaults apply:

        "eo:cloud_cover"                          eo_cloud_cover, float32
        {"key": "eo:cloud_cover", "dtype": "float64"}   keeps 24.001125 exact
        {"key": "platform", "dtype": "str"}       text; float32 fails on it
        {"key": "eo:cloud_cover", "name": "cloud"}      named cloud

    A step fusing several Items averages numbers and joins text with ",";
    pass `fuser` to combine them otherwise.

    `dtype` and `nodata` apply to the bands only, after loading: ODC would
    apply them to the properties too.

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
    stamps: Sequence[str] = DEFAULT_STAMPS,
    **options: Unpack[StacLoadOptions],
) -> xr.Dataset:
    """Load selected Items with ODC, then stamp what their documents state.

    No search occurs: choose the Items before loading. The Collection is the
    one the Items link to (`item.get_collection()`); searched Items and
    catalog rows carry it.
    ODC loads as it does natively: it chooses the grid from `geobox`, `like`
    or `crs`/`resolution` with a region, else guesses it from the Items; pass
    `chunks={}` or sizes for a lazy load.
    Each step's period, from its Items' earliest start to latest end, becomes
    the `start_datetime`/`end_datetime` cells, and each `with_properties`
    request a coordinate marked `stac_property`.
    The named stamps then write metadata, reading no pixels. Band metadata
    keeps the published digital numbers; decode them with `.gs.mask_and_scale()`.

    Args:
        items: Selected Items; generators are consumed once.
        stamps: Registered stamps to run, in order; see `stac.stamps`.
        **options: Options of `odc.stac.load`.

    Returns:
        The Dataset ODC loads, with `start_datetime`, `end_datetime` and the
        requested properties as `(time,)` coordinates, and what the stamps
        write: by default band metadata and the Collection.

    Raises:
        KeyError: A stamp name is not registered; raised before loading.
        ValueError: ODC refuses the Items or options, such as an empty
            selection or a grid it cannot guess, or a stamp refuses them,
            such as differing source scales/offsets of one band.

    Examples:
        >>> scene = stac.load_dataset(items, geobox=grid, chunks={}, with_properties=["eo:cloud_cover"])
        >>> scene["eo_cloud_cover"].attrs
        {'units': '1', 'stac_property': 'eo:cloud_cover'}
        >>> labels = stac.load_dataset(label_items, like=scene, chunks={})
    """
    check_stamps(stamps)
    items = list(items)

    # ODC loads each property as a band, so its `dtype` and `nodata` would
    # apply to them too; the bands get both once the properties are coordinates.
    dtype = options.pop("dtype", None)
    nodata = options.pop("nodata", None)
    properties = [*(options.get("with_properties") or []), *PERIOD_REQUESTS]
    options["with_properties"] = properties
    ds = odc.stac.load(items, **options)

    source = LoadSource(items, PropertyLoadRequest.from_user_input(properties))

    # compulsory stamps: period and properties, which the user may also request
    ds = stamp_properties(ds, source)
    ds = stamp_period(ds)

    ds = _cast_bands(ds, dtype, nodata)

    return run_stamps(ds, source, stamps)


def list_properties(items: Iterable[pystac.Item]) -> list[dict[str, str]]:
    """List a `with_properties` request for each scalar property the Items state.

    Numbers load as float64 and text as str, so values keep what the Items
    state instead of ODC's float32 default. The datetime and period keys are
    left out, since every load reads them; so are booleans, lists and objects.

    Args:
        items: Items whose properties to request.

    Returns:
        Requests in ODC's form, in the order the Items first state each key.

    Examples:
        >>> stac.list_properties(items)
        [{'key': 'eo:cloud_cover', 'dtype': 'float64'}, {'key': 'platform', 'dtype': 'str'}]
        >>> scene = stac.load_dataset(items, with_properties=stac.list_properties(items))
    """
    # Every load reads the datetime and the period itself.
    time_keys = ("datetime", START_DATETIME, END_DATETIME)
    requests_by_key: dict[str, dict[str, str]] = {}
    for item in items:
        for key, value in item.properties.items():
            if key in time_keys or key in requests_by_key:
                continue
            # bool is an int subclass, and a flag is not a number to average.
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                requests_by_key[key] = {"key": key, "dtype": "str"}
            elif isinstance(value, (int, float)):
                requests_by_key[key] = {"key": key, "dtype": "float64"}
    return list(requests_by_key.values())


def group_bands_by_grid(
    items: Iterable[pystac.Item], bands: str | Sequence[str] | None = None
) -> tuple[tuple[CRS, Resolution, list[str]], ...]:
    """Group the selected bands by the CRS and resolution their `proj:*` states.

    Bands of one group load together onto their native grid: ODC picks the
    finest resolution among a load's bands, which is every band's own. Bands
    an Item lacks, or whose grid `proj:*` does not state, are left out.

    Args:
        items: Items whose bands to group, as ODC parses them.
        bands: Bands to group, as `odc.stac.load` selects them; None groups all.

    Returns:
        One `(crs, resolution, bands)` per grid, finest first.

    Examples:
        >>> for crs, resolution, bands in stac.group_bands_by_grid(items):
        ...     print(crs, resolution, bands)
        EPSG:32633 Resolution(x=10, y=-10) ['B02', 'B03', 'B04', 'B08']
        EPSG:32633 Resolution(x=20, y=-20) ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    """
    bands_by_grid: dict[tuple[CRS, Resolution], list[str]] = {}
    for parsed_item in odc.stac.parse_items(items):
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
    stamps: Sequence[str] = DEFAULT_STAMPS,
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
        stamps: Registered stamps, as for `load_dataset`.
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
    for crs, resolution, bands in group_bands_by_grid(items, options.get("bands")):
        options["bands"] = bands
        native = load_dataset(items, stamps=stamps, **options)
        loaded.append((crs, resolution, native))
    return tuple(loaded)


def _cast_bands(
    ds: xr.Dataset, dtype: Band_DType | None, nodata: float | None
) -> xr.Dataset:
    """Cast the bands to `dtype`, then rewrite each band's fill to `nodata`, as ODC would.

    Only data variables change; coordinates keep their dtypes. A band
    without a fill keeps its pixels.
    """
    if isinstance(dtype, Mapping):
        for name, band_dtype in dtype.items():
            ds[name] = ds[name].astype(band_dtype)
    elif dtype is not None:
        ds = ds.astype(dtype)

    if nodata is None:
        return ds
    for name, band in list(ds.data_vars.items()):
        fill = band.attrs.get("nodata")
        if fill is None:
            continue
        # A NaN fill never equals itself.
        missing = band.isnull() if np.isnan(fill) else band == fill
        ds[name] = band.where(~missing, nodata).assign_attrs(nodata=nodata)
    return ds
