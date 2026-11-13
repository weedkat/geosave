"""Load selected STAC Items with ODC and write what their documents state."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict, Unpack

import numpy as np
import odc.stac
import pandas as pd
import pystac
import xarray as xr
from numpy.typing import DTypeLike
from odc.geo.geobox import GeoBox
from odc.loader.types import RasterSource, ReaderDriverSpec
from odc.stac.model import ParsedItem

import geosave.geodata.attrs as attrs
from geosave.geodata.core.conventions import (
    DEFAULT_COLLECTION,
    END_DATETIME,
    START_DATETIME,
    TIME_COORDINATE,
)
from geosave.geodata.errors import RegionFetchError
from geosave.geodata.core.vector.utils.time import read_timespan
from geosave.geodata.stac.stamps import (
    DEFAULT_STAMPS,
    LoadSource,
    check_stamps,
    run_stamps,
)
from geosave.geodata.stac.stamps.collection import stamp_collection
from geosave.geodata.stac.stamps.properties import stamp_properties


class StacLoadOptions(TypedDict, total=False):
    """Native ODC load options, with GeoSave's lazy and metadata defaults.

    Defaults are 1024-pixel spatial chunks; pass chunks=None for eager reading.
    A `with_properties` request becomes a coordinate named by its exact STAC
    key, float64 unless the request states its own dtype.
    """

    bands: str | Sequence[str] | None
    chunks: dict[str, int | Literal["auto"]] | None
    resampling: str | dict[str, str] | None
    dtype: DTypeLike | Mapping[str, DTypeLike] | None
    nodata: float | None
    fail_on_error: bool
    with_properties: Sequence[str | Mapping[str, Any]] | None
    pool: int | ThreadPoolExecutor | None
    progress: Any
    patch_url: Callable[[str], str] | None
    preserve_original_order: bool
    driver: ReaderDriverSpec | None
    fuse_func: str | Mapping[str, str | None] | None


def load_dataset(
    items: Iterable[pystac.Item],
    geobox: GeoBox,
    *,
    groupby: str | Callable[[pystac.Item, ParsedItem, int], Any] = "solar_day",
    collection: pystac.Collection | None = None,
    stamps: Sequence[str] = DEFAULT_STAMPS,
    mask_and_scale: bool = False,
    **options: Unpack[StacLoadOptions],
) -> xr.Dataset:
    """Load one collection's selected Items onto one grid.

    No search or Collection lookup occurs: choose the Items, including by
    time, before loading. The named stamps then write what the STAC documents
    state, reading no pixels. Band metadata keeps the published digital
    numbers; decode them with `mask_and_scale=True` or `.gs.mask_and_scale()`.

    Args:
        items: Selected Items of one collection; generators are consumed once.
        geobox: Grid every band lands on. Build one around a region with
            `GeoBox.from_geopolygon(region.extent, resolution=..., crs=...)`.
        groupby: "id" keeps Items separate; "time" or "solar_day" mosaics groups.
        collection: Optional Collection metadata, supplied without network lookup.
        stamps: Registered stamps to run, in order; see `stac.stamps`.
        mask_and_scale: Decode the stamped Dataset with native xarray CF decoding.
        **options: Typed native ODC options, for reading, chunking and properties.

    Returns:
        Dataset with CF axes and what the stamps write: by default band
        metadata, Item properties as coordinates marked `stac_property`, and
        the Collection. A step that is one Item stating `start_datetime` and
        `end_datetime` carries that period; a solar-day step its local day.

    Raises:
        RegionFetchError: No Items were supplied.
        KeyError: A stamp name is not registered; raised before loading.
        TypeError: An option is not a supported ODC option.
        ValueError: Collections conflict, a stamp refuses the Items (such as
            differing source scales/offsets), or property options are invalid.

    Examples:
        >>> scene = stac.load_dataset(items, grid, with_properties=["eo:cloud_cover"])
        >>> scene.gs.stac_properties
        {'eo:cloud_cover': 'eo:cloud_cover', 'platform': 'platform'}
    """
    matched = _check_items(items, collection)
    check_stamps(stamps)
    settings = _prepare_options(options, groupby)
    source = LoadSource(matched, collection)

    loaded = odc.stac.load(matched, geobox=geobox, **settings)
    loaded = _write_time_axis(loaded, source, groupby)
    loaded = _write_requested_properties(loaded, settings["with_properties"])
    loaded = run_stamps(loaded, source, stamps)

    if mask_and_scale:
        loaded = xr.decode_cf(loaded, decode_coords="all")
    return loaded


def load_datatree(
    items: Iterable[pystac.Item],
    *,
    collections: Sequence[pystac.Collection] = (),
    stamps: Sequence[str] = DEFAULT_STAMPS,
    **options: Unpack[StacLoadOptions],
) -> xr.DataTree:
    """Load Items as the STAC hierarchy: one node per asset of each Item, as stated.

    Each asset loads on the exact grid its `proj:*` states, so no band is
    resampled and Items may state different grids. A multi-band asset holds
    one variable per band. Each Item node holds the Item's time, period and
    properties, as scalar coordinates its leaves inherit with
    ``to_dataset(inherit="all_coords")``. Load onto a chosen grid with
    `stac.load_dataset(items, geobox)` instead.

    Args:
        items: Selected Items, of one or several collections, each stating
            its bands' grids in `proj:*`.
        collections: Collection metadata for the collection nodes; never
            fetched implicitly.
        stamps: Registered stamps each asset load runs, as for `load_dataset`.
        **options: Typed native ODC settings, as for `load_dataset`.

    Returns:
        DataTree of ``{collection}/{item}/{asset}`` nodes, such as
        ``sentinel-2-l2a/S2B_33UUP_20250115/B04``. Each collection node holds
        its Collection metadata; Items naming no collection sit under
        ``default``.

    Raises:
        RegionFetchError: No Items were supplied.
        KeyError: A stamp name is not registered.
        ValueError: A band states no grid, or an Item ID or asset key
            contains '/', which would nest the node.

    Examples:
        >>> tree = stac.load_datatree(items)
        >>> [leaf.path for leaf in tree.leaves]
        ['/s2/S2B_33UUP_20250115/B04', '/s2/S2B_33UUP_20250115/B11']
    """
    matched = list(items)
    if not matched:
        raise RegionFetchError("no STAC Items supplied for loading")
    check_stamps(stamps)

    items_by_collection: dict[str, list[pystac.Item]] = {}
    for entry in matched:
        identity = entry.collection_id or DEFAULT_COLLECTION
        items_by_collection.setdefault(identity, []).append(entry)

    nodes = {}
    for identity, entries in items_by_collection.items():
        collection = None
        for stated in collections:
            if stated.id == identity:
                collection = stated
        nodes[identity] = stamp_collection(
            xr.Dataset(), LoadSource(entries, collection)
        )

        for asset, grid, names, members in _batch_assets(entries, options):
            loaded = load_dataset(
                list(members.values()),
                grid,
                groupby="id",
                collection=collection,
                stamps=stamps,
                **{**options, "bands": names},
            )
            for position in range(loaded.sizes.get(TIME_COORDINATE, 1)):
                observation = loaded
                if TIME_COORDINATE in loaded.dims:
                    observation = loaded.isel({TIME_COORDINATE: position})
                instant = None
                if TIME_COORDINATE in observation.coords:
                    instant = pd.Timestamp(observation[TIME_COORDINATE].values)
                entry = members[instant]

                nodes[f"{identity}/{entry.id}"] = _build_item_node(observation, entry)
                nodes[f"{identity}/{entry.id}/{asset}"] = _build_asset_node(observation)
    return xr.DataTree.from_dict(nodes)


def _batch_assets(
    entries: Sequence[pystac.Item], options: Mapping[str, Any]
) -> list[tuple[str, GeoBox, list[str], dict[pd.Timestamp | None, pystac.Item]]]:
    """Group Items whose asset states one grid and bands, so they load in one call.

    Distinct instants per batch keep each time step one Item; a timeless Item
    loads alone.

    Raises:
        ValueError: An Item ID or asset key contains '/', which would nest a node.
    """
    batches = []
    for entry in entries:
        instant = _read_instant(entry)
        for asset, (grid, names) in _read_asset_grids(entry, options).items():
            if "/" in entry.id or "/" in asset:
                raise ValueError(
                    f"Item {entry.id!r} asset {asset!r}: a name with '/' cannot "
                    "name a tree node; STAC IDs and asset keys should avoid '/'"
                )
            joined = False
            for key, box, bands, members in batches:
                if instant is None or (key, box, bands) != (asset, grid, names):
                    continue
                if instant in members:
                    continue
                members[instant] = entry
                joined = True
                break
            if not joined:
                batches.append((asset, grid, names, {instant: entry}))
    return batches


def _build_item_node(observation: xr.Dataset, entry: pystac.Item) -> xr.Dataset:
    """Return the Item node: the Item's time and period, and its own properties."""
    times = {}
    for name in (TIME_COORDINATE, START_DATETIME, END_DATETIME):
        if name in observation.coords:
            times[name] = observation[name].variable
    return stamp_properties(xr.Dataset(coords=times), LoadSource([entry]))


def _build_asset_node(observation: xr.Dataset) -> xr.Dataset:
    """Return the asset leaf: its pixels and grid; time and properties sit on the Item."""
    moved = [TIME_COORDINATE, START_DATETIME, END_DATETIME]
    moved.extend(observation.gs.stac_properties.values())
    leaf = observation.drop_vars(moved, errors="ignore")
    leaf = attrs.clear_attrs(leaf, attrs.StacCollection)

    # A tree-published asset key carries its node path; the band is the last part.
    bands = {}
    for name in leaf.data_vars:
        bands[name] = str(name).rsplit("/", 1)[-1]
    return leaf.rename(bands)


def _read_asset_grids(
    entry: pystac.Item, options: Mapping[str, Any]
) -> dict[str, tuple[GeoBox, list[str]]]:
    """Read each asset's stated grid and the selected bands it holds."""
    grids: dict[str, tuple[GeoBox, list[str]]] = {}
    (parsed,) = odc.stac.parse_items([entry])
    for name, source in parsed.resolve_bands(options.get("bands")).items():
        grid = source.geobox if isinstance(source, RasterSource) else None
        if grid is None:
            raise ValueError(
                f"{parsed.id}'s band {name!r} states no grid in proj:*; load it "
                "onto a chosen grid with stac.load_dataset(items, geobox)"
            )
        asset = parsed.collection.band_key(name)[0]
        grids.setdefault(asset, (grid, []))[1].append(name)
    if not grids:
        raise ValueError("no raster bands selected")
    return grids


def _check_items(
    items: Iterable[pystac.Item], collection: pystac.Collection | None
) -> list[pystac.Item]:
    """Return the Items as a list, refusing none or more than one collection."""
    matched = list(items)
    if not matched:
        raise RegionFetchError("no STAC Items supplied for loading")
    identities = {
        entry.collection_id for entry in matched if entry.collection_id is not None
    }
    if len(identities) > 1:
        raise ValueError("load Items from one collection per call")
    if collection is not None and identities and identities != {collection.id}:
        raise ValueError("Collection metadata does not match the supplied Items")
    return matched


def _read_instant(entry: pystac.Item) -> pd.Timestamp | None:
    """Return the instant ODC labels an Item by, naive UTC; None if timeless."""
    span = read_timespan(
        entry.datetime,
        entry.properties.get("start_datetime"),
        entry.properties.get("end_datetime"),
    )
    if span is None:
        return None
    if entry.datetime is None:
        return span[0]
    return pd.to_datetime(entry.datetime, utc=True).tz_convert(None)


def _write_time_axis(
    loaded: xr.Dataset,
    source: LoadSource,
    groupby: str | Callable[[pystac.Item, ParsedItem, int], Any],
) -> xr.Dataset:
    """Drop the time axis of timeless Items, else state each step's period.

    ODC dates a timeless Item 0001-01-01; static data has no time axis instead.
    """
    timeless = True
    for entry in source.items:
        if _read_instant(entry) is not None:
            timeless = False
    if timeless and loaded.sizes.get(TIME_COORDINATE) == 1:
        return loaded.isel({TIME_COORDINATE: 0}, drop=True)

    cells = _read_item_cells(loaded, source.items)
    if cells is not None:
        starts, ends = cells
        return loaded.assign_coords(
            {
                START_DATETIME: (TIME_COORDINATE, starts),
                END_DATETIME: (TIME_COORDINATE, ends),
            }
        )
    if groupby == "solar_day":
        return _write_solar_day_cells(loaded)
    return loaded


def _read_item_cells(
    loaded: xr.Dataset, items: Sequence[pystac.Item]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read each step's period as its one Item states it.

    Returns None unless every step is exactly one Item and every Item states
    `start_datetime` and `end_datetime`.
    """
    if TIME_COORDINATE not in loaded.dims or loaded.sizes[TIME_COORDINATE] != len(
        items
    ):
        return None

    periods = {}
    for entry in items:
        span = read_timespan(
            entry.datetime,
            entry.properties.get("start_datetime"),
            entry.properties.get("end_datetime"),
        )
        instant = _read_instant(entry)
        stated = (
            "start_datetime" in entry.properties and "end_datetime" in entry.properties
        )
        if not stated or span is None or instant in periods:
            return None
        periods[instant] = span

    starts = []
    ends = []
    for label in pd.DatetimeIndex(loaded[TIME_COORDINATE].values):
        if label not in periods:
            return None
        starts.append(periods[label][0])
        ends.append(periods[label][1])
    return pd.DatetimeIndex(starts).to_numpy(), pd.DatetimeIndex(ends).to_numpy()


def _write_solar_day_cells(loaded: xr.Dataset) -> xr.Dataset:
    """State each label's cell as the local solar day ODC grouped it by.

    ODC shifts UTC by whole hours of the load grid's central longitude
    (`odc.stac.model._convert_to_solar_time`); the day runs midnight to
    midnight in that shifted time.
    """
    if "time" not in loaded.dims:
        return loaded
    ((longitude, _),) = loaded.odc.geobox.extent.centroid.to_crs("EPSG:4326").points
    offset = pd.Timedelta(hours=int(longitude / 15))
    labels = pd.DatetimeIndex(loaded.time.values)
    starts = (labels + offset).normalize() - offset
    ends = starts + pd.Timedelta(days=1)
    return loaded.assign_coords(
        {
            START_DATETIME: ("time", starts.to_numpy()),
            END_DATETIME: ("time", ends.to_numpy()),
        }
    )


def _write_requested_properties(
    loaded: xr.Dataset, requests: Sequence[Mapping[str, Any]]
) -> xr.Dataset:
    """Turn each property ODC loaded as a variable into a marked coordinate."""
    for request in requests:
        key = request["key"]
        values = loaded[key].values
        loaded = loaded.drop_vars(key)
        if values.ndim == 0:
            values = values.item()
        loaded = loaded.gs.write_stac_property(key, values)
    return loaded


def _prepare_options(
    options: Mapping[str, Any],
    groupby: str | Callable[[pystac.Item, ParsedItem, int], Any],
) -> dict[str, Any]:
    """Return ODC's load settings with GeoSave's defaults and property requests.

    Raises:
        TypeError: An option is not a supported ODC option.
        ValueError: A property request renames its STAC key.
    """
    unsupported = options.keys() - StacLoadOptions.__annotations__.keys()
    if unsupported:
        raise TypeError(f"unsupported load options: {', '.join(sorted(unsupported))}")
    settings = {"groupby": groupby, "chunks": {"x": 1024, "y": 1024}, **options}

    requests = []
    for request in options.get("with_properties") or ():
        fields = {"key": request} if isinstance(request, str) else dict(request)
        # ODC names a property variable after `name`; the STAC key keeps it exact.
        if fields.get("name", fields["key"]) != fields["key"]:
            raise ValueError("with_properties must keep the exact STAC key as its name")
        fields["name"] = fields["key"]
        # ODC's float32 default turns a cloud cover of 24.001125 into 24.0.
        fields.setdefault("dtype", "float64")
        requests.append(fields)
    settings["with_properties"] = requests
    return settings
