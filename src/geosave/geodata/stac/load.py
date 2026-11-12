"""Load selected STAC Items with ODC and retain GeoSave metadata."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
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
from odc.stac.model import ParsedItem, PropertyLoadRequest

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import (
    DEFAULT_COLLECTION,
    END_DATETIME,
    STAC_PROPERTIES,
    START_DATETIME,
    TIME_COORDINATE,
)
from geosave.geodata.errors import RegionFetchError
from geosave.geodata.stac import band, item, raster as stac_raster
from geosave.geodata.utils.mapping import intersect_mappings


class StacLoadOptions(TypedDict, total=False):
    """Native ODC load options, with GeoSave's lazy and metadata defaults.

    Defaults are 1024-pixel spatial chunks. Pass chunks=None for eager reading. with_properties keeps exact STAC names and ODC's fused values.
    """

    bands: str | Sequence[str] | None
    chunks: dict[str, int | Literal["auto"]] | None
    resampling: str | dict[str, str] | None
    dtype: DTypeLike | Mapping[str, DTypeLike] | None
    nodata: float | None
    fail_on_error: bool
    with_properties: Sequence[str | Mapping[str, Any]] | None
    stac_cfg: dict[str, Any] | None
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
    mask_and_scale: bool = False,
    **options: Unpack[StacLoadOptions],
) -> xr.Dataset:
    """Load one collection's selected Items onto one grid.

    No search or Collection lookup occurs: choose the Items, including by
    time, before loading. Band metadata keeps the published digital numbers;
    decode them with `mask_and_scale=True` or `.gs.mask_and_scale()`.

    Args:
        items: Selected Items of one collection; generators are consumed once.
        geobox: Grid every band lands on. Build one around a region with
            `GeoBox.from_geopolygon(region.extent, resolution=..., crs=...)`.
        groupby: "id" keeps Items separate; "time" or "solar_day" mosaics groups.
        collection: Optional Collection metadata, supplied without network lookup.
        mask_and_scale: Decode the stamped Dataset with native xarray CF decoding.
        **options: Typed native ODC options, for reading, chunking and metadata.

    Returns:
        Dataset with CF axes, band metadata and fused per-time STAC properties.
        A label that is one Item stating `start_datetime`/`end_datetime`
        carries that period; a solar-day label carries its local day.

    Raises:
        RegionFetchError: No Items were supplied.
        ValueError: Collections conflict, selected source scales/offsets differ,
            or property options are invalid.

    Examples:
        >>> scene = stac.load_dataset(items, grid)
        >>> scene.gs.geobox == grid
        True
    """
    matched = _check_items(items, collection)
    settings = _prepare_options(options, groupby)
    sources = _band_sources(matched, settings.get("bands"), settings.get("stac_cfg"))
    _check_source_scales(sources)
    loaded = odc.stac.load(matched, geobox=geobox, **settings)
    # ODC dates a timeless Item 0001-01-01; static data has no time axis instead.
    timeless = True
    for entry in matched:
        if item.read_instant(entry) is not None:
            timeless = False
    if timeless and loaded.sizes.get("time") == 1:
        loaded = loaded.isel(time=0, drop=True)
    # An Item's own period wins; a solar-day mosaic covers its local day.
    cells = _read_item_cells(loaded, matched)
    if cells is not None:
        loaded = loaded.assign_coords(
            {START_DATETIME: ("time", cells[0]), END_DATETIME: ("time", cells[1])}
        )
    elif groupby == "solar_day":
        loaded = _write_solar_day_cells(loaded)
    loaded = _stamp_date_properties(loaded, settings.get("with_properties"), matched)
    loaded = stamp_band_attrs(matched, loaded, cfg=settings.get("stac_cfg"))
    if mask_and_scale:
        loaded = xr.decode_cf(loaded, decode_coords="all")
    return attrs.stamp_attrs(
        loaded, stac_raster.build_stac_metadata(matched, collection)
    )


def load_datatree(
    items: Iterable[pystac.Item],
    *,
    collections: Sequence[pystac.Collection] = (),
    **options: Unpack[StacLoadOptions],
) -> xr.DataTree:
    """Load Items as the STAC hierarchy: one node per asset of each Item, as stated.

    Each asset loads on the exact grid its `proj:*` states, so no band is
    resampled and Items may state different grids. A multi-band asset holds
    one variable per band. Each Item node states the Item's time, period and
    `stac_properties` once, as scalar coordinates its leaves inherit with
    ``to_dataset(inherit="all_coords")``. Load onto a chosen grid with
    `stac.load_dataset(items, geobox)` instead.

    Args:
        items: Selected Items, of one or several collections, each stating
            its bands' grids in `proj:*`.
        collections: Collection metadata for the collection nodes; never
            fetched implicitly.
        **options: Typed native ODC settings, as for `load_dataset`.

    Returns:
        DataTree of ``{collection}/{item}/{asset}`` nodes, such as
        ``sentinel-2-l2a/S2B_33UUP_20250115/B04``. Each collection node holds
        its Collection metadata; Items naming no collection sit under
        ``default``.

    Raises:
        RegionFetchError: No Items were supplied.
        ValueError: A band states no grid, or an Item ID or asset key
            contains '/', which would nest the node.

    Examples:
        >>> tree = stac.load_datatree(items)
        >>> list(tree.gs.datasets)
        ['s2/S2B_33UUP_20250115/B04', 's2/S2B_33UUP_20250115/B11']
    """
    matched = list(items)
    if not matched:
        raise RegionFetchError("no STAC Items supplied for loading")

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
        metadata = stac_raster.build_stac_metadata(entries, collection)
        nodes[identity] = xr.Dataset(attrs=metadata.to_attrs())

        # Items stating one asset grid load in one ODC call. Distinct instants
        # per call keep each time slice one Item; a timeless Item loads alone.
        batches = []
        for entry in entries:
            instant = item.read_instant(entry)
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

        facts_by_item: dict[str, dict[Hashable, xr.Variable]] = {}
        for asset, grid, names, members in batches:
            loaded = load_dataset(
                list(members.values()),
                grid,
                groupby="id",
                collection=collection,
                **{**options, "bands": names},
            )
            # Whatever varies along time is the Item's: it moves to the Item node.
            along_time = []
            for name, coord in loaded.coords.items():
                if TIME_COORDINATE in coord.dims:
                    along_time.append(name)
            if TIME_COORDINATE in loaded.dims and STAC_PROPERTIES not in along_time:
                along_time.append(STAC_PROPERTIES)
            for position in range(loaded.sizes.get(TIME_COORDINATE, 1)):
                observation = loaded
                if TIME_COORDINATE in loaded.dims:
                    observation = loaded.isel({TIME_COORDINATE: position})
                instant = None
                if TIME_COORDINATE in observation.coords:
                    instant = pd.Timestamp(observation[TIME_COORDINATE].values)
                entry = members[instant]
                leaf = _place_asset(observation, entry)
                facts = facts_by_item.setdefault(entry.id, {})
                for name in along_time:
                    if name in leaf.coords:
                        facts[name] = leaf[name].variable
                nodes[f"{identity}/{entry.id}/{asset}"] = leaf.drop_vars(
                    along_time, errors="ignore"
                )
        for entry in entries:
            nodes[f"{identity}/{entry.id}"] = xr.Dataset(coords=facts_by_item[entry.id])
    return xr.DataTree.from_dict(nodes)


def _place_asset(observation: xr.Dataset, entry: pystac.Item) -> xr.Dataset:
    """Make one Item's slice of a batch load its asset, as the Item states it."""
    # A batch fuses only the properties its Items share; the Item states its own.
    stated = stac_raster.read_properties_coordinate(observation)
    fields = {
        **(stated[0] if stated else {}),
        **item.select_item_properties(entry.properties),
    }
    if fields:
        observation = stac_raster.stamp_properties_coordinate(observation, [fields])
    observation = attrs.clear_attrs(observation, attrs.StacCollection)
    # The Item's geometry is each asset's region; the grids stay whole.
    footprint = item.read_footprint([entry])
    if footprint is not None:
        observation = observation.gs.write_extent(footprint)
    # A tree-published asset key carries its node path; the band is the last part.
    bands = {}
    for name in observation.data_vars:
        bands[name] = str(name).rsplit("/", 1)[-1]
    return observation.rename(bands)


def _read_asset_grids(
    entry: pystac.Item, options: Mapping[str, Any]
) -> dict[str, tuple[GeoBox, list[str]]]:
    """Read each asset's stated grid and the selected bands it holds."""
    grids: dict[str, tuple[GeoBox, list[str]]] = {}
    (parsed,) = odc.stac.parse_items([entry], cfg=options.get("stac_cfg"))
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


def _read_item_cells(
    loaded: xr.Dataset, items: Sequence[pystac.Item]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read each label's cell as the period its one Item states.

    ODC labels each Item at its instant. Returns None unless every label is
    exactly one Item and every Item states `start_datetime`/`end_datetime`.
    """
    if "time" not in loaded.dims or loaded.sizes["time"] != len(items):
        return None
    periods = {}
    for entry in items:
        if (
            "start_datetime" not in entry.properties
            or "end_datetime" not in entry.properties
        ):
            return None
        instant = item.read_instant(entry)
        if instant in periods:
            return None
        periods[instant] = item.read_item_period(entry)
    starts = []
    ends = []
    for label in pd.DatetimeIndex(loaded.time.values):
        if label not in periods:
            return None
        start, end = periods[label]
        starts.append(start)
        ends.append(end)
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


def _prepare_options(
    options: Mapping[str, Any],
    groupby: str | Callable[[pystac.Item, ParsedItem, int], Any],
) -> dict[str, Any]:
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
        requests.append(fields)
    if requests:
        settings["with_properties"] = requests
    return settings


def _stamp_date_properties(
    loaded: xr.Dataset,
    properties: Sequence[str | Mapping[str, Any]] | None,
    items: Sequence[pystac.Item],
) -> xr.Dataset:
    """Store shared source properties and ODC summaries on each output observation."""
    source_properties = []
    for entry in items:
        source_properties.append(item.select_item_properties(entry.properties))
    shared = intersect_mappings(source_properties)

    keys = []
    for request in PropertyLoadRequest.from_user_input(properties or ()):
        keys.append(request.key)
    if not keys and not shared:
        return loaded
    values_by_key = {}
    for key in keys:
        values_by_key[key] = np.atleast_1d(loaded[key].values)
    mappings = []
    for index in range(loaded.sizes.get("time", 1)):
        fields = dict(shared)
        for key, values in values_by_key.items():
            value = values[index].item()
            if (isinstance(value, float) and np.isnan(value)) or value == "None":
                continue
            fields[key] = value
        mappings.append(fields)
    return stac_raster.stamp_properties_coordinate(loaded.drop_vars(keys), mappings)


def stamp_band_attrs(
    items: Sequence[pystac.Item],
    loaded: xr.Dataset,
    cfg: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Attach the band metadata every Item agrees on, keeping the loader's pixel metadata.

    Args:
        items: Items odc-stac loaded `loaded` from.
        loaded: Dataset odc-stac returned, one variable per band.
        cfg: The `stac_cfg` the load used, so band names resolve as it resolved them.

    Returns:
        `loaded` with each variable's agreed STAC band attrs, nodata in CF's spelling.

    Warns:
        DroppedAttrsWarning: Items disagree on a band's metadata.
    """

    sources = _band_sources(items, [str(name) for name in loaded.data_vars], cfg)
    stamped = {}
    for name, variable in loaded.data_vars.items():
        metadata = band.read_shared_attrs(sources[str(name)])
        stated = variable.assign_attrs({**metadata, **variable.attrs})

        # The declaration must match the fill ODC casts into the output dtype.
        fill = stated.attrs.pop("nodata", stated.attrs.get("_FillValue"))
        if fill is not None:
            stated = stated.rio.write_nodata(fill)
        stamped[name] = stated
    return loaded.assign(stamped)


def _band_sources(
    items: Sequence[pystac.Item],
    bands: str | Sequence[str] | None,
    cfg: dict[str, Any] | None,
) -> dict[str, list[tuple[pystac.Item, str, int]]]:
    """Resolve selected raster bands with the same collection aliases as ODC."""
    parsed = list(odc.stac.parse_items(items, cfg=cfg))
    names = parsed[0].collection.normalize_band_query(bands)
    sources = {}
    for name in names:
        sources[name] = []
        for entry, odc_item in zip(items, parsed, strict=True):
            key, index = odc_item.collection.band_key(name)
            if key in entry.assets and (key, index) in odc_item.bands:
                sources[name].append((entry, key, index - 1))
    return sources


def _check_source_scales(
    sources: Mapping[str, Sequence[tuple[pystac.Item, str, int]]],
) -> None:
    """Reject digital numbers that cannot share one scale and offset after fusion."""
    for name, entries in sources.items():
        expected = None
        for entry, key, index in entries:
            stated = attrs.CFPacking.model_validate(
                band.read_attrs(entry.assets[key], index)
            )
            # Omitted STAC scale/offset mean identity conversion. Nodata is
            # deliberately excluded: ODC remaps source sentinels on loading.
            scale = (stated.scale_factor, stated.add_offset)
            if expected is not None and scale != expected:
                raise ValueError(
                    f"band {name!r} has incompatible source scale/offset: "
                    f"{expected} and {scale} in Item {entry.id!r}; "
                    "load and decode the sources separately before combining them"
                )
            expected = scale
