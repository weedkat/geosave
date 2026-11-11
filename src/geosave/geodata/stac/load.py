"""Load selected STAC Items with ODC and retain GeoSave metadata."""

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
from odc.stac.model import ParsedItem, PropertyLoadRequest

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import END_DATETIME, START_DATETIME, to_yx
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
    **options: Unpack[StacLoadOptions],
) -> xr.Dataset:
    """Load one collection's selected Items onto one grid.

    No search or Collection lookup occurs: choose the Items, including by
    time, before loading. Band metadata keeps the published digital numbers;
    decode them with `.gs.mask_and_scale()`.

    Args:
        items: Selected Items of one collection; generators are consumed once.
        geobox: Grid every band lands on. Build one around a region with
            `GeoBox.from_geopolygon(region.extent, resolution=..., crs=...)`.
        groupby: "id" keeps Items separate; "time" or "solar_day" mosaics groups.
        collection: Optional Collection metadata, supplied without network lookup.
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
    loaded = to_yx(odc.stac.load(matched, geobox=geobox, **settings))
    # ODC dates a timeless Item 0001-01-01; static data has no time axis instead.
    timeless = True
    for entry in matched:
        if item.read_instant(entry) is not None:
            timeless = False
    if timeless and loaded.sizes.get("time") == 1:
        loaded = loaded.isel(time=0, drop=True)
    if "time" in loaded.coords:
        loaded = loaded.assign_coords(
            time=loaded.time.assign_attrs(standard_name="time", axis="T")
        )
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
    loaded = loaded.gs.write_crs()
    return attrs.stamp_attrs(
        loaded, stac_raster.build_stac_metadata(matched, collection)
    )


def load_datatree(
    items: Iterable[pystac.Item],
    *,
    groupby: str | Callable[[pystac.Item, ParsedItem, int], Any] = "solar_day",
    collection: pystac.Collection | None = None,
    **options: Unpack[StacLoadOptions],
) -> xr.DataTree:
    """Load one collection as one node per STAC asset, each on its own grid.

    An asset loads on the exact grid its `proj:*` states, so no band is
    resampled. A multi-band asset holds one variable per band. Observation
    coordinates every node shares, such as `time`, sit on the collection node,
    which xarray then keeps aligned across the asset nodes. Merge assets on
    one grid with `tree.gs.group_grid()`; crop with `tree.gs.crop(region)`.

    Args:
        items: Selected Items of one collection, each stating its bands'
            grids in `proj:*`.
        groupby: "id" keeps Items separate; "time" or "solar_day" mosaics groups.
        collection: Optional Collection metadata; never fetched implicitly.
        **options: Typed native ODC settings, as for `load_dataset`.

    Returns:
        DataTree of ``{collection}/{asset}`` nodes, such as
        ``sentinel-2-l2a/B04``, with the Collection metadata and shared
        coordinates on the collection node; Items naming no collection place
        their ``{asset}`` nodes at the root.

    Raises:
        ValueError: Items mix collections, a band states no grid, or Items
            state different grids for one asset.

    Examples:
        >>> tree = stac.load_datatree(items)
        >>> list(tree.gs.datasets)
        ['sentinel-2-l2a/B04', 'sentinel-2-l2a/B08', 'sentinel-2-l2a/B11']
        >>> list(tree.gs.group_grid().gs.datasets)
        ['sentinel-2-l2a/r10m', 'sentinel-2-l2a/r20m']
    """
    matched = _check_items(items, collection)
    identity = collection.id if collection is not None else matched[0].collection_id

    # Each asset's grid, as every Item states it in proj:*.
    grid_by_asset: dict[str, GeoBox] = {}
    bands_by_asset: dict[str, list[str]] = {}
    for parsed in odc.stac.parse_items(matched, cfg=options.get("stac_cfg")):
        for name, source in parsed.resolve_bands(options.get("bands")).items():
            grid = source.geobox if isinstance(source, RasterSource) else None
            if grid is None:
                raise ValueError(
                    f"{parsed.id}'s band {name!r} states no grid in proj:*; load it "
                    "onto a chosen grid with stac.load_dataset(items, geobox)"
                )
            asset = parsed.collection.band_key(name)[0]
            if grid_by_asset.setdefault(asset, grid) != grid:
                raise ValueError(
                    f"Items state different grids for asset {asset!r}; load them "
                    "onto one chosen grid with stac.load_dataset(items, geobox)"
                )
            names = bands_by_asset.setdefault(asset, [])
            if name not in names:
                names.append(name)
    if not bands_by_asset:
        raise ValueError("no raster bands selected")

    assets = {}
    metadata: attrs.StacMetadata | None = None
    for asset, names in bands_by_asset.items():
        loaded = load_dataset(
            matched,
            grid_by_asset[asset],
            groupby=groupby,
            collection=collection,
            **{**options, "bands": names},
        )
        # Collection metadata moves to the collection node.
        metadata = attrs.StacMetadata.from_attrs(loaded.attrs)
        loaded = attrs.clear_attrs(loaded, attrs.StacMetadata)
        # A tree-published asset key carries its node path; the band is the last part.
        bands = {}
        for name in loaded.data_vars:
            bands[name] = str(name).rsplit("/", 1)[-1]
        assets[asset] = loaded.rename(bands)

    stated = {} if metadata is None else metadata.to_attrs()
    return _build_collection_tree(identity, stated, assets)


def _build_collection_tree(
    identity: str | None, metadata: Mapping[str, Any], grids: Mapping[str, xr.Dataset]
) -> xr.DataTree:
    """Place asset nodes under their collection, sharing coordinates they agree on.

    Items naming no collection place their asset nodes at the root.
    """
    candidates = []
    for raster in grids.values():
        coords = {}
        for name, coord in raster.coords.items():
            # Spatial axes and the CRS belong to each grid; scalars need no sharing.
            if coord.ndim == 0 or set(coord.dims) & {"y", "x"}:
                continue
            # The variable alone: a coordinate array carries the CRS along with it.
            coords[name] = coord.variable
        candidates.append(coords)
    shared = intersect_mappings(candidates)
    parent = identity or "/"
    nodes = {parent: xr.Dataset(coords=shared, attrs=dict(metadata))}
    for path, raster in grids.items():
        node = path if identity is None else f"{identity}/{path}"
        nodes[node] = raster.drop_vars(list(shared))
    return xr.DataTree.from_dict(nodes)


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

        # ODC states its fill as `nodata`; normalize it to CF's spelling.
        packing = attrs.Packing.from_attrs(stated.attrs)
        stamped[name] = (
            stated if packing is None else attrs.stamp_attrs(stated, packing)
        )
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
            stated = attrs.Packing.model_validate(
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
