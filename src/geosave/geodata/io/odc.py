"""Load selected STAC Items with ODC and retain GeoSave metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict, Unpack

import numpy as np
import odc.stac
import pystac
import xarray as xr
from numpy.typing import DTypeLike
from odc.geo import SomeCRS, SomeResolution
from odc.geo.geobox import GeoBox, GeoboxAnchor
from odc.loader.types import RasterSource, ReaderDriverSpec
from odc.stac.model import ParsedItem, PropertyLoadRequest

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import to_yx
from geosave.geodata.errors import AnchorFetchError
from geosave.geodata.stac import band, item


class StacLoadOptions(TypedDict, total=False):
    """Native ODC load options, with GeoSave's lazy and metadata defaults.

    Defaults are solar-day grouping and 1024-pixel spatial chunks. Pass
    chunks=None for eager reading. Grid anchor means pixel snapping, not
    GeoAnchor. with_properties keeps exact STAC names and ODC's fused values.
    """

    bands: str | Sequence[str] | None
    groupby: str | Callable[[pystac.Item, ParsedItem, int], Any] | None
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
    crs: SomeCRS | None
    resolution: SomeResolution | None
    anchor: GeoboxAnchor | None
    geopolygon: Any
    intersects: Any
    bbox: tuple[float, float, float, float] | None
    lon: tuple[float, float] | None
    lat: tuple[float, float] | None
    x: tuple[float, float] | None
    y: tuple[float, float] | None
    like: xr.Dataset | xr.DataArray | GeoBox | None


def load_dataset(
    items: Iterable[pystac.Item],
    *,
    geobox: GeoBox | None = None,
    collection: pystac.Collection | None = None,
    **options: Unpack[StacLoadOptions],
) -> xr.Dataset:
    """Load one collection's selected Items onto an explicit or ODC-derived grid.

    Geometry plus CRS/resolution constructs a rectangular grid; it does not
    mask the polygon. GeoBox specifies exact pixel placement. No search or
    Collection lookup occurs. Band metadata preserves published digital numbers.

    Args:
        items: Selected Items; generators are consumed once.
        geobox: Exact output grid, mutually exclusive with other grid options.
        collection: Optional Collection metadata, supplied without network lookup.
        **options: Typed native ODC options, including grid construction settings.

    Returns:
        Dataset with CF axes, band metadata and fused per-time STAC properties.

    Raises:
        AnchorFetchError: No Items were supplied.
        ValueError: Collections conflict or grid/property options are invalid.
    """
    matched = _check_items(items, collection)
    settings = _prepare_options(options)
    loaded = _load(matched, geobox, settings)
    return attrs.stamp_attrs(loaded, item.build_stac_metadata(matched, collection))


def load_datatree(
    items: Iterable[pystac.Item],
    *,
    geobox: GeoBox | None = None,
    collection: pystac.Collection | None = None,
    **options: Unpack[StacLoadOptions],
) -> xr.DataTree:
    """Load collection/grid nodes while preserving relative resolution families.

    The finest selected family uses the requested grid; coarser families use
    grids scaled by their source resolution ratios. This does not guarantee
    absolute native resolution or original pixel alignment. Family metadata is
    read from the first Item, preserving the existing loading contract.

    Args:
        items: Selected Items belonging to one identified collection.
        geobox: Exact grid for the finest family, or None for ODC grid construction.
        collection: Optional Collection metadata; never fetched implicitly.
        **options: Typed native ODC settings, as for load_dataset.

    Returns:
        DataTree with Collection metadata on its parent and pixels on grid leaves.
    """
    from geosave.geodata.core.raster.grid import name_grid

    matched = _check_items(items, collection)
    identity = collection.id if collection is not None else matched[0].collection_id
    if not identity:
        raise ValueError("load_datatree needs a collection ID or Collection")
    settings = _prepare_options(options)
    parsed = next(
        iter(odc.stac.parse_items([matched[0]], cfg=settings.get("stac_cfg")))
    )
    families: dict[float, list[str]] = {}
    for name, source in parsed.resolve_bands(settings.get("bands")).items():
        grid = source.geobox if isinstance(source, RasterSource) else None
        if grid is None:
            raise ValueError(f"{identity}'s band {name!r} states no native grid")
        families.setdefault(abs(grid.resolution.x), []).append(name)
    if not families:
        raise ValueError("no raster bands selected")
    finest = min(families)
    nodes = {
        identity: xr.Dataset(
            attrs=item.build_stac_metadata(matched, collection).to_attrs()
        )
    }
    target = geobox
    for size, names in sorted(families.items()):
        if size == finest:
            loaded = _load(matched, target, {**settings, "bands": names})
            target = loaded.gs.geobox
        else:
            assert target is not None
            grid = target.zoom_out(size / finest)
            family_options = dict(settings)
            for key in (
                "crs",
                "resolution",
                "anchor",
                "geopolygon",
                "intersects",
                "bbox",
                "lon",
                "lat",
                "x",
                "y",
                "like",
            ):
                family_options.pop(key, None)
            family_options["bands"] = names
            loaded = _load(matched, grid, family_options)
        nodes[f"{identity}/{name_grid(loaded.gs.geobox)}"] = loaded
    return xr.DataTree.from_dict(nodes)


def _check_items(items, collection) -> list[pystac.Item]:
    matched = list(items)
    if not matched:
        raise AnchorFetchError("no STAC Items supplied for loading")
    identities = {
        entry.collection_id for entry in matched if entry.collection_id is not None
    }
    if len(identities) > 1:
        raise ValueError("load Items from one collection per call")
    if collection is not None and identities and identities != {collection.id}:
        raise ValueError("Collection metadata does not match the supplied Items")
    return matched


def _prepare_options(options: Mapping[str, Any]) -> dict[str, Any]:
    settings = {"groupby": "solar_day", "chunks": {"x": 1024, "y": 1024}, **options}
    requests = []
    for request in PropertyLoadRequest.from_user_input(
        options.get("with_properties") or ()
    ):
        if request.name is not None and request.name != request.key:
            raise ValueError("with_properties must keep the exact STAC key as its name")
    for request in options.get("with_properties") or ():
        fields = {"key": request} if isinstance(request, str) else dict(request)
        fields["name"] = fields["key"]
        requests.append(fields)
    if requests:
        settings["with_properties"] = requests
    return settings


def _load(matched, geobox, options) -> xr.Dataset:
    loaded = to_yx(odc.stac.load(matched, geobox=geobox, **options))
    loaded = _stamp_date_properties(loaded, options.get("with_properties"))
    loaded = stamp_band_attrs(matched, loaded, cfg=options.get("stac_cfg"))
    return loaded.gs.write_crs()


def _stamp_date_properties(loaded, properties) -> xr.Dataset:
    keys = []
    for request in PropertyLoadRequest.from_user_input(properties or ()):
        keys.append(request.key)
    if not keys:
        return loaded
    values_by_key = {}
    for key in keys:
        values_by_key[key] = np.atleast_1d(loaded[key].values)
    mappings = []
    for index in range(loaded.sizes["time"]):
        properties = {}
        for key, values in values_by_key.items():
            value = values[index].item()
            if (isinstance(value, float) and np.isnan(value)) or value == "None":
                continue
            properties[key] = value
        mappings.append(properties)
    return item.stamp_properties_coordinate(loaded.drop_vars(keys), mappings)


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
    import odc.stac

    parsed = list(odc.stac.parse_items(items, cfg=cfg))
    stamped = {}
    for name, variable in loaded.data_vars.items():
        sources = []
        for entry, odc_item in zip(items, parsed, strict=True):
            key, index = odc_item.collection.band_key(str(name))
            sources.append((entry, key, index - 1))
        metadata = band.read_shared_attrs(variable, sources)
        stated = variable.assign_attrs({**metadata, **variable.attrs})

        # ODC states its fill as `nodata`; normalize it to CF's spelling.
        nodata = attrs.Nodata.from_attrs(stated.attrs)
        stamped[name] = stated if nodata is None else attrs.stamp_attrs(stated, nodata)
    return loaded.assign(stamped)
