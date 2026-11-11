"""Publish rasters as STAC Items and assets, and read the assets Items describe."""

from __future__ import annotations

from geosave.geodata.core.raster.transform.packing import mask_missing

from collections.abc import Hashable, Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime as DateTime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pystac
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave.geodata import attrs
from geosave.geodata.io.storage import Location, StorageOptions, location
from geosave.geodata.stac import band, item
from geosave.geodata.utils.mapping import intersect_mappings


def read_scene_pixels(
    entry: pystac.Item,
    geobox: GeoBox,
    variables: Sequence[str] | None = None,
) -> xr.Dataset:
    """Read an Item's pixels inside `geobox`, each file from the overview it needs.

    Only the window of each file that `geobox` overlaps is read, from the
    coarsest overview still at least as fine as `geobox`, then reprojected onto it.

    Args:
        entry: Native STAC Item.
        geobox: Grid to read onto, such as a map's view.
        variables: Variables to read. None reads every data asset.

    Returns:
        Dataset on `geobox`, one float32 variable per requested name, nodata
        and pixels outside the files as NaN.

    Raises:
        KeyError: A requested variable is not in the Item.
    """
    import numpy as np

    names_by_asset = item.read_variable_names(entry)
    known_names = []
    for names in names_by_asset.values():
        known_names.extend(names)
    requested_names = known_names if variables is None else list(variables)
    for name in requested_names:
        if name not in known_names:
            raise KeyError(f"{entry.id} has no variable {name!r}")

    rasters = []
    with ExitStack() as opened:
        for key, asset in item.select_data_assets(entry).items():
            asset_names = []
            for name in names_by_asset[key]:
                if name in requested_names:
                    asset_names.append(name)
            if not asset_names:
                continue
            raster = _read_asset_pixels(
                asset, names_by_asset[key], key, asset_names, geobox, opened
            )
            if raster is not None:
                rasters.append(raster)

    # Variables whose files miss the view are NaN, without a read.
    read_names = set()
    for raster in rasters:
        read_names.update(raster.gs.raster_variables)
    missing = {}
    for name in requested_names:
        if name not in read_names:
            missing[name] = (("y", "x"), np.full(geobox.shape, np.nan, "float32"))
    if missing:
        rasters.append(xr.Dataset(missing).gs.write_grid(geobox))
    first = rasters[0]
    dimensions = first[first.gs.raster_variables[0]].dims
    aligned = []
    names = set()
    for raster in rasters:
        if raster.gs.geobox != geobox:
            raise ValueError("asset pixels do not share the requested grid")
        if names.intersection(raster.gs.raster_variables):
            raise ValueError("asset pixels contain duplicate variables")
        names.update(raster.gs.raster_variables)
        if any(raster[name].dims != dimensions for name in raster.gs.raster_variables):
            raise ValueError("asset pixels require equal variable dimensions")
        raster = raster.gs.write_crs()
        if not raster.coords.to_dataset().equals(first.coords.to_dataset()):
            raise ValueError("asset pixels require equal coordinates")
        aligned.append(raster)
    pixels = cast(
        "xr.Dataset",
        xr.merge(
            aligned, join="exact", compat="equals", combine_attrs=attrs.combine_attrs
        ),
    )
    return pixels[requested_names]


def _read_asset_pixels(
    asset: pystac.Asset,
    names: Sequence[str],
    key: str,
    asset_names: list[str],
    geobox: GeoBox,
    opened: ExitStack,
) -> xr.Dataset | None:
    """Read `asset_names` of one asset onto `geobox`; None when the file misses it."""
    import numpy as np
    from odc.geo.overlap import compute_reproject_roi
    from odc.geo.roi import roi_is_empty
    from odc.geo.xr import xr_coords, xr_reproject

    from geosave.geodata.conventions import GRID_MAPPING
    from geosave.geodata.io import rasterio
    from geosave.geodata.stac.extensions import projection

    # 1. A grid the asset states answers a miss without opening the file
    stated = projection.read_geobox(asset)
    if stated is not None and stated.crs is not None:
        if roi_is_empty(compute_reproject_roi(stated, geobox).roi_src):
            return None

    # 2. The base image's header: where the view falls, and how much it shrinks
    raster = _open_asset(asset, names, key, opened, chunks=None)
    roi = compute_reproject_roi(raster.odc.geobox, geobox)
    if roi_is_empty(roi.roi_src):
        return None

    # 3. The coarsest overview whose pixels are still no larger than the view's
    href = asset.get_absolute_href() or asset.href
    overview_level = None
    for level, factor in enumerate(rasterio.read_overview_factors(location(href))):
        if factor <= roi.read_shrink:
            overview_level = level
    if overview_level is not None:
        raster = _open_asset(
            asset, names, key, opened, chunks=None, overview_level=overview_level
        )
        roi = compute_reproject_roi(raster.odc.geobox, geobox)

    # 4. Read only the window, then put it on the view's grid
    window = raster[asset_names].isel(y=roi.roi_src[0], x=roi.roi_src[1])
    window = window.gs._map_variables(mask_missing).astype("float32").load()
    pixels = xr_reproject(window, geobox, resampling="nearest", dst_nodata=np.nan)
    # The file's own CRS attrs survive the warp; the pixels now sit on `geobox`.
    return pixels.odc.assign_crs(geobox.crs).assign_coords(
        {GRID_MAPPING: xr_coords(geobox)[GRID_MAPPING]}
    )


def _open_asset(
    asset: pystac.Asset,
    names: Sequence[str],
    key: str,
    opened: ExitStack,
    **options: Any,
) -> xr.Dataset:
    """Open one asset and apply its band identities and published metadata.

    `options` reach the file's reader, such as `chunks` or `overview_level`.
    """
    from geosave.geodata.io.openers import open_raster

    href = asset.get_absolute_href() or asset.href
    raster = open_raster(location(href), **options)
    opened.callback(raster.close)
    if not isinstance(raster, xr.Dataset):
        raise ValueError(
            f"asset {key!r} must open one raster Dataset; select its subdataset explicitly"
        )

    raster = _place_on_stated_grid(raster, asset, href)
    raster = rename_bands(raster, names, source=href, fallback=key)
    for index, name in enumerate(raster.gs.raster_variables):
        variable = raster[name]
        metadata = band.read_attrs(asset, index)
        raster[name] = variable.assign_attrs({**metadata, **variable.attrs})

    return raster


def _place_on_stated_grid(
    raster: xr.Dataset, asset: pystac.Asset, href: str
) -> xr.Dataset:
    """Give a file without a grid the one its asset states; warn when the two disagree.

    The file's own grid is the truth, as publication reads written headers. A
    stated grid whose shape differs from the file's belongs to other pixels and
    is not assigned.
    """
    import warnings

    from odc.geo.xr import xr_coords

    from geosave.geodata.stac.extensions import projection

    stated = projection.read_geobox(asset)
    if stated is None or stated.crs is None:
        return raster
    own = raster.odc.geobox
    if own is not None and own.crs is not None:
        if own != stated:
            warnings.warn(
                f"{href} states a grid in its file and another in its asset; "
                f"the file's is kept: file {own}, asset {stated}",
                UserWarning,
                stacklevel=3,
            )
        return raster
    if tuple(stated.shape) != (raster.sizes["y"], raster.sizes["x"]):
        return raster
    return raster.assign_coords(xr_coords(stated))


if TYPE_CHECKING:
    from rasterio.io import DatasetReader
    from shapely.geometry.base import BaseGeometry
    from geosave.geodata.stac.extensions.eo import Band
    from geosave.geodata.stac.extensions.raster import RasterBand


def rename_bands(
    raster: xr.Dataset,
    names: Sequence[Hashable],
    *,
    source: str,
    fallback: str | None = None,
) -> xr.Dataset:
    """Name file bands by position, expanding a lone name for a multiband file.

    Args:
        raster: Opened file with variables in stored band order.
        names: Names stated by the file header or catalog.
        source: File location for a band-count error.
        fallback: Prefix for unnamed bands; defaults to the lone stated name.

    Returns:
        Dataset with named variables and unchanged pixels.

    Raises:
        ValueError: The file and stated band counts disagree.
    """
    found = list(raster.gs.raster_variables)
    stated = list(names)
    if len(stated) == 1 and len(found) > 1:
        prefix = stated[0] if fallback is None else fallback
        stated = [f"{prefix}.{index}" for index in range(1, len(found) + 1)]
    if len(found) != len(stated):
        raise ValueError(
            f"{source} holds {len(found)} bands but its metadata names {len(stated)}: {stated}"
        )

    return raster.rename(dict(zip(found, stated, strict=True)))


def _read_observation(
    assets: Mapping[str, xr.Dataset],
    collection: pystac.Collection | None,
    item_options: Mapping[str, Any],
) -> tuple[
    pystac.Collection | None,
    dict[str, Any],
    dict[str, dict[str, object]],
    dict[str, dict[str, Any]],
]:
    """Read shared Item semantics and retain each asset's labels and properties.

    Returns the Collection, constructor options, per-asset scalar labels and
    properties. Only coordinate and attribute metadata is inspected, never
    raster pixels.
    """
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    from geosave.geodata.io.paths import read_scalar_coords
    from geosave.geodata.stac.catalog import merge_collections

    definitions = []
    properties_by_asset = {}
    labels_by_asset = {}
    spans = []
    footprints = []
    for key, raster in assets.items():
        raster.gs.validate(georeferenced=True)
        metadata = attrs.StacMetadata.from_attrs(raster.attrs)
        properties = {}
        if metadata is not None:
            if metadata.collection is not None:
                definitions.append(pystac.Collection.from_dict(metadata.collection))
        properties.update(raster.gs.stac_common_properties)
        properties_by_asset[key] = item.select_item_properties(properties)
        labels_by_asset[key] = read_scalar_coords(raster)
        span = raster.gs.timespan
        if span is not None:
            spans.append(span)
        footprints.append(raster.gs.region.extent.to_crs("EPSG:4326").geom)

    collections = (
        merge_collections(definitions)
        if collection is None
        else {collection.id: collection}
    )
    if len(collections) > 1:
        raise ValueError("one observation cannot belong to multiple Collections")
    owner = next(iter(collections.values()), None)

    options = dict(item_options)
    properties = intersect_mappings(list(properties_by_asset.values()))
    properties.update(options.pop("properties", {}) or {})
    if options.get("input_datetime") is None and not (
        "start_datetime" in properties and "end_datetime" in properties
    ):
        if len(spans) != len(assets):
            raise ValueError(
                "observation time is missing; supply input_datetime or interval properties"
            )
        start = min(span[0] for span in spans)
        end = max(span[1] for span in spans)
        if start == end:
            options["input_datetime"] = start
        else:
            properties["start_datetime"] = start.isoformat()
            properties["end_datetime"] = end.isoformat()

    stated_datetime = properties.pop("datetime", None)
    if options.get("input_datetime") is not None and stated_datetime is not None:
        raise ValueError("supply input_datetime or properties datetime, not both")
    time = item.normalize_time(
        options.get("input_datetime")
        if options.get("input_datetime") is not None
        else stated_datetime,
        properties.pop("start_datetime", None),
        properties.pop("end_datetime", None),
    )
    options["input_datetime"] = time["datetime"]
    start = time["start_datetime"]
    end = time["end_datetime"]
    if start is not None and end is not None:
        properties["start_datetime"] = start.isoformat()
        properties["end_datetime"] = end.isoformat()
    options["properties"] = properties
    options.setdefault("geometry", mapping(unary_union(footprints)))
    footprint = shape(options["geometry"])
    if footprint.is_empty or not footprint.is_valid:
        raise ValueError("Item footprint must be nonempty and valid")
    return owner, options, labels_by_asset, properties_by_asset


def _format_item_id(
    id: str | None,
    template: str | None,
    labels: Mapping[str, object],
    options: Mapping[str, Any],
    dimensions: Sequence[str],
) -> str:
    """Name one observation, with explicit IDs taking precedence over templates."""
    if id is not None:
        return id
    if template is not None:
        return template.format_map(labels)

    instant = options["input_datetime"]
    properties = options["properties"]
    if instant is not None:
        name = instant.isoformat()
    else:
        name = f"{properties['start_datetime']}_{properties['end_datetime']}"

    name = name.replace(":", "")
    for dimension in dimensions:
        if dimension != "time" and dimension in labels:
            name += "_" + str(labels[dimension])
    return name


def publish_items(
    items: Sequence[Mapping[str, xr.Dataset]],
    path: Location,
    *,
    split_item: Sequence[str] = ("time",),
    id: str | None = None,
    format_id: str | None = None,
    format_paths: str | None = None,
    collection: pystac.Collection | None = None,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    cog_options: Mapping[str, Any] | None = None,
    input_datetime=None,
    properties: Mapping[str, Any] | None = None,
    geometry: Mapping[str, Any] | None = None,
    asset_roles: Sequence[str] | None = ("data",),
    asset_media_type: str = "auto",
    asset_hrefs: Mapping[str, str] | None = None,
    with_proj: bool = True,
    with_raster: bool = True,
    with_eo: bool = True,
    geom_densify_pts: int = 0,
    geom_precision: int = -1,
) -> list[pystac.Item]:
    """Preflight an entire publication batch, save COGs, then describe the files.

    `items` holds ``{asset name: Dataset}`` per Item, as `split_items` returns;
    `split_item` names the dimensions that separated them, which a default
    Item ID spells out. No file is written until every ID, destination and
    raster shape is checked.
    """
    from geosave.geodata.io.paths import format_path
    from pystac.utils import is_absolute_href
    from geosave.geodata.stac.extensions import eo
    from geosave.geodata.io.formats import geotiff

    if geom_densify_pts < 0 or geom_precision < -1:
        raise ValueError(
            "geometry densification must be nonnegative and precision at least -1"
        )
    for href in (asset_hrefs or {}).values():
        if not is_absolute_href(href):
            raise ValueError(f"advertised asset href must be absolute: {href!r}")
    if cog_options and "compute" in cog_options:
        raise ValueError("Item publication writes its assets immediately; omit compute")

    item_options = dict(
        input_datetime=input_datetime,
        properties=properties,
        geometry=geometry,
        asset_roles=asset_roles,
        asset_media_type=asset_media_type,
        asset_hrefs=asset_hrefs,
        with_proj=with_proj,
        with_raster=with_raster,
        with_eo=with_eo,
        geom_densify_pts=geom_densify_pts,
        geom_precision=geom_precision,
    )
    if geometry is None:
        item_options.pop("geometry")

    root = location(path, storage_options)
    if root.suffix.lower() in (".tif", ".tiff"):
        raise ValueError(
            "publication path must be a folder; use format_paths for filenames"
        )
    if id is not None and len(items) != 1:
        raise ValueError(
            "an explicit id describes one Item; use format_id for multiple observations"
        )

    files = {}
    plans = []
    identities = set()
    for assets in items:
        owner, options, labels_by_asset, properties_by_asset = _read_observation(
            assets,
            collection,
            item_options,
        )
        labels = intersect_mappings(list(labels_by_asset.values()))
        if owner is not None:
            labels["collection"] = owner.id
        item_id = _format_item_id(id, format_id, labels, options, split_item)
        identity = (None if owner is None else owner.id, item_id)
        if not item_id or identity in identities:
            raise ValueError(
                f"duplicate or empty Item identity {identity!r}; use format_id to distinguish observations"
            )
        identities.add(identity)

        sources = {}
        asset_properties = {}
        spectral_bands = {}
        extensions = []
        for asset, raster in assets.items():
            destination = root / format_path(
                format_paths or "{id}/{asset}.tif",
                raster,
                **labels,
                id=item_id,
                asset=asset,
            )
            sources[asset] = destination
            if destination in files:
                raise ValueError(
                    f"format_paths destinations collide at {destination}; include the split elements"
                )
            files[destination] = raster
            asset_properties[asset] = properties_by_asset[asset]

            if with_eo:
                spectral_bands[asset] = [
                    eo.build_band(raster[name]) for name in raster.gs.raster_variables
                ]
            metadata = attrs.StacMetadata.from_attrs(raster.attrs)
            if metadata is not None:
                for extension in metadata.extensions or ():
                    if extension not in extensions:
                        extensions.append(extension)

        if asset_hrefs and asset_hrefs.keys() - sources.keys():
            raise ValueError("asset_hrefs contains keys absent from sources")

        plans.append(
            (
                item_id,
                sources,
                owner,
                options,
                asset_properties,
                spectral_bands,
                extensions,
            )
        )

    geotiff.write_files(files, overwrite=overwrite, **dict(cog_options or {}))
    entries = []
    for (
        item_id,
        sources,
        owner,
        options,
        asset_properties,
        spectral_bands,
        extensions,
    ) in plans:
        entry = create_stac_item(
            sources, id=item_id, storage_options=storage_options, **options
        )
        if owner is not None:
            entry.set_collection(owner)
        for key, fields in asset_properties.items():
            for name, value in fields.items():
                if entry.properties.get(name) != value:
                    entry.assets[key].extra_fields[name] = value
        for key, bands in spectral_bands.items():
            eo.write_asset(entry.assets[key], bands)
        for extension in extensions:
            if extension not in entry.stac_extensions:
                entry.stac_extensions.append(extension)

        entries.append(entry)
    return entries


def _read_footprint(
    source: DatasetReader, densify: int, precision: int
) -> BaseGeometry:
    """Transform the saved pixel-grid footprint to geographic coordinates."""
    from odc.geo.geobox import GeoBox
    from rasterio.warp import transform_geom
    from shapely.geometry import shape

    grid = GeoBox((source.height, source.width), source.transform, str(source.crs))
    footprint = grid.extent
    if densify:
        length = min(
            abs(source.bounds.right - source.bounds.left),
            abs(source.bounds.top - source.bounds.bottom),
        )
        footprint = footprint.segmented(length / (densify + 1))

    return shape(
        transform_geom(source.crs, "EPSG:4326", footprint.json, precision=precision)
    )


def _read_media_type(source: DatasetReader) -> str | None:
    """Identify GeoTIFF and COG from the saved driver's metadata."""
    if source.driver != "GTiff":
        return None
    if source.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG":
        return pystac.MediaType.COG
    return pystac.MediaType.GEOTIFF


def create_stac_item(
    sources: dict[str, Location],
    *,
    id: str,
    input_datetime: DateTime | None = None,
    properties: dict[str, Any] | None = None,
    geometry: dict[str, Any] | None = None,
    collection: str | None = None,
    collection_url: str | None = None,
    asset_roles: Sequence[str] | None = None,
    asset_media_type: str = "auto",
    asset_hrefs: dict[str, str] | None = None,
    with_proj: bool = False,
    with_raster: bool = False,
    with_eo: bool = False,
    geom_densify_pts: int = 0,
    geom_precision: int = -1,
    storage_options: StorageOptions | None = None,
) -> pystac.Item:
    """Describe named existing raster files as one native STAC Item.

    Args:
        sources: Asset keys mapped to readable local paths or filesystem URLs.
        id: Nonempty Item identifier, unique within its Collection.
        input_datetime: Observation instant. Otherwise properties must contain
            both start_datetime and end_datetime as ISO timestamps.
        properties: Additional STAC Item properties, copied without mutation.
        geometry: Optional longitude/latitude GeoJSON footprint. Otherwise use
            the union of file footprints; no pixels are inspected.
        collection: Optional Collection ID.
        collection_url: Optional URL of the Collection document.
        asset_roles: Roles applied to every asset, e.g. ["data"]. Customize
            individual assets on the returned Item.
        asset_media_type: Media type, or "auto" to recognize GeoTIFF/COG.
        asset_hrefs: Asset keys mapped to absolute advertised href overrides.
        with_proj: Extract each file's grid onto its Asset.
        with_raster: Extract saved dtype, nodata, packing, units and classes per
            band.
        with_eo: Extract known band identities and spectral metadata.
        geom_densify_pts: Extra points per footprint edge before reprojection.
        geom_precision: Decimal places in output geometry; -1 preserves precision.
        storage_options: Filesystem options for reading sources.

    Returns:
        Item with absolute asset hrefs. Additional assets and extensions can be
        added using native PySTAC methods.

    Raises:
        ValueError: Sources, ID, temporal coverage, CRS or hrefs are invalid.
    """
    from contextlib import ExitStack

    import rasterio
    from odc.geo.geobox import GeoBox
    from pystac.utils import is_absolute_href
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    from geosave.geodata.io import storage
    from geosave.geodata.io.rasterio import read_legend
    from geosave.geodata.stac.extensions import (
        classification,
        eo,
        projection,
        raster,
    )

    if not id or not sources:
        raise ValueError("create_stac_item needs a nonempty id and at least one source")
    if geom_densify_pts < 0 or geom_precision < -1:
        raise ValueError(
            "geometry densification must be nonnegative and precision at least -1"
        )
    fields = deepcopy(properties or {})
    stated_datetime = fields.pop("datetime", None)
    if input_datetime is not None and stated_datetime is not None:
        raise ValueError("supply input_datetime or properties datetime, not both")
    time = item.normalize_time(
        input_datetime if input_datetime is not None else stated_datetime,
        fields.pop("start_datetime", None),
        fields.pop("end_datetime", None),
    )

    overrides = dict(asset_hrefs or {})
    if overrides.keys() - sources.keys():
        raise ValueError("asset_hrefs contains keys absent from sources")
    for href in overrides.values():
        if not is_absolute_href(href):
            raise ValueError(f"advertised asset href must be absolute: {href!r}")

    entry = item.create_item(id=id, time=time, properties=fields, collection=collection)
    if collection_url is not None:
        entry.add_link(pystac.Link(pystac.RelType.COLLECTION, collection_url))

    footprints = []
    for key, path in sources.items():
        if not key:
            raise ValueError("asset keys must be nonempty")
        path = storage.location(path, storage_options)
        with ExitStack() as opened:
            address = str(path)
            if not storage.is_local(path):
                address = opened.enter_context(path.open("rb"))
            source = opened.enter_context(rasterio.open(address))
            if source.crs is None:
                raise ValueError(f"source {key!r} has no CRS")

            grid = GeoBox(
                (source.height, source.width), source.transform, str(source.crs)
            )
            footprints.append(_read_footprint(source, geom_densify_pts, geom_precision))
            media_type = asset_media_type
            if media_type == "auto":
                media_type = _read_media_type(source)

            asset = pystac.Asset(
                overrides.get(key, storage.absolute_location(path)),
                media_type=media_type,
                roles=None if asset_roles is None else list(asset_roles),
            )
            entry.add_asset(key, asset)
            if with_proj:
                projection.write_asset(asset, grid)
            if with_raster:
                bands = _read_raster_bands(source)
                for index, band in zip(source.indexes, bands, strict=True):
                    classification.write_band(asset, band, read_legend(source, index))
                raster.write_asset(asset, bands)
            if with_eo:
                identities = _read_eo_bands(source)
                if identities:
                    eo.write_asset(asset, identities)

    footprint = (
        shape(dict(geometry)) if geometry is not None else unary_union(footprints)
    )
    if footprint.is_empty or not footprint.is_valid:
        raise ValueError("Item footprint must be nonempty and valid")
    entry.geometry = mapping(footprint)
    entry.bbox = list(footprint.bounds)
    return entry


def _read_raster_bands(source: DatasetReader) -> list[RasterBand]:
    """Read stored band metadata from an open rasterio dataset, without pixels."""
    from math import isfinite
    from geosave.geodata.stac.extensions.raster import (
        DataType,
        NoDataStrings,
        RasterBand,
    )

    bands = []
    for index in source.indexes:
        position = index - 1
        nodata = source.nodatavals[position]
        if nodata is not None and not isfinite(nodata):
            nodata = NoDataStrings(str(nodata))

        bands.append(
            RasterBand.create(
                data_type=DataType(source.dtypes[position]),
                nodata=nodata,
                scale=source.scales[position],
                offset=source.offsets[position],
                unit=source.units[position],
            )
        )
    return bands


def _read_eo_bands(source: DatasetReader) -> list[Band]:
    """Read known band identities and spectral tags without estimating them.

    Returns:
        One band per file band, or none when a band states no name, since a
        loader then names it by its asset key and position.
    """
    from geosave.geodata.stac.extensions.eo import Band

    bands = []
    for index in source.indexes:
        tags = source.tags(index)
        facts = attrs.Spectral.from_attrs(tags)
        name = tags.get("variable_name") or source.descriptions[index - 1]
        if name is None:
            return []
        bands.append(
            Band.create(
                name=name,
                description=source.descriptions[index - 1],
                common_name=None if facts is None else facts.common_name,
                center_wavelength=None if facts is None else facts.center_wavelength,
                full_width_half_max=None
                if facts is None
                else facts.full_width_half_max,
            )
        )
    return bands


# Root of a lone raster, as DataTree names its own root node.
ROOT_PATH = "."


def split_items(
    rasters: Mapping[str, xr.Dataset],
    split_item: Sequence[str],
    split_asset: Sequence[str],
) -> list[dict[str, xr.Dataset]]:
    """Split rasters into Items, each mapping asset name to the Dataset it writes.

    Args:
        rasters: Node path to raster, with ``"."`` for a lone raster. Nodes
            share observation axes.
        split_item: Dimensions whose positions make separate Items. Each node
            splits along those it has; a node lacking all joins the first Item.
        split_asset: Dimensions whose positions make separate assets within an
            Item, and ``"variable"`` for one asset per data variable.

    Returns:
        One ``{asset name: lazy Dataset}`` per Item, in position order. A lone
        raster's sole asset takes its variable's name, or ``asset`` when it
        holds several. Otherwise a name joins the
        variable and the split labels, below its node path; a node written
        whole is named by its path.

    Raises:
        ValueError: Asset names repeat within an Item.

    Examples:
        >>> items = split_items({".": scene}, ("time",), ("variable",))
        >>> list(items[0])
        ['red', 'nir']
    """
    asset_dims = []
    for name in split_asset:
        if name != "variable":
            asset_dims.append(name)

    # Per Item, each asset's node path, split labels and Dataset, in order.
    labelled_items: list[list[tuple[str, tuple[str, ...], xr.Dataset]]] = []
    for path, raster in rasters.items():
        item_dims = _select_present(split_item, raster)
        for position, observation in enumerate(raster.gs.expand_rasters(item_dims)):
            if position == len(labelled_items):
                labelled_items.append([])

            # Each variable is its own asset, or the observation is written whole.
            pieces: dict[tuple[str, ...], xr.Dataset] = {}
            if "variable" in split_asset:
                for variable in observation.gs.raster_variables:
                    # Dropping the others keeps every coordinate, CF time bounds included.
                    others = []
                    for name in observation.gs.raster_variables:
                        if name != variable:
                            others.append(name)
                    pieces[(str(variable),)] = observation.drop_vars(others)
            else:
                pieces[()] = observation

            for parts, piece in pieces.items():
                dims = _select_present(asset_dims, piece)
                for asset in piece.gs.expand_rasters(dims):
                    labels = list(parts)
                    for dim in dims:
                        labels.append(_format_label(asset, dim))
                    labelled_items[position].append((path, tuple(labels), asset))

    items = []
    for labelled in labelled_items:
        names: dict[str, xr.Dataset] = {}
        for path, labels, asset in labelled:
            name = _name_asset(path, labels, len(labelled), asset)
            if name in names:
                raise ValueError(
                    f"asset name {name!r} repeats within one Item; assign distinct "
                    "coordinate labels or split along another dimension"
                )
            names[name] = asset
        items.append(names)
    return items


def _select_present(dims: Sequence[str], raster: xr.Dataset) -> tuple[str, ...]:
    """Keep the dimensions this raster has, in the order asked."""
    present = []
    for dim in dims:
        if dim in raster.dims:
            present.append(dim)
    return tuple(present)


def _format_label(asset: xr.Dataset, dim: str) -> str:
    """Spell one split label as asset names have it, dates as pandas prints them."""
    coord = asset[dim]
    if coord.dtype.kind == "M":
        # A datetime64 item is a count of nanoseconds, which Timestamp reads.
        return str(pd.Timestamp(coord.item()))
    return str(coord.item())


def _name_asset(
    path: str, labels: tuple[str, ...], count: int, asset: xr.Dataset
) -> str:
    """Name one asset by its node path and labels, unique within its Item.

    A lone raster's sole asset takes its variable's name, or ``asset`` when it
    holds several; a tree node keeps its path.
    """
    name = "_".join(labels)
    if path == ROOT_PATH:
        if count == 1 and len(asset.gs.raster_variables) == 1:
            # A loader names a band by its asset key, so the key keeps the name.
            return str(asset.gs.raster_variables[0])
        if count == 1 or not name:
            return "asset"
        return name
    return f"{path}/{name}" if name else path
