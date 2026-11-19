"""Build native STAC Items from a region and the rasters of its assets."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Any, Self, Unpack, cast
from collections.abc import Mapping

import pandas as pd

import pystac
import xarray as xr
from pystac.utils import datetime_to_str

from geosave.geodata.core.conventions import END_DATETIME, START_DATETIME
from geosave.geodata.core.raster.rules import dataset as dataset_rules
from geosave.geodata.stac.extensions import classification, eo, projection
from geosave.geodata.stac.extensions import raster as raster_extension

if TYPE_CHECKING:
    from geosave.geodata.core.region import GeoRegion
    from geosave.geodata.io.raster.geotiff import COGWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions
    from upath import UPath


class StacItem(pystac.Item):
    """A pystac Item built from a region, with assets added from rasters.

    It is a `pystac.Item` everywhere: `clone`, `from_dict`, extensions and
    validation keep working, and `add_asset` adds any other asset.
    `publish_item` also accepts ordinary native Items.

    Examples:
        >>> item = StacItem.from_region(scene.gs.region, id="20250601")
        >>> item.add_raster_asset("red", red, "s3://bucket/s2/20250601/red.tif")
        <Asset href=s3://bucket/s2/20250601/red.tif>
    """

    @classmethod
    def from_region(
        cls,
        region: GeoRegion,
        *,
        id: str,
        properties: dict[str, Any] | None = None,
    ) -> Self:
        """Create the Item covering a region, without assets.

        The region's extent, in WGS84, is the footprint. An instant becomes
        the Item datetime; a longer timespan becomes `start_datetime` and
        `end_datetime`, with no datetime.

        Args:
            region: Extent and timespan, as `.gs.region` of a raster gives them.
            id: Item id.
            properties: STAC properties to state beside the time.

        Returns:
            Item without assets; add them with `add_raster_asset`.

        Raises:
            ValueError: The region is timeless.

        Examples:
            >>> StacItem.from_region(scene.gs.region, id="20250601").datetime
            datetime.datetime(2025, 6, 1, 10, 30, tzinfo=datetime.timezone.utc)
        """
        if region.timespan is None:
            raise ValueError(f"Item {id!r} needs a region with a timespan")
        start, end = region.timespan
        start = start.replace(tzinfo=UTC)
        end = end.replace(tzinfo=UTC)

        item_properties = dict(properties or {})
        item_datetime = start
        if start != end:
            item_datetime = None
            item_properties[START_DATETIME] = datetime_to_str(start)
            item_properties[END_DATETIME] = datetime_to_str(end)

        footprint = region.extent.to_crs("EPSG:4326")
        return cls(
            id=id,
            geometry=footprint.json,
            bbox=list(footprint.boundingbox),
            datetime=item_datetime,
            properties=item_properties,
        )

    def add_raster_asset(
        self,
        key: str,
        raster: xr.Dataset,
        href: str,
        *,
        roles: list[str] | None = None,
        media_type: str | None = None,
        title: str | None = None,
        with_proj: bool = True,
        with_raster: bool = True,
        with_eo: bool = False,
    ) -> pystac.Asset:
        """Add the asset of one file holding `raster`'s variables, one band each.

        Args:
            key: Asset key.
            raster: The file's Dataset, on its own grid.
            href: Where the file is.
            roles: Asset roles; None is `["data"]`.
            media_type: Media type; None is a Cloud Optimized GeoTIFF.
            title: Human-readable title; None takes the `long_name` of a
                one-variable raster.
            with_proj: State the grid as `proj:*`.
            with_raster: State storage type, fill, packing and units as `raster:bands`.
            with_eo: State each band's spectrum as `eo:bands`, from `EOBand` attrs.

        Returns:
            The asset; the Item declares each extension it uses.

        Examples:
            >>> item.add_raster_asset("red", red, "s3://bucket/red.tif").extra_fields["proj:shape"]
            [2, 2]
        """
        return _add_raster_asset(
            self,
            key,
            raster,
            href,
            roles=roles,
            media_type=media_type,
            title=title,
            with_proj=with_proj,
            with_raster=with_raster,
            with_eo=with_eo,
        )


def _add_raster_asset(
    item: pystac.Item,
    key: str,
    raster: xr.Dataset,
    href: str,
    *,
    roles: list[str] | None = None,
    media_type: str | None = None,
    title: str | None = None,
    with_proj: bool = True,
    with_raster: bool = True,
    with_eo: bool = False,
) -> pystac.Asset:
    """Attach raster metadata to an asset of a native Item."""
    if title is None and len(raster.data_vars) == 1:
        title = next(iter(raster.data_vars.values())).attrs.get("long_name")
    asset = pystac.Asset(
        href,
        title=title,
        media_type=media_type or pystac.MediaType.COG,
        roles=roles or ["data"],
    )
    # An extension declares its schema on the Item an asset belongs to.
    item.add_asset(key, asset)

    if with_proj:
        projection.write_asset(asset, raster.odc.geobox)
    if with_raster:
        raster_bands = []
        for variable in raster.data_vars.values():
            band = raster_extension.build_band(variable)
            classification.write_band(asset, band, variable.gs.flags)
            raster_bands.append(band)
        raster_extension.write_asset(asset, raster_bands)
    if with_eo:
        eo_bands = []
        for variable in raster.data_vars.values():
            eo_bands.append(eo.build_band(variable))
        eo.write_asset(asset, eo_bands)
    return asset


def _timestamp(value: Any) -> pd.Timestamp:
    """Normalize declared observation metadata without changing its instant."""
    stamp = pd.Timestamp(value)
    if stamp is pd.NaT:
        raise ValueError("Observation metadata cannot contain NaT")
    stamp = cast("pd.Timestamp", stamp)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _check_observation(item: pystac.Item, raster: xr.Dataset) -> None:
    """Check available metadata; explicit membership remains a caller assertion."""
    raster.gs.validate()
    if not raster.data_vars:
        raise ValueError("A published asset needs raster variables")
    spatial = raster.gs.spatial_dims or ()
    for variable in raster.data_vars.values():
        if set(variable.dims) != set(spatial):
            raise ValueError(
                "Slice every observation dimension by position before publication"
            )

    labels = {}
    for key in ("time", START_DATETIME, END_DATETIME):
        if key not in raster.coords:
            continue
        if raster[key].ndim != 0:
            raise ValueError(f"Observation coordinate {key!r} must be scalar")
        labels[key] = _timestamp(raster[key].values[()])
    if not labels:
        raise ValueError(
            "Timeless reference rasters cannot be assets of an observation Item"
        )
    if (START_DATETIME in labels) != (END_DATETIME in labels):
        raise ValueError(
            "Observation periods need both start_datetime and end_datetime"
        )

    item_start = item.properties.get(START_DATETIME)
    item_end = item.properties.get(END_DATETIME)
    if (item_start is None) != (item_end is None):
        raise ValueError("Item periods need both start_datetime and end_datetime")
    if item_start is None:
        if item.datetime is None:
            raise ValueError("The Item needs a datetime or an observation period")
        start = end = _timestamp(item.datetime)
    else:
        start, end = _timestamp(item_start), _timestamp(item_end)
    if start > end:
        raise ValueError("The Item observation period is reversed")
    if START_DATETIME in labels:
        if (labels[START_DATETIME], labels[END_DATETIME]) != (start, end):
            raise ValueError("Asset observation period conflicts with the Item")
    if "time" in labels:
        time = labels["time"]
        if not start <= time <= end or (
            item.datetime is not None and time != _timestamp(item.datetime)
        ):
            raise ValueError("Asset observation time conflicts with the Item")


def _check_file_component(name: str) -> None:
    """Keep Item IDs and asset keys as single destination path components."""
    if not name or name in {".", ".."} or any(char in name for char in "/\\"):
        raise ValueError(f"Expected a single file-name component, got {name!r}")


def publish_item(
    item: pystac.Item,
    *,
    assets: Mapping[str, xr.Dataset],
    root: Location,
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **options: Unpack[COGWriteOptions],
) -> pystac.Item:
    """Write explicit observation assets as COGs and return a completed Item.

    Each mapping entry is one file on its own grid. Prepare variables with
    Dataset selection or tree split/merge operations before publishing. Files
    are written to ``<root>/<item.id>/<asset key>.tif``. The input Item and
    rasters remain unchanged; supplied Item properties, geometry and Collection
    links are preserved. No Item JSON or catalog is written.

    Available scalar time/period metadata must agree with the Item. Membership
    asserts that every asset belongs to this observation: copied timestamps
    cannot prove provenance. Independent static/reference products do not belong
    in the asset map. Slice dimension positions explicitly, including length-one
    dimensions. Different assets may use different grids and representations.

    Metadata, representation and destination checks run before any file write.
    Runtime I/O failures can leave earlier files; this is not a transaction.

    Args:
        item: Native Item describing one observation or derived product.
        assets: Asset keys mapped to georeferenced, observation-sliced Datasets.
        root: Destination folder or URL prefix.
        overwrite: Replace existing files when true.
        storage_options: Filesystem options for the destination.
        **options: Native GDAL COG creation options.

    Returns:
        A cloned Item with actual written hrefs and band/grid/class metadata.

    Raises:
        ValueError: Membership, observation metadata or file representation is invalid.
        FileExistsError: A destination exists and overwrite is false.

    Examples:
        >>> published = publish_item(item, assets={"image": image, "scl": scl}, root="out")
        >>> catalog = create_catalog([published])
        >>> catalog.gs.to_geoparquet("out/catalog.parquet")
    """
    from geosave.geodata.io import storage
    from geosave.geodata.io.raster import geotiff

    if not assets:
        raise ValueError("Publication needs at least one raster asset")
    _check_file_component(item.id)
    result = item.clone()
    folder = storage.create_upath(root, storage_options) / item.id
    files = {}
    for key, raster in assets.items():
        _check_file_component(key)
        if key in result.assets:
            raise ValueError(f"Asset {key!r} already exists on the Item")
        _check_observation(item, raster)
        # A multiband file has one storage representation. Refuse promotion or
        # loss of individual encoding recipes before the writer stacks bands.
        raster.gs.validate(
            dataset_rules.same_pixel_meaning, dataset_rules.same_encoding
        )
        first = next(iter(raster.data_vars.values()))
        for variable in raster.data_vars.values():
            if variable.dtype != first.dtype:
                raise ValueError(
                    "One COG needs matching band dtypes; prepare or split assets"
                )
        path = folder / f"{key}.tif"
        files[path] = raster
        _add_raster_asset(result, key, raster, str(path.absolute()), with_eo=True)

    written = cast(
        "list[UPath]",
        geotiff.write_files(files, driver="COG", overwrite=overwrite, **options),
    )
    for key, path in zip(assets, written, strict=True):
        result.assets[key].href = str(path.absolute())
    return result
