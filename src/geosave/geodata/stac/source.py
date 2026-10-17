"""One STAC collection loaded onto an anchor's grid."""

from __future__ import annotations


import copy
import warnings

import odc.stac
import pystac
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime as dt
from typing import TYPE_CHECKING, Any, Literal, Self

from odc.stac.model import PropertyLoadRequest
from pydantic import BaseModel, ConfigDict, Field, model_validator

from geosave import __path__ as _package_paths
import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import SPATIAL_DIMENSIONS, to_yx
from geosave.geodata.warnings import DroppedAttrsWarning
from geosave.geodata.errors import AnchorFetchError
from geosave.geodata.transform.warp import Resampling

from .search.query import StacQuery
from .translate import band, collection
from .extensions import eo

if TYPE_CHECKING:
    import xarray as xr
    from odc.stac.model import ParsedItem

    from geosave.geodata.core import GeoAnchor
    from geosave.geodata.stac.search.api import StacClient
    from geosave.geodata import Dataset, GeoDataFrame

Bands = list[str] | tuple[str, ...]
StacGroupby = Literal["solar_day", "id", "time"]
ChunkSize = int | Literal["auto"]


def _default_chunks() -> dict[str, ChunkSize]:
    """Return the default spatial Dask chunks."""
    return {"x": 1024, "y": 1024}


class StacSourceConfig(BaseModel):
    """How one source loads its pixels.

    Attributes:
        bands: Band names to load. None loads whatever the matched items
            publish.
        groupby: How odc-stac groups scenes along the time axis.
        chunks: Dask chunk sizes for the spatial dims. None loads eagerly.
        resampling: Resampling odc-stac applies, one mode or one per band.
            None keeps each asset's default.
        dtype: Output dtype. None keeps each collection's own.
        nodata: Nodata value, paired with `dtype`. None keeps each asset's own.
        fail_on_error: False skips a scene that fails to read instead of
            raising.
        with_properties: Item properties loaded by ODC along `time`,
            as property names or request mappings with `key` and optional
            `dtype`, `nodata`, `units`, and `fuser` (a supplied `name` must equal `key`), e.g.
            `("view:sun_azimuth", "eo:cloud_cover")`.
        stac_cfg: Per-collection band overrides odc-stac applies, correcting
            nodata, dtype, or asset aliases a catalog publishes wrongly.
        pool: Reader threads odc-stac uses. None takes its default.
        progress: Progress bar odc-stac reports reads through. None is silent.
        patch_url: Called on every asset href before it is read, for a catalog
            whose URLs need signing at load time.
        preserve_original_order: True keeps the matched items in search order
            rather than sorting them by time.

    Examples:
        >>> StacSourceConfig(bands=["B04", "B08"], with_properties=["eo:cloud_cover"])
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    bands: Bands | None = None
    groupby: StacGroupby = "solar_day"
    chunks: dict[str, ChunkSize] | None = Field(default_factory=_default_chunks)
    resampling: Resampling | dict[str, Resampling] | None = None
    dtype: str | None = None
    nodata: float | None = None
    fail_on_error: bool = True
    with_properties: tuple[str | dict[str, Any], ...] | None = None
    stac_cfg: dict[str, Any] | None = None
    pool: int | None = None
    progress: Any | None = None
    patch_url: Callable[[str], str] | None = None
    preserve_original_order: bool = False

    @model_validator(mode="after")
    def exact_property_names(self) -> Self:
        """Keep STAC property identity in coordinate names."""
        for request in PropertyLoadRequest.from_user_input(self.with_properties or ()):
            if request.name is not None and request.name != request.key:
                raise ValueError(
                    "with_properties must keep the exact STAC key as its name"
                )
        return self

    def to_load_kwargs(self) -> dict[str, Any]:
        """Return native odc-stac settings.

        Returns:
            New keyword mapping for `odc.stac.load`, omitting unset options
            except `chunks=None`, which explicitly requests eager loading.
            Property requests are emitted as a list.
        """
        options = self.model_dump(exclude_none=True)
        # None explicitly selects eager loading; ODC otherwise chooses its default.
        options["chunks"] = self.chunks
        if self.with_properties is not None:
            requests = []
            for request in self.with_properties:
                fields = {"key": request} if isinstance(request, str) else dict(request)
                # ODC sanitizes names by default; its native name option preserves the key.
                fields["name"] = fields["key"]
                if units := eo.property_units(fields["key"]):
                    fields.setdefault("units", units)
                    if fields["units"] != units:
                        raise ValueError(f"{fields['key']} uses percent units (%)")
                requests.append(fields)
            options["with_properties"] = requests
        return options


_UNSET: Any = object()


class StacSource:
    """One STAC collection, loaded onto an anchor's own grid.

    Pixels arrive as the provider published them, with no radiometric scaling.
    A fresh source loads on every default; `set_config` and `set_query` return
    a narrowed copy, so one source can seed several.

    Args:
        client: Client used to search the catalog.
        collection: Collection ID to load.

    Examples:
        >>> from geosave.geodata import configure_gdal
        >>> configure_gdal(aws_no_sign_request=True, gdal_http_max_retry=3)
        >>> source = client.source("sentinel-2-l2a").set_config(bands=["B04", "B08"])
        >>> rows = source.search(anchor)              # the Items load reads
        >>> ds = source.load(anchor)
    """

    def __init__(self, client: StacClient, *, collection: str) -> None:
        """Bind the client and collection, on default settings.

        Args:
            client: Client used to search the catalog.
            collection: Collection ID to load.
        """
        self.client = client
        self._collection = collection
        self.config = StacSourceConfig()
        self.query = StacQuery(collections=[collection])

    def set_config(
        self,
        *,
        bands: Bands | None = _UNSET,
        groupby: StacGroupby = _UNSET,
        chunks: dict[str, ChunkSize] | None = _UNSET,
        resampling: Resampling | dict[str, Resampling] | None = _UNSET,
        dtype: str | None = _UNSET,
        nodata: float | None = _UNSET,
        fail_on_error: bool = _UNSET,
        with_properties: Sequence[str | Mapping[str, Any]] | None = _UNSET,
        stac_cfg: dict[str, Any] | None = _UNSET,
        pool: int | None = _UNSET,
        progress: Any | None = _UNSET,
        patch_url: Callable[[str], str] | None = _UNSET,
        preserve_original_order: bool = _UNSET,
    ) -> Self:
        """Return a copy with load settings merged onto its config.

        Only the fields passed change; the rest keep their current value. See
        `StacSourceConfig` for what each field means. This source is unchanged.

        Args:
            bands: Band names to load. None loads whatever the matched items
                publish.
            groupby: How odc-stac groups scenes along the time axis.
            chunks: Dask chunk sizes for the spatial dims. None loads eagerly.
            resampling: Resampling odc-stac applies, one mode or one per band.
                None keeps each asset's default.
            dtype: Output dtype. None keeps each collection's own.
            nodata: Nodata value, paired with `dtype`. None keeps each asset's
                own.
            fail_on_error: False skips a scene that fails to read instead of
                raising.
            with_properties: Item properties loaded as ODC variables along time.
            stac_cfg: Per-collection band overrides odc-stac applies.
            pool: Reader threads odc-stac uses. None takes its default.
            progress: Progress bar odc-stac reports reads through. None is
                silent.
            patch_url: Called on every asset href before it is read.
            preserve_original_order: True keeps the matched items in search
                order rather than sorting them by time.

        Returns:
            New source carrying the merged config.

        Raises:
            ValidationError: A value does not satisfy `StacSourceConfig`.

        Examples:
            >>> red = source.set_config(bands=["B04", "B08"], groupby="time")
        """
        changes: dict[str, Any] = {
            "bands": bands,
            "groupby": groupby,
            "chunks": chunks,
            "resampling": resampling,
            "dtype": dtype,
            "nodata": nodata,
            "fail_on_error": fail_on_error,
            "with_properties": with_properties,
            "stac_cfg": stac_cfg,
            "pool": pool,
            "progress": progress,
            "patch_url": patch_url,
            "preserve_original_order": preserve_original_order,
        }
        updates = {name: v for name, v in changes.items() if v is not _UNSET}
        narrowed = copy.copy(self)
        narrowed.config = type(self.config)(**{**self.config.model_dump(), **updates})
        return narrowed

    def set_query(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        intersects: dict[str, Any] | None = None,
        datetime: dt | str | tuple[dt, dt] | None = None,
        ids: Sequence[str] | None = None,
        filter: str | None = None,
        sortby: Sequence[str] | None = None,
        max_items: int | None = None,
        limit: int | None = None,
    ) -> Self:
        """Return a copy whose search is narrowed beyond each anchor's extent.

        Every call builds the query from scratch on this source's own
        collection. This source is unchanged.

        Args:
            bbox: WGS84 bounding box `(min_lon, min_lat, max_lon, max_lat)`.
            intersects: GeoJSON geometry to intersect.
            datetime: Instant, ISO 8601 string, or `(start, end)` range.
            ids: Item IDs to fetch directly, bypassing spatial and temporal
                narrowing.
            filter: CQL2 text expression, e.g. `"eo:cloud_cover <= 10"`.
            sortby: Field paths, each optionally prefixed `-` for descending
                or `+` for ascending, e.g. `"-properties.eo:cloud_cover"`.
            max_items: Client-side cap on items returned.
            limit: Server page-size hint.

        Returns:
            New source carrying the query.

        Raises:
            ValueError: `bbox` is not a valid WGS84 box, `filter` is not valid
                CQL2 text, `sortby` has an empty field name, or `max_items` or
                `limit` is below one.

        Examples:
            >>> clear = source.set_query(
            ...     filter="eo:cloud_cover <= 10", sortby=["properties.eo:cloud_cover"]
            ... )
        """
        query = StacQuery(
            collections=[self._collection],
            bbox=bbox,
            intersects=intersects,
            datetime=datetime,
            ids=list(ids) if ids is not None else None,
            max_items=max_items,
            limit=limit,
        )
        if filter is not None:
            query = query.set_filter(filter)
        if sortby is not None:
            query = query.sort_by(*sortby)
        narrowed = copy.copy(self)
        narrowed.query = query
        return narrowed

    def __repr__(self) -> str:
        """Describe this source's collection and current load configuration.

        Returns:
            Multi-line text containing the collection ID and every load
            configuration field. This does not search or read pixels.
        """
        lines = "\n".join(
            f"  {key}: {value!r}" for key, value in self.config.model_dump().items()
        )
        return f"{type(self).__name__}\n  collection: {self._collection!r}\n{lines}"

    def sample_item(self) -> pystac.Item | None:
        """Fetch one item from the collection to read its shape.

        Use it to discover asset names and property keys before writing a
        filter.

        Returns:
            One item, or None when the collection is empty.
        """
        found = self.client.search_items(
            StacQuery(collections=[self._collection], max_items=1)
        )
        return found[0] if found else None

    def list_item_properties(self) -> tuple[str, ...]:
        """Name the per-acquisition properties this collection publishes.

        These are the names `StacSourceConfig.with_properties` loads as
        variables, and the columns of `search(anchor)`.

        Returns:
            Property names, sorted. Empty when the collection has no items.

        Examples:
            >>> [p for p in source.list_item_properties() if "sun" in p]
            ['view:sun_azimuth', 'view:sun_elevation']
        """
        sample = self.sample_item()
        return () if sample is None else tuple(sorted(sample.properties))

    @property
    def collection(self) -> pystac.Collection:
        """Return the Collection this source loads, as its catalog publishes it."""
        return self.client.collection(self._collection)

    def search(self, anchor: GeoAnchor) -> GeoDataFrame:
        """List the Items `load(anchor)` reads, as an item table.

        Args:
            anchor: Grid and datetime window to search.

        Returns:
            One row per matching Item, read through `gs` as a catalog. Empty
            where nothing matches.

        Examples:
            >>> source.search(anchor)["eo:cloud_cover"].tolist()
            [4.2, 11.0]
        """
        return self.client.search(self._search_query(anchor))

    def load(self, anchor: GeoAnchor) -> Dataset:
        """Load this collection onto an anchor's grid.

        Collection metadata becomes the root's ACDD attrs, read back with
        `ds.gs.acdd`. Each variable carries the loader's nodata plus the
        packing, units, spectral facts and legend its band states. Pixels stay
        the published digital numbers.

        Args:
            anchor: Grid and datetime window to load.

        Returns:
            Dataset on the anchor's exact geobox. Its arrays are
            Dask-backed unless `chunks=None` was configured. ODC owns the time
            axis, grouping, and requested property variables. Search limits do
            not squeeze dimensions, and composite bounds are not reconstructed.

        Raises:
            AnchorFetchError: The search matched no items.
            ValueError: The loaded Dataset carries no locatable grid.

        Warns:
            DroppedAttrsWarning: Contributing Items disagree on band metadata.
                Conflicting source attributes are omitted; loader metadata remains
                authoritative and published digital numbers are not decoded.

        Examples:
            >>> ds = source.load(anchor)
            >>> ds.attrs["title"]
            'Sentinel-2 Level-2A'
        """
        matched = self.client.search_items(self._search_query(anchor))
        if not matched:
            raise AnchorFetchError(
                f"no {self._collection!r} items matched the anchor's extent and window"
            )

        loaded = odc.stac.load(
            matched, geobox=anchor.geobox, **self.config.to_load_kwargs()
        )
        data = self._apply_metadata(matched, to_yx(loaded))
        return data.gs.write_crs()

    def _apply_metadata(
        self, items: Sequence[pystac.Item], loaded: xr.Dataset
    ) -> xr.Dataset:
        """Apply agreed source metadata, preserving the loader's pixel metadata."""
        if not items:
            raise ValueError("reading a STAC load's attrs needs at least one item")

        parsed = list(odc.stac.parse_items(items, cfg=self.config.stac_cfg))
        described = {}
        for name, variable in loaded.data_vars.items():
            # ODC property variables retain their own values, dimensions, and attrs.
            if not set(SPATIAL_DIMENSIONS) <= set(variable.dims):
                continue
            metadata = self._read_band_attrs(items, parsed, str(name), variable)
            stated = variable.assign_attrs({**metadata, **variable.attrs})

            # ODC states its fill as `nodata`; normalize it to CF's spelling.
            nodata = attrs.Nodata.from_attrs(stated.attrs)
            described[name] = (
                stated if nodata is None else attrs.stamp_attrs(stated, nodata)
            )

        return loaded.assign(described).assign_attrs(
            collection.read_attrs(self.collection)
        )

    @staticmethod
    def _read_band_attrs(
        items: Sequence[pystac.Item],
        parsed: Sequence[ParsedItem],
        name: str,
        variable: xr.DataArray,
    ) -> dict[str, Any]:
        """Resolve an ODC band and keep only metadata all contributing Items agree on."""
        published = []
        for entry, odc_item in zip(items, parsed, strict=True):
            key, index = odc_item.collection.band_key(name)
            asset = entry.assets.get(key)
            published.append(
                {} if asset is None else band.read_attrs(variable, asset, index - 1)
            )

        kept = attrs.intersect(published)
        stated_keys = set()
        for fields in published:
            stated_keys.update(fields)
        disputed = sorted(stated_keys - kept.keys())
        if disputed:
            values = {}
            for key in disputed:
                values[key] = {
                    entry.id: fields.get(key)
                    for entry, fields in zip(items, published, strict=True)
                }
            warnings.warn(
                f"STAC items publish {disputed} differently for {name!r}, so "
                f"those source attrs are dropped: {values}",
                DroppedAttrsWarning,
                skip_file_prefixes=tuple(_package_paths),
            )

        return kept

    def _search_query(self, anchor: GeoAnchor) -> StacQuery:
        """Narrow this source's search to one anchor.

        Explicit item IDs bypass target search selectors. Otherwise the target
        supplies only an omitted spatial selector and datetime window.

        Args:
            anchor: Grid and datetime window to load.

        Returns:
            Query narrowed to the anchor.
        """
        if self.query.ids is not None:
            return self.query

        query = self.query
        if query.bbox is None and query.intersects is None:
            bounds = anchor.geobox.geographic_extent.boundingbox
            query = replace(query, bbox=tuple(bounds))
        if query.datetime is None:
            query = replace(query, datetime=anchor.timespan)
        return query
